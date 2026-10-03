"""Offline ledger tests: durability, caps, single dispatch, replay and tamper detection."""
from __future__ import annotations

from decimal import Decimal
import json

import pytest

from selfref_scaling import instruments_a1 as I
from selfref_scaling import paid
from selfref_scaling.common import canonical, digest

PRICES = {"gpt-6-astra": ["10", "50"], "claude-opus-5-5": ["4", "20"]}


def binding(max_attempts=2, **extra):
    return {"schema": paid.SCHEMA, "purpose": "test", "prices_per_million": PRICES,
            "max_attempts": max_attempts, **extra}


def evaluate(call, raw):
    text = I.extract(call["provider"], raw)
    if text == "ok":
        return {"status": "ok", "text": text}
    return {"status": "schema_failure", "retryable": text == "bad"}


def raw(text="ok", model="gpt-6-astra", output_tokens=10, **extra):
    return {"id": "resp_1", "model": model, "status": "completed", "created_at": 1.5,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 100, "output_tokens": output_tokens, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 2}}, **extra}


def spec(slot="s1", attempt=0, model="gpt-6-astra", provider="openai", content="hello"):
    return {"slot": slot, "attempt": attempt, "provider": provider, "model": model,
            "request": {"model": model, "input": content, "max_output_tokens": 100}, "context": {"slot": slot}}


def open_ledger(root, cap=Decimal("10"), **kwargs):
    return paid.Ledger(root, evaluate=evaluate, binding=binding(**kwargs), cap=cap)


def reservation(content="hello", model="gpt-6-astra"):
    return paid.reservation_usd(PRICES[model], spec(content=content, model=model)["request"])


def write_events(root, pairs):
    """A journal with a valid hash chain around arbitrary content: a consistent forgery."""
    lines, previous = [], None
    for seq, (kind, data) in enumerate(pairs, 1):
        row = {"seq": seq, "previous": previous, "utc": "2026-10-03T00:00:00+00:00", "kind": kind, "data": data}
        row["sha256"] = digest(row)
        previous = row["sha256"]
        lines.append(canonical(row) + "\n")
    (root / paid.JOURNAL).write_text("".join(lines))


def pairs(root):
    return [(e["kind"], e["data"]) for e in paid.read_journal(root)]


def test_reservation_is_durable_before_the_single_dispatch(tmp_path):
    seen = []

    def send(provider, request):
        events = paid.read_journal(tmp_path)
        seen.append([e["kind"] for e in events])
        assert provider == "openai" and events[-1]["data"]["request"] == request
        request["input"] = "mutated by the sender"
        return raw()

    with open_ledger(tmp_path) as ledger:
        [reserved] = ledger.reserve(spec())
        assert reserved["reservation_usd"] == paid.usd(reservation()) and reserved["spent_before_usd"] == "0"
        assert ledger.spent() == reservation()
        result = ledger.dispatch(reserved["attempt_id"], send)
        with pytest.raises(paid.Halted):
            ledger.dispatch(reserved["attempt_id"], send)
        with pytest.raises(paid.Halted):
            ledger.reserve(spec())
        assert ledger.attempts("s1")[0]["request"]["input"] == "hello"
    assert seen == [["binding", "session", "reserve"]]
    assert result["status"] == "ok" and result["cost_known"] and not result["fatal"]
    assert Decimal(result["cost_usd"]) == (Decimal(100) * 10 * Decimal("1.25") + 10 * 50) / 10**6
    with paid.Ledger(tmp_path, evaluate=evaluate) as reader:
        assert reader.spent() == Decimal(result["cost_usd"])
        assert [r["status"] for r in reader.results()] == ["ok"]
        with pytest.raises(paid.Halted):
            reader.reserve(spec("s2"))


def test_cap_refusal_is_atomic_and_appends_nothing(tmp_path):
    calls = []
    one = reservation()
    with open_ledger(tmp_path, cap=one * 2 - Decimal("0.000001")) as ledger:
        before = (tmp_path / paid.JOURNAL).read_bytes()
        with pytest.raises(paid.BudgetExceeded):
            ledger.reserve(spec("a"), spec("b"))
        assert (tmp_path / paid.JOURNAL).read_bytes() == before
        [first] = ledger.reserve(spec("a"))
        with pytest.raises(paid.BudgetExceeded):
            ledger.reserve(spec("b"))  # In-flight reservations count at full value.
        ledger.dispatch(first["attempt_id"], lambda p, r: calls.append(r) or raw())
        ledger.reserve(spec("b"))  # Actual usage released the unused reservation.
    assert len(calls) == 1


def test_refusal_is_final_only_after_in_flight_calls_settle(tmp_path):
    one = reservation()
    with open_ledger(tmp_path, cap=one * Decimal("1.5")) as ledger:
        ledger.reserve(spec("a"))
        settled = []

        def settle():
            pending = ledger.unresolved()
            for attempt_id in pending:
                ledger.dispatch(attempt_id, lambda p, r: raw())
            settled.append(len(pending))
            return bool(pending)

        [second] = paid.reserve_in_order(ledger, [spec("b")], settle, lambda: False)
        assert second["slot"] == "b" and settled == [1]  # "a" held its full reservation until it resolved.
        assert paid.reserve_in_order(ledger, [spec("c")], settle, lambda: True) is None
        with pytest.raises(paid.BudgetExceeded):
            paid.reserve_in_order(ledger, [spec("c", content="x" * 4000)], settle, lambda: False)
        assert settled == [1, 1, 0] and ledger.spent() <= one * Decimal("1.5")


def test_callable_and_invalid_caps(tmp_path):
    cap = {"value": Decimal("0.01")}
    with open_ledger(tmp_path, cap=lambda: cap["value"]) as ledger:
        with pytest.raises(paid.BudgetExceeded):
            ledger.reserve(spec())
        cap["value"] = Decimal("1")
        [reserved] = ledger.reserve(spec())
        assert reserved["cap_usd"] == "1"
        cap["value"] = 1.0
        with pytest.raises(ValueError):
            ledger.reserve(spec("s2"))
    with paid.Ledger(tmp_path / "x", evaluate=evaluate, binding=binding(), cap=None) as ledger:
        with pytest.raises(paid.Halted):
            ledger.reserve(spec())


def test_transport_failure_charges_full_reservation_and_halts(tmp_path):
    class ProviderError(Exception):
        status_code = 503

    def send(provider, request):
        raise ProviderError("SYNTHETIC_CREDENTIAL_TEXT_should_never_be_stored")

    with open_ledger(tmp_path) as ledger:
        [reserved] = ledger.reserve(spec())
        result = ledger.dispatch(reserved["attempt_id"], send)
        assert (result["status"], result["fatal"], result["cost_known"]) == ("transport_unknown", True, False)
        assert result["cost_usd"] == reserved["reservation_usd"] and result["http_status"] == 503
        assert result["error_type"] == "ProviderError" and ledger.next_attempt("s1") is None
        with pytest.raises(paid.Halted):
            ledger.reserve(spec("s2"))
    assert b"SYNTHETIC_CREDENTIAL_TEXT" not in (tmp_path / paid.JOURNAL).read_bytes()
    with open_ledger(tmp_path) as ledger:
        assert ledger.spent() == reservation()
        with pytest.raises(paid.Halted):
            ledger.require_resolved()
        with pytest.raises(paid.Halted):
            ledger.reserve(spec("s2"))


def test_non_dictionary_or_unserializable_reply_is_a_transport_failure(tmp_path):
    with open_ledger(tmp_path) as ledger:
        [a] = ledger.reserve(spec("a"))
        assert ledger.dispatch(a["attempt_id"], lambda p, r: "text")["status"] == "transport_unknown"
    with open_ledger(tmp_path / "b") as ledger:
        [b] = ledger.reserve(spec("b"))
        assert ledger.dispatch(b["attempt_id"], lambda p, r: {"x": {1, 2}})["error_type"] == "TypeError"


def test_unresolved_reservation_refuses_resume_and_dispatch(tmp_path):
    with open_ledger(tmp_path) as ledger:
        [reserved] = ledger.reserve(spec())  # Crash: no result is ever recorded.
    with open_ledger(tmp_path) as ledger:
        assert ledger.unresolved() == ["s1#0"]
        with pytest.raises(paid.Halted):
            ledger.require_resolved()
        with pytest.raises(paid.Halted):
            ledger.reserve(spec("s2"))
        with pytest.raises(paid.Halted):
            ledger.dispatch(reserved["attempt_id"], lambda p, r: raw())
    assert pairs(tmp_path)[-1] == ("session", {"unresolved": ["s1#0"]})
    # A forged journal that dispatches after that boundary does not replay.
    events = pairs(tmp_path)
    write_events(tmp_path, events + [events[2]])
    with pytest.raises(paid.Halted, match="unresolved"):
        open_ledger(tmp_path).__enter__()


def test_torn_tail_chain_and_framing_are_detected(tmp_path):
    with open_ledger(tmp_path) as ledger:
        [reserved] = ledger.reserve(spec())
        ledger.dispatch(reserved["attempt_id"], lambda p, r: raw())
    journal = tmp_path / paid.JOURNAL
    good = journal.read_bytes()
    lines = good.splitlines(keepends=True)
    variants = {
        "torn": good + b'{"seq":6',
        "edited": good.replace(b'"input":"hello"', b'"input":"hellp"'),
        "duplicated": good + lines[-1],
        "removed": b"".join(lines[:2] + lines[3:]),
        "noncanonical": lines[0].replace(b'"kind":', b'"kind": ') + b"".join(lines[1:]),
        "blank": good + b"\n",
        "nan": good.replace(b'"created_at":1.5', b'"created_at":NaN'),
    }
    for name, data in variants.items():
        journal.write_bytes(data)
        with pytest.raises(paid.Halted):
            open_ledger(tmp_path).__enter__()
        assert journal.read_bytes() == data, name  # Never repaired or truncated.
    journal.write_bytes(good)
    with open_ledger(tmp_path) as ledger:
        assert ledger.spent() > 0


def test_consistent_forgeries_do_not_replay(tmp_path):
    with open_ledger(tmp_path) as ledger:
        [reserved] = ledger.reserve(spec())
        ledger.dispatch(reserved["attempt_id"], lambda p, r: raw())
    events = pairs(tmp_path)
    kinds = [k for k, _ in events]
    assert kinds == ["binding", "session", "reserve", "result"]
    reserve, result = events[2][1], events[3][1]

    def forged(index, **changes):
        copy = json.loads(json.dumps(events))
        copy[index][1].update(changes)
        return [tuple(e) for e in copy]

    cases = [
        forged(3, cost_usd="0.000001"),                       # cost does not reconstruct from usage
        forged(3, status="schema_failure"),                   # status does not reconstruct
        forged(3, evaluated={"status": "ok", "text": "no"}),  # stored evaluation differs from replay
        forged(3, returned_model="gpt-6-astra-2026-01-01"),
        forged(2, reservation_usd="0.01"),                    # reservation is not the request's bound
        forged(2, spent_before_usd="1"),
        forged(2, cap_usd="0.01"),                            # reserved beyond its recorded cap
        forged(2, attempt=1, attempt_id="s1#1"),              # unlicensed retry
        events[:4] + [("result", result)],                    # duplicate result
        events[:3] + [("reserve", reserve)],                  # duplicate reservation
        [events[0], ("result", result)],                      # result without reservation
        events[:2] + [("session", {"unresolved": ["s1#0"]})],  # wrong session boundary
        [("session", {"unresolved": []})] + events[1:],       # no binding first
        events + [("unknown", {})],
    ]
    for case in cases:
        write_events(tmp_path, case)
        with pytest.raises(paid.Halted):
            open_ledger(tmp_path).__enter__()
    write_events(tmp_path, events)
    with paid.Ledger(tmp_path, evaluate=evaluate) as reader:
        assert reader.final("s1")["status"] == "ok"
    with pytest.raises(paid.Halted, match="another binding"):
        paid.Ledger(tmp_path, evaluate=evaluate, binding=binding(purpose="other"), cap=Decimal(1)).__enter__()


def test_model_drift_and_mismatch_are_fatal(tmp_path):
    with open_ledger(tmp_path) as ledger:
        a, b = ledger.reserve(spec("a"), spec("b"))
        assert ledger.dispatch(a["attempt_id"], lambda p, r: raw())["status"] == "ok"
        drifted = ledger.dispatch(b["attempt_id"], lambda p, r: raw(model="gpt-6-astra-2026-09-30"))
        assert (drifted["status"], drifted["fatal"], drifted["model_drift"]) == ("model_drift", True, True)
        with pytest.raises(paid.Halted):
            ledger.reserve(spec("c"))
    with paid.Ledger(tmp_path, evaluate=evaluate) as reader:
        assert [r["status"] for r in reader.results()] == ["ok", "model_drift"]
    with open_ledger(tmp_path / "wrong") as ledger:
        [c] = ledger.reserve(spec("c"))
        assert ledger.dispatch(c["attempt_id"], lambda p, r: raw(model="gpt-5"))["status"] == "model_drift"
    with open_ledger(tmp_path / "anthropic") as ledger:
        [d] = ledger.reserve(spec("d", provider="anthropic", model="claude-opus-5-5"))
        reply = {"id": "msg", "model": "claude-opus-5-5", "stop_reason": "end_turn",
                 "content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 5, "output_tokens": 5}}
        assert ledger.dispatch(d["attempt_id"], lambda p, r: reply)["status"] == "ok"


def test_unknown_usage_and_overspend_are_fatal(tmp_path):
    with open_ledger(tmp_path) as ledger:
        [a] = ledger.reserve(spec("a"))
        result = ledger.dispatch(a["attempt_id"], lambda p, r: {**raw(), "usage": None})
        assert (result["status"], result["cost_known"], result["evaluated"]) == ("usage_failure", False, None)
        assert result["cost_usd"] == a["reservation_usd"]
    with open_ledger(tmp_path / "over") as ledger:
        [b] = ledger.reserve(spec("b"))
        result = ledger.dispatch(b["attempt_id"], lambda p, r: raw(output_tokens=10_000))
        assert result["status"] == "budget_contract_failure" and result["fatal"]
        assert Decimal(result["cost_usd"]) > Decimal(b["reservation_usd"])
        assert ledger.spent() == Decimal(result["cost_usd"])
    with open_ledger(tmp_path / "crash") as ledger:
        [c] = ledger.reserve(spec("c"))
        result = ledger.dispatch(c["attempt_id"], lambda p, r: {**raw(), "output": [None]})
        assert result["status"] == "evaluation_error" and result["fatal"]
    with paid.Ledger(tmp_path / "crash", evaluate=evaluate) as reader:
        assert reader.final("c")["evaluated"] == {"status": "evaluation_error", "error_type": "AttributeError"}


def test_schema_retry_is_licensed_once(tmp_path):
    with open_ledger(tmp_path) as ledger:
        with pytest.raises(paid.Halted):
            ledger.reserve(spec(attempt=1))
        [first] = ledger.reserve(spec())
        assert ledger.dispatch(first["attempt_id"], lambda p, r: raw("bad"))["status"] == "schema_failure"
        assert ledger.next_attempt("s1") == 1
        [second] = ledger.reserve(spec(attempt=1))
        ledger.dispatch(second["attempt_id"], lambda p, r: raw("bad"))
        assert ledger.next_attempt("s1") is None
        with pytest.raises(paid.Halted):
            ledger.reserve(spec(attempt=2))
        [other] = ledger.reserve(spec("s2"))
        ledger.dispatch(other["attempt_id"], lambda p, r: raw("final"))  # Not retryable.
        assert ledger.next_attempt("s2") is None
        assert [a["attempt"] for a in ledger.attempts("s1")] == [0, 1]
    with open_ledger(tmp_path / "single", max_attempts=1) as ledger:
        [only] = ledger.reserve(spec())
        ledger.dispatch(only["attempt_id"], lambda p, r: raw("bad"))
        assert ledger.next_attempt("s1") is None
    with open_ledger(tmp_path) as ledger:  # Replay keeps the licenses.
        assert ledger.next_attempt("s1") is None and len(list(ledger.results(start=1))) == 2


def test_missing_slots_need_a_failed_dependency(tmp_path):
    with open_ledger(tmp_path, max_attempts=1) as ledger:
        a, b = ledger.reserve(spec("a"), spec("b"))
        ledger.dispatch(a["attempt_id"], lambda p, r: raw())
        ledger.dispatch(b["attempt_id"], lambda p, r: raw("final"))
        with pytest.raises(ValueError):
            ledger.missing("dep-a", "source_not_ok", "a")
        with pytest.raises(ValueError):
            ledger.missing("dep-x", "source_not_ok", "nowhere")
        ledger.missing("dep-b", "source_not_ok", "b")
        assert ledger.missing_record("dep-b")["dependency_status"] == "schema_failure"
        assert ledger.next_attempt("dep-b") is None
        with pytest.raises(paid.Halted):
            ledger.missing("dep-b", "source_not_ok", "b")
        with pytest.raises(paid.Halted):
            ledger.reserve(spec("dep-b"))
    with paid.Ledger(tmp_path, evaluate=evaluate) as reader:
        assert reader.slots() == ["a", "b", "dep-b"] and reader.spent() > 0


def test_process_lock_and_external_writes(tmp_path):
    with open_ledger(tmp_path) as ledger:
        with pytest.raises(paid.Halted, match="Another process"):
            open_ledger(tmp_path).__enter__()
        with (tmp_path / paid.JOURNAL).open("ab") as handle:
            handle.write(b"{}\n")
        with pytest.raises(paid.Halted, match="outside this process"):
            ledger.reserve(spec())
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError):
        open_ledger(tmp_path / "link").__enter__()


def test_binding_and_spec_validation(tmp_path):
    for bad in ({"max_attempts": 3}, {"prices_per_million": {"m": ["1"]}}, {"prices_per_million": {"m": [1, 2]}},
                {"schema": "other"}, {"purpose": ""}):
        with pytest.raises(ValueError):
            paid.Ledger(tmp_path / "b", evaluate=evaluate, binding={**binding(), **bad}, cap=Decimal(1)).__enter__()
    with open_ledger(tmp_path) as ledger:
        for bad in ({**spec(), "extra": 1}, {**spec(), "model": "unpriced"}, {**spec(), "slot": "a#b"},
                    {**spec(), "request": {"model": "claude-opus-5-5", "max_output_tokens": 1}},
                    {**spec(), "request": {"model": "gpt-6-astra", "max_output_tokens": 0}},
                    {**spec(), "context": {"x": float("nan")}}, {**spec(), "provider": "google"}):
            with pytest.raises((ValueError, TypeError)):
                ledger.reserve(bad)
        with pytest.raises(ValueError):
            ledger.reserve(spec("a"), spec("a"))
    assert [k for k, _ in pairs(tmp_path)] == ["binding", "session"]


def test_money_helpers():
    assert paid.usd(Decimal("0.0659850")) == "0.065985" and paid.usd(Decimal("1E+2")) == "100"
    for bad in (1.0, True, "nan", "-1", "Infinity", None):
        with pytest.raises(ValueError):
            paid.money(bad)
