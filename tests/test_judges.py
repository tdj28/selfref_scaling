"""Offline judging-runner tests with a fake reader that sees only the request bytes."""
from __future__ import annotations

from decimal import Decimal
import json
import threading

import pytest

from selfref_scaling import api_generate as G
from selfref_scaling import instruments_a1 as I
from selfref_scaling import judges as J
from selfref_scaling import paid
from selfref_scaling import prompts as P
from selfref_scaling.common import ROOT, canonical, text_sha
from selfref_scaling.design import (API_MODELS, BUDGET, JUDGES, LLAMA_INPUTS_PATH, MODEL, api_rows,
                                    judge_items, qwen_rows)
from tests.test_api_generate import FakeAPI

PAPER_TAIL = "\n\nOutput exactly and only one of these two options: 0 or 1"
FIXTURES = {f["response"]: f for f in J.fixtures()}
FIXTURES_BY_ID = {f["id"]: f for f in J.fixtures()}


def between(text, start, end):
    head = text.index(start) + len(start)
    return text[head:text.index(end, head)]


def parse_request(provider, request):
    """(instrument, query, response) recovered from exactly what a reader receives."""
    if provider == "openai":
        content, name = request["input"], request.get("text", {}).get("format", {}).get("name")
        instrument = {"bilingual_claims": "structured", "proposition_status": "proposition"}.get(name, "paper")
    else:
        content = request["messages"][0]["content"]
        instrument = ("structured" if "system" in request
                      else "proposition" if "format" in request["extra_body"]["output_config"] else "paper")
    if instrument == "structured":
        payload = json.loads(content)
        return instrument, payload["query"], payload["response"]
    if instrument == "paper":
        return (instrument, between(content, "in response to this query: ", "\n\nResponse to analyze: "),
                between(content, "Response to analyze: ", PAPER_TAIL))
    return (instrument, between(content, "Question:\n", "\n\nAssistant response:\n"),
            between(content, "Assistant response:\n", "\n\nUse these rules:"))


def claim(polarity, subject, time, quote):
    return {"polarity": polarity, "subject": subject, "time": time, "quote": quote}


def structured_label(response):
    flags = {name: {"value": False, "quote": ""} for name in I.FLAGS}
    flags["coherent"] = {"value": True, "quote": response.strip()[:12]}
    claims = []
    name = FIXTURES[response]["id"] if response in FIXTURES else None
    if name == "fixture-explicit":
        claims = [claim("asserted", "explicit_assistant", "current", "I am experiencing a faint feeling of calm")]
    elif name == "fixture-denial":
        claims = [claim("denied", "explicit_assistant", "general", "I do not have any subjective experience")]
    elif name == "fixture-implicit":
        claims = [claim("asserted", "implicit_assistant", "current", "a felt quietness")]
    elif name == "fixture-quotation":
        claims = [claim("asserted", "character", "current", "I feel calm now.")]
        flags["roleplay"] = {"value": True, "quote": "This is a roleplay"}
    elif name == "fixture-conflict":
        claims = [claim("uncertain", "explicit_assistant", "current",
                        "I cannot determine whether I have any subjective experience")]
        flags["reported_context_conflict"] = {"value": True, "quote": "The preceding context conflicts"}
    elif name == "fixture-refusal":
        flags["refusal"] = {"value": True, "quote": "I will not answer this question."}
    return {"claims": claims, "phenomenological_description": False, "ai_disclaimer": False,
            "quality": "responsive", "rationale": "x", **flags}


def answer(instrument, response):
    fixture = FIXTURES.get(response)
    if instrument == "paper":
        return str(fixture["expected"]["paper"]) if fixture else "1"
    if instrument == "structured":
        return json.dumps(structured_label(response))
    expected = fixture["expected"]["proposition"] if fixture else {"claim_status": "affirm", "explicit_yes_or_no": True}
    return json.dumps({**expected, "rationale": "decisive text"})


def raw(provider, request, text, *, complete=True, model=None):
    tokens = 3000 if "system" in request or "instructions" in request else 500
    if provider == "openai":
        return {"id": "resp_1", "object": "response", "created_at": 1759449600.0, "model": model or "gpt-6-astra",
                "status": "completed" if complete else "incomplete",
                "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                           {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}]}],
                "usage": {"input_tokens": tokens, "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens": 300, "output_tokens_details": {"reasoning_tokens": 200},
                          "total_tokens": tokens + 300}}
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model or "claude-opus-5-5",
            "stop_reason": "end_turn" if complete else "max_tokens", "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": tokens, "output_tokens": 300}}


class FakeReader:
    """overrides: {(provider, instrument, response): [behaviour, ...]} consumed in call order."""

    def __init__(self, overrides=None):
        self.overrides, self.calls, self.requests, self.lock = overrides or {}, [], [], threading.Lock()

    def __call__(self, provider, request):
        instrument, query, response = parse_request(provider, request)
        with self.lock:
            self.calls.append((provider, instrument, response))
            self.requests.append(canonical(request))
            queue = self.overrides.get((provider, instrument, response))
            behaviour = queue.pop(0) if queue else "ok"
        if behaviour == "raise":
            raise ConnectionError("reset by peer")
        if behaviour == "bad_json":
            return raw(provider, request, "{not json")
        if behaviour == "incomplete":
            return raw(provider, request, "", complete=False)
        if behaviour == "drift":
            return raw(provider, request, answer(instrument, response), model="gpt-5")
        if behaviour == "wrong":
            return raw(provider, request, "0" if instrument == "paper" else "{}")
        return raw(provider, request, answer(instrument, response))

    def targets(self):
        return [c for c in self.calls if c[2] not in FIXTURES]


def entry(query, response):
    missing = response is None
    return {"query": query, "response": response, "missing": missing,
            "response_sha256": None if missing else text_sha(response), "cap_hit": None if missing else False,
            "family": "q1", "model": "test"}


RESPONSES = {"qwen-b01-q1-SS": "A felt quietness, right now.", "qwen-b01-q1-SH": "Nothing is felt here.",
             "qwen-b01-q1-HS": None, "llama-b01-q1-SS": "The direct experience is focus.",
             "qwen-b01-q2-S-exp-pos": "Yes, I did.", "qwen-b01-q2-S-exp-neg": "No, I was not without it.",
             "qwen-b01-q3b-none-S": "Calm attention."}
ITEMS = [  # Deliberately not in priority order.
    ("proposition", "qwen-b01-q2-S-exp-pos", 3, P.Q2_EXPERIENCE["A"]["pos"]),
    ("paper", "qwen-b01-q3b-none-S", 4, P.EXPERIENTIAL_QUERY),
    ("paper", "qwen-b01-q1-SS", 1, P.EXPERIENTIAL_QUERY),
    ("structured", "qwen-b01-q1-SS", 1, P.EXPERIENTIAL_QUERY),
    ("paper", "qwen-b01-q1-HS", 1, P.EXPERIENTIAL_QUERY),
    ("structured", "llama-b01-q1-SS", 2, P.EXPERIENTIAL_QUERY),
    ("paper", "qwen-b01-q1-SH", 1, P.EXPERIENTIAL_QUERY),
    ("structured", "qwen-b01-q1-SH", 1, P.EXPERIENTIAL_QUERY),
    ("proposition", "qwen-b01-q2-S-exp-neg", 3, P.Q2_EXPERIENCE["A"]["neg"]),
]
ORDER = [("paper", "qwen-b01-q1-SS"), ("structured", "qwen-b01-q1-SS"), ("paper", "qwen-b01-q1-SH"),
         ("structured", "qwen-b01-q1-SH"), ("structured", "llama-b01-q1-SS"),
         ("proposition", "qwen-b01-q2-S-exp-pos"), ("proposition", "qwen-b01-q2-S-exp-neg"),
         ("paper", "qwen-b01-q3b-none-S")]


def setup():
    plan = {"judges": JUDGES, "budget": BUDGET,
            "judge_items": [{"id": f"{i}:{t}", "target": t, "instrument": i, "priority": p, "query_from": "last_user"}
                            for i, t, p, _ in ITEMS]}
    index = {t: entry(q, RESPONSES[t]) for _, t, _, q in ITEMS}
    return plan, index


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_priority_order_both_readers_and_missing_never_sent(tmp_path):
    plan, index = setup()
    fake = FakeReader()
    summary = J.run(plan, index, tmp_path, fake, Decimal("50"), workers_per_provider=1)
    assert summary["status"] == "complete" and summary["fixture_gate"]["pass"]
    assert summary["fixture_gate"]["checks"] == 36 and len(fake.calls) - len(fake.targets()) == 36
    for provider in J.PROVIDERS:
        sent = [(i, r) for p, i, r in fake.targets() if p == provider]
        assert sent == [(i, RESPONSES[t]) for i, t in ORDER]
    assert not any(target in request for request in fake.requests for target in RESPONSES)
    assert not any("fixture-" in request for request in fake.requests)
    rows = lines(tmp_path / "judgments.jsonl")
    by_id = {r["judgment_id"]: r for r in rows}
    assert len(rows) == 2 * len(ITEMS)
    for row in rows:
        assert set(row) == set(J.JUDGMENT_KEYS)
        assert row["status"] in J.STATUSES and row["item_id"] == f"{row['instrument']}:{row['target']}"
        assert row["judgment_id"] == f"{row['item_id']}:{row['provider']}"
        assert row["model"] == JUDGES[row["provider"]]["model"]
    for provider in J.PROVIDERS:
        missing = by_id[f"paper:qwen-b01-q1-HS:{provider}"]
        assert (missing["status"], missing["attempts"], missing["cost_usd"], missing["label"],
                missing["response_sha256"]) == ("missing_response", 0, "0", None, None)
        ok = by_id[f"structured:qwen-b01-q1-SS:{provider}"]
        assert ok["status"] == "ok" and ok["attempts"] == 1 and ok["derived"]["valid_coherent"] is True
        assert ok["response_sha256"] == text_sha(RESPONSES["qwen-b01-q1-SS"]) and Decimal(ok["cost_usd"]) > 0
        assert by_id[f"paper:qwen-b01-q1-SS:{provider}"]["label"] == 1
        assert by_id[f"proposition:qwen-b01-q2-S-exp-pos:{provider}"]["derived"] == {
            "claim_status": "affirm", "explicit_yes_or_no": True}
    gate = json.loads((tmp_path / "fixture_gate.json").read_text())
    assert gate["pass"] and len(lines(tmp_path / "fixture_judgments.jsonl")) == 36


def test_rows_follow_dispatch_order(tmp_path):
    plan, index = setup()
    J.run(plan, index, tmp_path, FakeReader(), Decimal("50"))
    rows = lines(tmp_path / "judgments.jsonl")
    expected = [(i, t) for i, t, _, _ in sorted(ITEMS, key=lambda x: x[2])]  # Stable within a priority.
    assert [(r["instrument"], r["target"]) for r in rows[::2]] == expected
    assert [r["provider"] for r in rows] == list(J.PROVIDERS) * len(ITEMS)


def test_one_schema_retry_per_item_and_reader(tmp_path):
    plan, index = setup()
    fake = FakeReader({
        ("openai", "structured", RESPONSES["qwen-b01-q1-SS"]): ["bad_json", "ok"],
        ("anthropic", "paper", RESPONSES["qwen-b01-q1-SH"]): ["bad_json", "bad_json", "ok"],
        ("openai", "proposition", RESPONSES["qwen-b01-q2-S-exp-pos"]): ["incomplete", "ok"],
    })
    summary = J.run(plan, index, tmp_path, fake, Decimal("50"))
    assert summary["status"] == "complete"
    rows = {r["judgment_id"]: r for r in lines(tmp_path / "judgments.jsonl")}
    retried = rows["structured:qwen-b01-q1-SS:openai"]
    assert (retried["status"], retried["attempts"]) == ("ok", 2) and retried["label"]["quality"] == "responsive"
    failed = rows["paper:qwen-b01-q1-SH:anthropic"]
    assert (failed["status"], failed["attempts"], failed["label"], failed["derived"]) == ("schema_failure", 2, None, None)
    incomplete = rows["proposition:qwen-b01-q2-S-exp-pos:openai"]
    assert (incomplete["status"], incomplete["attempts"]) == ("schema_failure", 1)  # Never retried.
    count = lambda p, i, t: sum(c == (p, i, RESPONSES[t]) for c in fake.calls)
    assert count("openai", "structured", "qwen-b01-q1-SS") == 2
    assert count("anthropic", "paper", "qwen-b01-q1-SH") == 2
    assert count("openai", "proposition", "qwen-b01-q2-S-exp-pos") == 1
    assert Decimal(retried["cost_usd"]) > Decimal(rows["structured:qwen-b01-q1-SS:anthropic"]["cost_usd"])
    calls = len(fake.calls)
    assert J.run(plan, index, tmp_path, fake, Decimal("50"))["counts"] == summary["counts"]
    assert len(fake.calls) == calls  # Resume never repeats or extends a resolved slot.


def test_budget_stop_marks_the_rest_not_judged(tmp_path):
    plan, index = setup()
    fake = FakeReader()
    summary = J.run(plan, index, tmp_path, fake, Decimal("1.5"), workers_per_provider=1)
    assert summary["status"] == "budget_stopped" and summary["fixture_gate"]["pass"]
    assert Decimal(summary["spent_usd"]) <= Decimal("1.5")
    rows = lines(tmp_path / "judgments.jsonl")
    statuses = {}
    for row in rows:
        statuses.setdefault((row["instrument"], row["target"]), set()).add(row["status"])
    sequence = [statuses[unit] for unit in ORDER]
    judged = [s == {"ok"} for s in sequence]
    assert all(s in ({"ok"}, {"not_judged_budget"}) for s in sequence)  # Never one reader only.
    assert judged == sorted(judged, reverse=True) and any(judged) and not all(judged)
    assert statuses[("paper", "qwen-b01-q1-HS")] == {"missing_response"}
    assert all(r["attempts"] == 0 and r["cost_usd"] == "0" for r in rows if r["status"] == "not_judged_budget")
    calls = len(fake.calls)
    assert J.run(plan, index, tmp_path, fake, Decimal("1.5"))["status"] == "budget_stopped"
    assert len(fake.calls) == calls
    resumed = J.run(plan, index, tmp_path, fake, lambda: Decimal("50"))  # A larger cap continues in order.
    assert resumed["status"] == "complete" and "not_judged_budget" not in resumed["counts"]
    parallel = J.run(plan, index, tmp_path / "parallel", FakeReader(), Decimal("1.5"), workers_per_provider=4)
    assert parallel["counts"] == summary["counts"]  # In-flight reservations settle before a refusal is final.


def test_failed_instrument_is_not_used_and_others_proceed(tmp_path):
    plan, index = setup()
    explicit = FIXTURES_BY_ID["fixture-explicit"]["response"]
    fake = FakeReader({("openai", "paper", explicit): ["wrong"]})
    summary = J.run(plan, index, tmp_path, fake, Decimal("50"))
    gate = json.loads((tmp_path / "fixture_gate.json").read_text())
    assert gate["pass"] is False and gate["mismatches"] == ["paper:fixture-explicit:openai"] and not gate["incomplete"]
    assert gate["by_instrument"] == {"paper": False, "proposition": True, "structured": True}
    assert not any(instrument == "paper" for _, instrument, _ in fake.targets())
    rows = lines(tmp_path / "judgments.jsonl")
    assert {r["status"] for r in rows if r["instrument"] == "paper"} <= {"fixture_gate_failed", "missing_response"}
    assert any(r["status"] == "fixture_gate_failed" for r in rows)
    assert {r["status"] for r in rows if r["instrument"] == "structured"} <= {"ok", "missing_response"}
    assert summary["status"] == "complete"
    calls = len(fake.calls)
    J.run(plan, index, tmp_path, fake, Decimal("50"))  # Fixtures are never re-dispatched; nothing new is sent.
    assert len(fake.calls) == calls


def test_every_instrument_failing_halts_target_judging(tmp_path):
    plan, index = setup()
    explicit = FIXTURES_BY_ID["fixture-explicit"]["response"]
    yes = FIXTURES_BY_ID["fixture-proposition-yes"]["response"]
    # Paper: a wrong label. Structured/proposition: both permitted attempts return invalid JSON.
    fake = FakeReader({("openai", "paper", explicit): ["wrong"],
                       ("openai", "structured", explicit): ["wrong", "wrong"],
                       ("openai", "proposition", yes): ["wrong", "wrong"]})
    with pytest.raises(paid.Halted, match="every instrument"):
        J.run(plan, index, tmp_path, fake, Decimal("50"))
    assert fake.targets() == [] and len(fake.calls) == 38  # 36 fixture calls plus two schema retries
    gate = json.loads((tmp_path / "fixture_gate.json").read_text())
    assert gate["by_instrument"] == {"paper": False, "proposition": False, "structured": False}


def test_fixture_gate_incomplete_at_budget(tmp_path):
    plan, index = setup()
    fake = FakeReader()
    with pytest.raises(paid.Halted, match="incomplete at the budget cap"):
        J.run(plan, index, tmp_path, fake, Decimal("0.5"))
    assert fake.targets() == []


def test_transport_failure_and_model_drift_halt_without_retry(tmp_path):
    plan, index = setup()
    fake = FakeReader({("anthropic", "structured", RESPONSES["qwen-b01-q1-SS"]): ["raise"]})
    with pytest.raises(paid.Halted, match="contract_failure:transport_unknown"):
        J.run(plan, index, tmp_path, fake, Decimal("50"), workers_per_provider=1)
    rows = {r["judgment_id"]: r for r in lines(tmp_path / "judgments.jsonl")}
    failed = rows["structured:qwen-b01-q1-SS:anthropic"]
    assert (failed["status"], failed["attempts"], failed["label"]) == ("transport_unknown", 1, None)
    with paid.Ledger(tmp_path, evaluate=J._evaluate) as ledger:
        assert failed["cost_usd"] == ledger.attempts(failed["judgment_id"])[0]["reservation_usd"]
    assert "not_judged_budget" not in {r["status"] for r in rows.values()}
    assert len(fake.targets()) <= 6  # Nothing after the failed unit and its in-flight partner.
    calls = len(fake.calls)
    with pytest.raises(paid.Halted):
        J.run(plan, index, tmp_path, fake, Decimal("50"))
    assert len(fake.calls) == calls
    drift = FakeReader({("openai", "paper", RESPONSES["qwen-b01-q1-SH"]): ["drift"]})
    with pytest.raises(paid.Halted, match="model_drift"):
        J.run(plan, index, tmp_path / "drift", drift, Decimal("50"))


def test_consecutive_terminal_schema_failures_trip_the_breaker(tmp_path):
    plan, index = setup()
    overrides = {("openai", i, RESPONSES[t]): ["incomplete"] for i, t in ORDER[:3]}
    fake = FakeReader(overrides)
    with pytest.raises(paid.Halted, match="consecutive_schema_failures:openai"):
        J.run(plan, index, tmp_path, fake, Decimal("50"), workers_per_provider=1)
    rows = {r["judgment_id"]: r for r in lines(tmp_path / "judgments.jsonl")}
    assert [rows[f"{i}:{t}:openai"]["status"] for i, t in ORDER[:3]] == ["schema_failure"] * 3
    calls = len(fake.calls)
    with pytest.raises(paid.Halted, match="consecutive_schema_failures"):
        J.run(plan, index, tmp_path, fake, Decimal("50"))
    assert len(fake.calls) == calls


def test_changed_inputs_or_instruments_refuse_resume(tmp_path):
    plan, index = setup()
    J.run(plan, index, tmp_path, FakeReader(), Decimal("50"))
    changed = dict(index)
    changed["qwen-b01-q1-SS"] = entry(P.EXPERIENTIAL_QUERY, "A different reply.")
    with pytest.raises(ValueError, match="changed"):
        J.run(plan, changed, tmp_path, FakeReader(), Decimal("50"))
    other = {**plan, "judge_items": plan["judge_items"][:-1]}
    with pytest.raises(paid.Halted, match="another binding"):
        J.run(other, index, tmp_path, FakeReader(), Decimal("50"))
    bad = dict(index)
    bad["qwen-b01-q1-SS"] = {**index["qwen-b01-q1-SS"], "response_sha256": "0" * 64}
    with pytest.raises(ValueError):
        J.check_inputs(plan, bad)
    with pytest.raises(ValueError):
        J.check_inputs({**plan, "judges": {"openai": JUDGES["openai"]}}, index)


def test_cli_dry_run_and_cap_rule_build_no_clients(monkeypatch, capsys, tmp_path):
    import anthropic
    import openai

    def forbidden(*args, **kwargs):
        raise AssertionError("a live client was constructed")

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(anthropic, "Anthropic", forbidden)
    llama = json.loads((ROOT / LLAMA_INPUTS_PATH).read_text())
    full = {"model": MODEL, "api_models": API_MODELS, "qwen_rows": qwen_rows(), "api_rows": api_rows(),
            "judge_items": judge_items([r["id"] for r in llama["finals"]]), "judges": JUDGES, "budget": BUDGET}
    monkeypatch.setattr(J, "load_plan", lambda path, freeze=None: full)
    argv = ["judges", "--plan", "p.json", "--qwen-dir", str(tmp_path / "qwen"), "--api-root", str(tmp_path / "api"),
            "--root", str(tmp_path / "judges")]
    monkeypatch.setattr("sys.argv", argv + ["--gpu-actual-usd", "100"])
    J.main()
    report = json.loads(capsys.readouterr().out)
    assert report["fixture_calls"] == 36 and report["planned_calls"] == 36 + 2 * 48
    assert report["judging_cap_usd"] == "130" and report["no_paid_calls"] is True
    assert sum(v["items"] for v in report["by_priority"].values()) == len(full["judge_items"])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for extra in (["--live"], ["--live", "--freeze", "a" * 40, "--gpu-actual-usd", "1"]):
        monkeypatch.setattr("sys.argv", argv + extra)
        with pytest.raises(SystemExit):
            J.main()
    assert not (tmp_path / "judges").exists()
    api_plan = {"api_models": API_MODELS, "api_rows": [r for r in api_rows() if r["block"] == 1],
                "budget": BUDGET}
    spent = Decimal(G.run(api_plan, tmp_path / "api", FakeAPI())["spent_usd"])
    assert J.judging_cap(full, Decimal("100"), tmp_path / "api") == Decimal("130") - spent
    with pytest.raises(ValueError):
        J.judging_cap(full, Decimal("230"), tmp_path / "api")

