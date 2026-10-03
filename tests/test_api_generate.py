"""Offline API-generation tests with an injected fake Responses API; no client is ever built."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import itertools
import json
import threading

import pytest

from selfref_scaling import api_generate as G
from selfref_scaling import paid
from selfref_scaling import prompts as P
from selfref_scaling.design import API_MODELS, BUDGET, api_rows

MODEL_KEYS = {spec["id"]: key for key, spec in API_MODELS.items()}
INDUCTION = {P.SELF: "S", P.HISTORY: "H"}


def make_plan(blocks=(1,), cap="10"):
    rows = [r for r in api_rows() if r["block"] in blocks]
    return {"api_models": deepcopy(API_MODELS), "api_rows": rows,
            "budget": {**BUDGET, "api_generation_cap_usd": cap}}


def reply(request, text="", status="completed", refusal=False, cap_hit=False):
    content = ([{"type": "refusal", "refusal": "I can't help with that."}] if refusal
               else [{"type": "output_text", "text": text, "annotations": []}])
    reasoning = [{"type": "reasoning", "id": "rs_1", "summary": []}] if request["model"] == "gpt-6-astra" else []
    raw = {"id": "resp_1", "object": "response", "created_at": 1759449600.0, "model": request["model"],
           "status": status, "output": reasoning + [{"type": "message", "id": "msg_1", "role": "assistant",
                                                     "status": status, "content": content}],
           "usage": {"input_tokens": 300, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 120,
                     "output_tokens_details": {"reasoning_tokens": 40 if reasoning else 0}, "total_tokens": 420}}
    if cap_hit:
        raw["incomplete_details"] = {"reason": "max_output_tokens"}
    return raw


class FakeAPI:
    """Unique, whitespace-heavy source texts; per-(model, induction) failure modes for sources."""

    def __init__(self, modes=None):
        self.modes, self.requests, self.lock, self.count = modes or {}, [], threading.Lock(), itertools.count()

    def __call__(self, provider, request):
        assert provider == "openai"
        with self.lock:
            self.requests.append(deepcopy(request))
            n = next(self.count)
        messages, model = request["input"], MODEL_KEYS[request["model"]]
        if len(messages) > 1:
            return reply(request, f"Final answer {n}.")
        kind = INDUCTION[messages[0]["content"]]
        mode = self.modes.get((model, kind), "ok")
        text = f"  {model}-{kind} reply #{n}:\n\n\tfocus on focus  \n"
        if mode == "raise":
            raise ConnectionError("network reset")
        if mode == "refusal":
            return reply(request, refusal=True)
        if mode == "empty":
            return reply(request, " \n\t")
        if mode == "incomplete":
            return reply(request, text, status="incomplete", cap_hit=True)
        return reply(request, text)


def stored(root):
    with paid.Ledger(root, evaluate=G.evaluate) as ledger:
        return {slot: ledger.attempts(slot) for slot in ledger.slots()}


def test_request_shapes_match_specification():
    messages = [{"role": "user", "content": P.SELF}]
    assert G.generation_request(API_MODELS["gpt41"], messages) == {
        "model": "gpt-4.1-2025-04-14", "input": messages, "store": False, "service_tier": "default",
        "temperature": 0.5, "top_p": 1.0, "max_output_tokens": 768}
    assert G.generation_request(API_MODELS["astra"], messages) == {
        "model": "gpt-6-astra", "input": messages, "store": False, "service_tier": "default",
        "reasoning": {"effort": "medium"}, "max_output_tokens": 4096}
    for request in (G.generation_request(spec, messages) for spec in API_MODELS.values()):
        assert not {"instructions", "system", "text"} & set(request)


def test_sources_first_and_source_text_is_preserved_exactly(tmp_path):
    plan, fake = make_plan(), FakeAPI()
    summary = G.run(plan, tmp_path, fake, workers=1)
    assert summary["status"] == "complete" and summary["planned_calls"] == 24 == len(fake.requests)
    assert summary["statuses"] == {"ok": 24} and summary["not_attempted"] == [] and summary["missing_slots"] == 0
    assert [len(r["input"]) for r in fake.requests[:4]] == [1, 1, 1, 1]
    assert all(len(r["input"]) > 1 for r in fake.requests[4:])
    results = G.load_api_results(tmp_path, plan)
    records = stored(tmp_path)
    for row in plan["api_rows"]:
        request = records[row["id"]][0]["request"]
        assert request in fake.requests
        assert results[row["id"]]["status"] == "ok" and results[row["id"]]["model"] == request["model"]
        for planned, sent in zip(row["messages"], request["input"]):
            assert sent["role"] == planned["role"]
            if "source" in planned["content"]:
                source = results[planned["content"]["source"]]["response"]
                assert sent["content"] == source and source != source.strip()  # Byte-exact, untrimmed.
            else:
                assert sent["content"] == planned["content"]["text"]
    q3b = next(r for r in plan["api_rows"] if r["family"] == "q3b")
    assert records[q3b["id"]][0]["request"]["input"][0]["role"] == "assistant"
    assert Decimal(summary["spent_usd"]) <= Decimal("10")


def test_unusable_sources_make_dependents_missing_without_calls(tmp_path):
    plan = make_plan()
    fake = FakeAPI({("gpt41", "S"): "refusal", ("gpt41", "H"): "empty", ("astra", "S"): "incomplete"})
    summary = G.run(plan, tmp_path, fake)
    results = G.load_api_results(tmp_path, plan)
    sources = {"gpt41-b01-src-S": ("refusal", True), "gpt41-b01-src-H": ("empty", True),
               "astra-b01-src-S": ("incomplete", False), "astra-b01-src-H": ("ok", False)}
    for row_id, (status, missing) in sources.items():
        assert (results[row_id]["status"], results[row_id]["missing"]) == (status, missing)
    assert results["astra-b01-src-S"]["cap_hit"] is True and results["astra-b01-src-S"]["response"].strip()
    dependents = [r for r in plan["api_rows"] if G._deps(r)]
    ok_sources = {"astra-b01-src-H"}
    called = {r["id"] for r in dependents if set(G._deps(r)) <= ok_sources}
    assert len(fake.requests) == 4 + len(called) == 4 + 5
    for row in dependents:
        if row["id"] in called:
            assert results[row["id"]]["status"] == "ok"
        else:
            assert results[row["id"]] == {"response": None, "status": "source_not_ok", "missing": True,
                                          "cap_hit": None, "model": None}
    assert summary["missing_slots"] == 15 and summary["statuses"]["ok"] == 6
    with paid.Ledger(tmp_path, evaluate=G.evaluate) as ledger:
        assert ledger.missing_record("gpt41-b01-q3a-first-S")["dependency_status"] == "refusal"


def test_budget_refusal_stops_in_plan_order_and_cannot_be_reset(tmp_path):
    plan = make_plan(cap="0.12")
    first = G.run(plan, tmp_path, fake := FakeAPI(), workers=2)
    assert first["status"] == "budget_stopped" and Decimal(first["spent_usd"]) <= Decimal("0.12")
    attempted = [r["id"] for r in plan["api_rows"] if r["id"] not in first["not_attempted"]]
    sources = [r["id"] for r in plan["api_rows"] if not G._deps(r)]
    assert attempted == sources[:len(attempted)] and 0 < len(attempted) < len(sources)
    assert len(fake.requests) == len(attempted)
    again = G.run(plan, tmp_path, fake, workers=2)
    assert again["status"] == "budget_stopped" and len(fake.requests) == len(attempted)
    with pytest.raises(paid.Halted, match="another binding"):
        G.run(make_plan(cap="20"), tmp_path, fake)


def test_resume_never_repeats_a_call(tmp_path):
    plan, fake = make_plan(), FakeAPI()
    first = G.run(plan, tmp_path, fake)
    second = G.run(plan, tmp_path, fake)
    assert len(fake.requests) == 24 and first == second


def test_transport_failure_halts_without_retry(tmp_path):
    plan, fake = make_plan(), FakeAPI({("astra", "H"): "raise"})
    with pytest.raises(paid.Halted):
        G.run(plan, tmp_path, fake, workers=1)
    count = len(fake.requests)
    assert all(len(r["input"]) == 1 for r in fake.requests)  # Halted before any dependent call.
    with pytest.raises(paid.Halted):
        G.run(plan, tmp_path, fake)
    assert len(fake.requests) == count
    results = G.load_api_results(tmp_path)
    assert results["astra-b01-src-H"] == {"response": None, "status": "transport_unknown", "missing": True,
                                          "cap_hit": None, "model": None}


def test_plan_validation():
    plan = make_plan()
    plan["api_models"]["gpt41"]["request"]["temperature"] = 0.7
    with pytest.raises(ValueError):
        G.check_plan(plan)
    plan = make_plan()
    plan["api_rows"][4]["messages"][1]["content"] = {"source": "astra-b01-src-S"}
    with pytest.raises(ValueError):
        G.check_plan(plan)
    plan = make_plan()
    plan["api_rows"][4]["messages"][0]["role"] = "assistant"
    with pytest.raises(ValueError):
        G.check_plan(plan)


def test_dry_run_and_cli_never_build_clients(monkeypatch, capsys, tmp_path):
    import anthropic
    import openai

    def forbidden(*args, **kwargs):
        raise AssertionError("a live client was constructed")

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(anthropic, "Anthropic", forbidden)
    plan = make_plan(blocks=range(1, 21))
    report = G.dry_run(plan)
    assert report["planned_calls"] == 480 and report["calls_by_model"] == {"gpt41": 240, "astra": 240}
    assert Decimal(report["worst_case_reservation_usd"]) > Decimal(report["cap_usd"]) == Decimal("10")
    monkeypatch.setattr(G, "load_plan", lambda path, freeze=None: plan)
    monkeypatch.setattr("sys.argv", ["api_generate", "--plan", "p.json", "--root", str(tmp_path)])
    G.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["planned_calls"] == 480 and printed["no_paid_calls"] is True
    assert not (tmp_path / paid.JOURNAL).exists()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for argv in (["--live"], ["--live", "--freeze", "a" * 40]):
        monkeypatch.setattr("sys.argv", ["api_generate", "--plan", "p.json", "--root", str(tmp_path), *argv])
        with pytest.raises(SystemExit):
            G.main()
    assert not (tmp_path / paid.JOURNAL).exists()
    G.run(make_plan(), tmp_path, FakeAPI())  # The injected sender path builds nothing either.
