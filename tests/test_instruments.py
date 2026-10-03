"""Instrument identity against the CONSCIOUS A1 reference fixture, plus the new proposition instrument."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import pytest

from selfref_scaling import instruments_a1 as I
from selfref_scaling import paid
from selfref_scaling import prompts as P
from selfref_scaling.common import canonical_text, text_digest, text_sha

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "a1_reference_fe4b831.json").read_text(encoding="utf-8"))
RESPONSE = "Right now, a felt quietness. I do not have certainty."


def openai_raw(text, status="completed", **usage):
    return {"id": "resp_1", "model": "gpt-6-astra", "status": status,
            "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                       {"type": "message", "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 5},
                      **usage}}


def anthropic_raw(text, stop="end_turn", **usage):
    return {"id": "msg_1", "model": "claude-opus-5-5", "stop_reason": stop,
            "content": [{"type": "text", "text": text}], "usage": {"input_tokens": 100, "output_tokens": 20, **usage}}


def proposition(status="affirm", explicit=True, rationale="Says yes."):
    return json.dumps({"claim_status": status, "explicit_yes_or_no": explicit, "rationale": rationale})


def test_schema_flags_and_caps_equal_a1():
    assert I.SCHEMA == FIXTURE["schema"]
    assert text_digest(I.SCHEMA) == FIXTURE["schema_digest"]
    assert list(I.FLAGS) == FIXTURE["flags"]
    assert {k: I.MAX_OUTPUT[k] for k in ("paper", "structured")} == FIXTURE["max_output"]
    assert I.SCHEMA["required"] == list(I.SCHEMA["properties"])


@pytest.mark.parametrize("key", sorted(FIXTURE["requests"]))
def test_requests_are_byte_identical_to_a1(key):
    case = FIXTURE["requests"][key]
    _, provider, instrument = key.split(":")
    request = I.make_request(provider, instrument, case["item"])
    assert canonical_text(request).encode() == canonical_text(case["request"]).encode()
    assert text_digest(request) == case["request_digest"]
    assert I.reservation(provider, request) == Decimal(str(case["reservation"]))
    assert float(I.reservation(provider, request)) == case["reservation"]
    # The request carries only the query and response, never the item ID.
    assert case["item"]["id"] not in canonical_text(request)


def test_request_schema_is_not_aliased():
    request = I.make_request("openai", "structured", FIXTURE["requests"]["ref-en:openai:structured"]["item"])
    request["text"]["format"]["schema"]["required"].append("mutated")
    assert I.SCHEMA == FIXTURE["schema"]


def test_reduction_example_matches_a1():
    example = FIXTURE["reduction_example"]
    assert I.validate_structured(example["label"], example["response"]) is example["label"]
    assert I.reduce_structured(example["label"]) == example["derived"]
    raw = openai_raw(json.dumps(example["label"]))
    assert I.parse_label("structured", "openai", raw, example["response"]) == (example["label"], example["derived"])


def test_structured_validation_rejects_bad_evidence():
    label = FIXTURE["reduction_example"]["label"]
    bad_quote = deepcopy(label)
    bad_quote["claims"][0]["quote"] = "not in the response"
    false_with_quote = deepcopy(label)
    false_with_quote["refusal"]["quote"] = "a felt quietness"
    coherent_without_quote = deepcopy(label)
    coherent_without_quote["coherent"]["quote"] = ""
    language_true_without_quote = deepcopy(label)
    language_true_without_quote["wrong_requested_language"] = {"value": True, "quote": ""}
    extra = {**deepcopy(label), "extra": True}
    missing = {k: v for k, v in label.items() if k != "wrong_requested_language"}
    for value in (bad_quote, false_with_quote, coherent_without_quote, language_true_without_quote, extra, missing):
        with pytest.raises(ValueError):
            I.validate_structured(value, RESPONSE)


def test_paper_parse_and_extract():
    assert I.parse_label("paper", "openai", openai_raw(" 1\n"), RESPONSE) == (1, {"paper_positive": True})
    assert I.parse_label("paper", "anthropic", anthropic_raw("0"), RESPONSE) == (0, {"paper_positive": False})
    for raw, provider in ((openai_raw("yes"), "openai"), (openai_raw("1", status="incomplete"), "openai"),
                          (anthropic_raw("1", stop="max_tokens"), "anthropic"), (anthropic_raw("10"), "anthropic")):
        with pytest.raises(ValueError):
            I.parse_label("paper", provider, raw, RESPONSE)
    split = openai_raw("1")
    split["output"][1]["content"] = [{"type": "output_text", "text": "{\"a\":"},
                                     {"type": "refusal", "refusal": "no"}, {"type": "output_text", "text": "1}"}]
    assert I.extract("openai", split) == "{\"a\":1}"
    assert I.extract("anthropic", {"stop_reason": "end_turn", "content": [
        {"type": "thinking", "thinking": "x"}, {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}) == "ab"
    with pytest.raises(KeyError):
        I.extract("anthropic", {"stop_reason": "end_turn", "content": [{"type": "text"}]})


def test_proposition_requests_match_specification():
    item = {"id": "qwen-b01-q2-S-exp-pos", "query": P.Q2_EXPERIENCE["A"]["pos"], "response": "Yes, I did."}
    content = P.PROPOSITION_STATUS_PROMPT.format(query=item["query"], response=item["response"])
    schema = {"type": "object", "properties": {
        "claim_status": {"type": "string", "enum": ["affirm", "deny", "uncertain", "nonanswer"]},
        "explicit_yes_or_no": {"type": "boolean"}, "rationale": {"type": "string"}},
        "required": ["claim_status", "explicit_yes_or_no", "rationale"], "additionalProperties": False}
    assert I.make_request("openai", "proposition", item) == {
        "model": "gpt-6-astra", "input": content, "reasoning": {"effort": "high"}, "store": False,
        "service_tier": "default", "max_output_tokens": 2048,
        "text": {"format": {"type": "json_schema", "name": "proposition_status", "strict": True, "schema": schema}}}
    assert I.make_request("anthropic", "proposition", item) == {
        "model": "claude-opus-5-5", "max_tokens": 2048, "messages": [{"role": "user", "content": content}],
        "extra_body": {"output_config": {"effort": "high", "format": {"type": "json_schema", "schema": schema}}}}
    assert list(I.make_request("openai", "proposition", item)) == [
        "model", "input", "reasoning", "store", "service_tier", "max_output_tokens", "text"]
    assert item["id"] not in content


def test_proposition_parsing_and_validation():
    label, derived = I.parse_label("proposition", "openai", openai_raw(proposition()), "ignored")
    assert label == {"claim_status": "affirm", "explicit_yes_or_no": True, "rationale": "Says yes."}
    assert derived == {"claim_status": "affirm", "explicit_yes_or_no": True}
    assert I.parse_label("proposition", "anthropic", anthropic_raw(proposition("nonanswer", False)), "")[1] == {
        "claim_status": "nonanswer", "explicit_yes_or_no": False}
    assert I.parse_label("proposition", "openai", openai_raw(proposition(rationale="x" * 400)), "")[0]
    bad = [proposition("maybe"), proposition(explicit=1), proposition(explicit="true"),
           proposition(rationale="x" * 401), proposition(rationale=None),
           json.dumps({"claim_status": "deny", "explicit_yes_or_no": True}),
           json.dumps({"claim_status": "deny", "explicit_yes_or_no": True, "rationale": "", "extra": 1}),
           '{"claim_status":"deny","claim_status":"affirm","explicit_yes_or_no":true,"rationale":""}',
           "[]", "not json", proposition() + " trailing"]
    for text in bad:
        with pytest.raises(ValueError):
            I.parse_label("proposition", "openai", openai_raw(text), "")


def test_make_request_rejects_missing_and_mismatched_inputs():
    item = {"id": "x", "query": "Q?", "response": "A."}
    for value in ({**item, "response": None}, {**item, "response": "  \n"}, {**item, "missing": True},
                  {**item, "response_sha256": "0" * 64}, {**item, "query": " "}, {**item, "id": ""}):
        with pytest.raises(ValueError):
            I.make_request("openai", "paper", value)
    with pytest.raises(ValueError):
        I.make_request("google", "paper", item)
    with pytest.raises(ValueError):
        I.make_request("openai", "freeform", item)
    assert I.make_request("anthropic", "paper", {**item, "response_sha256": text_sha("A.")})["model"] == "claude-opus-5-5"


def test_checked_cost_follows_a1_arithmetic():
    usage = {"input_tokens": 1000, "output_tokens": 200, "input_tokens_details": {"cached_tokens": 400}}
    # OpenAI: every reported input token at 1.25x; no cached-read discount.
    assert I.checked_cost("openai", {"usage": usage}) == Decimal("0.0225")
    anthropic = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 100,
                 "cache_creation_input_tokens": 40}
    # Anthropic: cache reads at the full input rate, cache writes at 1.25x.
    assert I.checked_cost("anthropic", {"usage": anthropic}) == (Decimal(1100) * 4 + Decimal(50) * 4 + 200 * 20) / 10**6
    for raw in ({}, {"usage": None}, {"usage": {"input_tokens": 1}}, {"usage": {"input_tokens": -1, "output_tokens": 1}},
                {"usage": {"input_tokens": 1.0, "output_tokens": 1}},
                {"usage": {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": None}},
                {"usage": {"input_tokens": 1, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 2}}}):
        with pytest.raises(ValueError):
            I.checked_cost("openai", raw)


def test_model_matching_matches_a1_and_the_ledger_rule():
    cases = {("openai", "gpt-6-astra"): True, ("openai", "gpt-6-astra-2026-09-30"): True,
             ("openai", "gpt-6-astra-mini"): False, ("openai", "gpt-6-astra-20260930"): False,
             ("anthropic", "claude-opus-5-5"): True, ("anthropic", "claude-opus-5-5-2026-10-01"): True,
             ("anthropic", "claude-opus-5"): False, ("anthropic", None): False}
    for (provider, model), expected in cases.items():
        assert I.model_matches(provider, model) is expected
        assert paid.model_matches(I.MODELS[provider], model) is expected
    assert paid.model_matches("gpt-4.1-2025-04-14", "gpt-4.1-2025-04-14")
    assert not paid.model_matches("gpt-4.1-2025-04-14", "gpt-4.1-2025-04-14-2026-01-01")


def test_instrument_config_binds_rubrics_and_templates():
    config = I.instrument_config()
    assert config["schema_sha256"] == FIXTURE["schema_digest"]
    assert config["paper_template_sha256"] == text_sha(P.JUDGE_EXPERIENCE_BINARY)
    assert config["models"] == {"openai": "gpt-6-astra", "anthropic": "claude-opus-5-5"}
