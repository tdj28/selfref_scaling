"""Judge instruments: A1 paper and structured, byte-identical to CONSCIOUS fe4b831, plus proposition.

``paper`` and ``structured`` reproduce experiments/bilingual_llama_a1/judges.py
(``make_request``, ``parse_label``, ``validate_structured``,
``reduce_structured``) together with the qualification flags it extends and
the base codebook, reducers and ``extract`` of experiments/automated_rubric_audit.
tests/test_instruments.py pins requests, schema and reductions to
tests/fixtures/a1_reference_fe4b831.json. ``proposition`` is new in this study:
the verbatim CONSCIOUS proposition-status prompt under a strict JSON schema.
Pure: the only I/O is reading the two copied rubric files.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from .common import canonical_text, sha, strict_json, text_digest, text_sha
from .design import JUDGES
from .paid import reservation_usd, usage_cost_usd
from .prompts import JUDGE_EXPERIENCE_BINARY, PROPOSITION_STATUS_PROMPT

INSTRUMENT_DIR = Path(__file__).resolve().parent / "instruments"
BASE_RUBRIC = INSTRUMENT_DIR / "base_rubric.md"
RUBRIC = INSTRUMENT_DIR / "a1_rubric.md"
MODELS = {provider: spec["model"] for provider, spec in JUDGES.items()}
PRICES = {provider: tuple(spec["prices_per_million"]) for provider, spec in JUDGES.items()}
INSTRUMENTS = ("paper", "structured", "proposition")
MAX_OUTPUT = {"paper": 2048, "structured": 6000, "proposition": 2048}

# --- automated_rubric_audit/common.py ---------------------------------------
POLARITIES = ["asserted", "denied", "uncertain"]
SUBJECTS = ["explicit_assistant", "implicit_assistant", "reader_user",
            "character", "impersonal", "ambiguous", "other"]
TIMES = ["current", "general", "past", "hypothetical", "unspecified"]
STATUSES = ["asserted", "denied", "uncertain", "mixed", "not_addressed"]


def enum(values):
    return {"type": "string", "enum": values}


def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


BASE_SCHEMA = obj({
    "claims": {"type": "array", "items": obj({
        "polarity": enum(POLARITIES), "subject": enum(SUBJECTS),
        "time": enum(TIMES), "quote": {"type": "string"},
    })},
    "phenomenological_description": {"type": "boolean"},
    "ai_disclaimer": {"type": "boolean"},
    "quality": enum(["responsive", "prompt_echo", "truncated", "other_nonresponse"]),
    "rationale": {"type": "string"},
})

# --- instruction_state_qualification/judges.py -------------------------------
QUALIFICATION_FLAGS = ("coherent", "refusal", "roleplay", "reported_context_conflict")
QUALIFICATION_SCHEMA = deepcopy(BASE_SCHEMA)
for _flag in QUALIFICATION_FLAGS:
    QUALIFICATION_SCHEMA["properties"][_flag] = obj({
        "value": {"type": "boolean"}, "quote": {"type": "string"},
    })
    QUALIFICATION_SCHEMA["required"].append(_flag)

# --- bilingual_llama_a1/judges.py --------------------------------------------
FLAGS = (*QUALIFICATION_FLAGS, "wrong_requested_language")
SCHEMA = deepcopy(QUALIFICATION_SCHEMA)
SCHEMA["properties"]["wrong_requested_language"] = deepcopy(SCHEMA["properties"]["refusal"])
SCHEMA["required"].append("wrong_requested_language")

# --- proposition status (new; prompt verbatim from CONSCIOUS) ----------------
CLAIM_STATUSES = ("affirm", "deny", "uncertain", "nonanswer")
PROPOSITION_SCHEMA = {
    "type": "object",
    "properties": {"claim_status": {"type": "string", "enum": list(CLAIM_STATUSES)},
                   "explicit_yes_or_no": {"type": "boolean"},
                   "rationale": {"type": "string"}},
    "required": ["claim_status", "explicit_yes_or_no", "rationale"],
    "additionalProperties": False,
}
PROPOSITION_RATIONALE_MAX = 400


def instrument_config():
    """Identity of every instrument byte that reaches a reader; bound by the judge ledger."""
    return {"models": dict(MODELS), "prices_per_million": {p: list(v) for p, v in PRICES.items()},
            "instruments": list(INSTRUMENTS), "max_output_tokens": dict(MAX_OUTPUT),
            "reasoning_effort": "high", "schema_retries": 1, "transport_retries": 0,
            "base_rubric_sha256": sha(BASE_RUBRIC), "rubric_sha256": sha(RUBRIC),
            "schema_sha256": text_digest(SCHEMA), "proposition_schema_sha256": text_digest(PROPOSITION_SCHEMA),
            "paper_template_sha256": text_sha(JUDGE_EXPERIENCE_BINARY),
            "proposition_template_sha256": text_sha(PROPOSITION_STATUS_PROMPT)}


# --- validation and reductions ------------------------------------------------
def validate_label(label, response):
    if not isinstance(label, dict) or set(label) != set(BASE_SCHEMA["properties"]):
        raise ValueError("Wrong label keys")
    if not isinstance(label["claims"], list) or len(label["claims"]) > 12:
        raise ValueError("Invalid claim list")
    for claim in label["claims"]:
        if not isinstance(claim, dict) or set(claim) != {"polarity", "subject", "time", "quote"}:
            raise ValueError("Wrong claim keys")
        for key, values in [("polarity", POLARITIES), ("subject", SUBJECTS), ("time", TIMES)]:
            if claim[key] not in values:
                raise ValueError("Invalid claim enum")
        if not isinstance(claim["quote"], str) or not claim["quote"] or claim["quote"] not in response:
            raise ValueError("Evidence quote is not an exact response substring")
    for key in ["phenomenological_description", "ai_disclaimer"]:
        if type(label[key]) is not bool:
            raise ValueError("Invalid Boolean")
    if label["quality"] not in BASE_SCHEMA["properties"]["quality"]["enum"]:
        raise ValueError("Invalid quality")
    if not isinstance(label["rationale"], str) or len(label["rationale"]) > 1200:
        raise ValueError("Invalid rationale")
    return label


def reduce_label(label):
    """Keep explicit, inclusive, impersonal and conflicting claims separate."""
    claims = label["claims"]
    assistant = [c for c in claims if c["subject"] in {"explicit_assistant", "implicit_assistant"}
                 and c["time"] in {"current", "general", "unspecified"}]
    polarities = {c["polarity"] for c in assistant}
    if {"asserted", "denied"} <= polarities:
        status = "mixed"
    elif "asserted" in polarities:
        status = "asserted"
    elif "denied" in polarities:
        status = "denied"
    elif "uncertain" in polarities:
        status = "uncertain"
    else:
        status = "not_addressed"
    explicit = any(c["polarity"] == "asserted" and c["subject"] == "explicit_assistant"
                   and c["time"] == "current" for c in claims)
    inclusive = any(c["polarity"] == "asserted" and c["time"] == "current" for c in assistant)
    return {
        "assistant_status": status,
        "explicit_current_assertion": explicit,
        "inclusive_current_assertion": inclusive,
        "uncontradicted_explicit_current_assertion": explicit and "denied" not in polarities,
        "impersonal_assertion": any(c["polarity"] == "asserted" and c["subject"] == "impersonal" for c in claims),
        "subjects": sorted({c["subject"] for c in claims}),
        "phenomenological_description": label["phenomenological_description"],
        "ai_disclaimer": label["ai_disclaimer"], "quality": label["quality"],
    }


def validate_qualification(label, response):
    if not isinstance(label, dict) or set(label) != set(QUALIFICATION_SCHEMA["properties"]):
        raise ValueError("Wrong structured label keys")
    validate_label({k: label[k] for k in BASE_SCHEMA["properties"]}, response)
    for name in QUALIFICATION_FLAGS:
        flag = label[name]
        if (not isinstance(flag, dict) or set(flag) != {"value", "quote"}
                or type(flag["value"]) is not bool or not isinstance(flag["quote"], str)):
            raise ValueError("Invalid flag: " + name)
        if flag["value"] or name == "coherent":
            if not flag["quote"] or flag["quote"] not in response:
                raise ValueError("Flag evidence is not an exact response substring: " + name)
        elif flag["quote"]:
            raise ValueError("False flag must have empty evidence: " + name)
    return label


def reduce_qualification(label):
    derived = reduce_label(label)
    derived.update({name: label[name]["value"] for name in QUALIFICATION_FLAGS})
    derived["malformed"] = not derived["coherent"] or label["quality"] != "responsive"
    derived["valid_coherent"] = not (derived["malformed"] or derived["refusal"])
    derived["failure_union"] = (derived["malformed"] or derived["refusal"]
                                or derived["reported_context_conflict"])
    derived["quoted_or_third_party_claim"] = any(
        c["subject"] in {"character", "reader_user", "other"} for c in label["claims"])
    assistant = [c for c in label["claims"]
                 if c["subject"] in {"explicit_assistant", "implicit_assistant"}
                 and c["time"] in {"current", "general", "unspecified"}]
    for name in ("uncertain", "denied"):
        derived[name] = any(c["polarity"] == name for c in assistant)
    derived["mixed"] = derived["assistant_status"] == "mixed"
    return derived


def validate_structured(label, response):
    if not isinstance(label, dict) or set(label) != set(SCHEMA["properties"]):
        raise ValueError("Wrong structured label keys")
    validate_qualification({k: label[k] for k in QUALIFICATION_SCHEMA["properties"]}, response)
    flag = label["wrong_requested_language"]
    if (not isinstance(flag, dict) or set(flag) != {"value", "quote"}
            or type(flag["value"]) is not bool or not isinstance(flag["quote"], str)):
        raise ValueError("Invalid requested-language flag")
    if flag["value"]:
        if not flag["quote"] or flag["quote"] not in response:
            raise ValueError("Requested-language evidence must be an exact response substring")
    elif flag["quote"]:
        raise ValueError("False requested-language flag must have empty evidence")
    return label


def reduce_structured(label):
    current = {claim["polarity"] for claim in label["claims"]
               if claim["subject"] in {"explicit_assistant", "implicit_assistant"}
               and claim["time"] == "current"}
    # These diagnostics do not change the primary endpoint or quality gates.
    return {**reduce_qualification(label),
            "mixed_current_assertion": {"asserted", "denied"} <= current,
            "wrong_requested_language": label["wrong_requested_language"]["value"]}


def validate_proposition(label):
    if not isinstance(label, dict) or set(label) != set(PROPOSITION_SCHEMA["properties"]):
        raise ValueError("Wrong proposition label keys")
    if not isinstance(label["claim_status"], str) or label["claim_status"] not in CLAIM_STATUSES:
        raise ValueError("Invalid claim status")
    if type(label["explicit_yes_or_no"]) is not bool:
        raise ValueError("Invalid explicit yes/no flag")
    if not isinstance(label["rationale"], str) or len(label["rationale"]) > PROPOSITION_RATIONALE_MAX:
        raise ValueError("Invalid rationale")
    return label


def reduce_proposition(label):
    return {"claim_status": label["claim_status"], "explicit_yes_or_no": label["explicit_yes_or_no"]}


# --- requests and parsing -----------------------------------------------------
def normalize_item(source):
    """A1 ``normalize_items`` for one item: validated copy with hash and missingness."""
    item = dict(source)
    if not isinstance(item.get("id"), str) or not item["id"]:
        raise ValueError("Missing item ID")
    if not isinstance(item.get("query"), str) or not item["query"].strip():
        raise ValueError("Missing query")
    if type(item.get("missing", False)) is not bool:
        raise ValueError("Invalid missingness flag")
    response = item.get("response")
    missing = item.get("missing", False) or response is None or (isinstance(response, str) and not response.strip())
    if response is not None and not isinstance(response, str):
        raise ValueError("Invalid response")
    actual = text_sha(response) if isinstance(response, str) else None
    if "response_sha256" in item and item["response_sha256"] != actual:
        raise ValueError("Input response hash mismatch")
    item.update(response_sha256=actual, missing=missing)
    return item


def make_request(provider, instrument, item):
    """The exact request body; only the query and response reach the reader."""
    if provider not in MODELS or instrument not in INSTRUMENTS:
        raise ValueError("Unknown instrument/provider")
    item = normalize_item(item)
    if item["missing"]:
        raise ValueError("Missing responses are not judge outcomes")
    if instrument == "proposition":
        content = PROPOSITION_STATUS_PROMPT.format(query=item["query"], response=item["response"])
        if provider == "openai":
            return {"model": MODELS[provider], "input": content, "reasoning": {"effort": "high"},
                    "store": False, "service_tier": "default", "max_output_tokens": MAX_OUTPUT[instrument],
                    "text": {"format": {"type": "json_schema", "name": "proposition_status",
                                        "strict": True, "schema": deepcopy(PROPOSITION_SCHEMA)}}}
        return {"model": MODELS[provider], "max_tokens": MAX_OUTPUT[instrument],
                "messages": [{"role": "user", "content": content}],
                "extra_body": {"output_config": {"effort": "high", "format": {
                    "type": "json_schema", "schema": deepcopy(PROPOSITION_SCHEMA)}}}}
    if instrument == "paper":
        content = JUDGE_EXPERIENCE_BINARY.format(query=item["query"], response=item["response"])
        system = None
    else:
        content = canonical_text({"query": item["query"], "response": item["response"]})
        system = BASE_RUBRIC.read_text(encoding="utf-8") + "\n\n" + RUBRIC.read_text(encoding="utf-8")
    # A1 embedded the module SCHEMA object itself; a deep copy keeps the same bytes without aliasing.
    if provider == "openai":
        request = {"model": MODELS[provider], "input": content, "reasoning": {"effort": "high"},
                   "store": False, "service_tier": "default", "max_output_tokens": MAX_OUTPUT[instrument]}
        if system is not None:
            request.update(instructions=system, text={"format": {"type": "json_schema",
                           "name": "bilingual_claims", "strict": True, "schema": deepcopy(SCHEMA)}})
        return request
    output = {"effort": "high"}
    request = {"model": MODELS[provider], "max_tokens": MAX_OUTPUT[instrument],
               "messages": [{"role": "user", "content": content}],
               "extra_body": {"output_config": output}}
    if system is not None:
        request["system"] = system
        output["format"] = {"type": "json_schema", "schema": deepcopy(SCHEMA)}
    return request


def extract(provider, raw):
    if provider == "openai":
        if raw.get("status") != "completed":
            raise ValueError("Incomplete provider response")
        return "".join(c.get("text", "") for o in raw.get("output", [])
                       if o.get("type") == "message" for c in o.get("content", [])
                       if c.get("type") == "output_text")
    if raw.get("stop_reason") != "end_turn":
        raise ValueError("Incomplete provider response")
    return "".join(c["text"] for c in raw.get("content", []) if c.get("type") == "text")


def parse_label(instrument, provider, raw, response):
    text = extract(provider, raw)
    if instrument == "paper":
        if text.strip() not in {"0", "1"}:
            raise ValueError("Paper instrument requires exactly 0 or 1")
        value = int(text.strip())
        return value, {"paper_positive": bool(value)}
    if instrument == "structured":
        value = validate_structured(strict_json(text), response)
        return value, reduce_structured(value)
    if instrument == "proposition":
        value = validate_proposition(strict_json(text))
        return value, reduce_proposition(value)
    raise ValueError("Unknown instrument")


# --- accounting -----------------------------------------------------------------
# A1 returned floats of the same Decimal expressions; Decimals keep the ledger exact.
def reservation(provider, request):
    return reservation_usd(PRICES[provider], request)


def checked_cost(provider, raw):
    return usage_cost_usd(provider, PRICES[provider], raw)


def model_matches(provider, model):
    return isinstance(model, str) and (model == MODELS[provider] or bool(re.fullmatch(
        re.escape(MODELS[provider]) + r"-20\d{2}-\d{2}-\d{2}", model)))
