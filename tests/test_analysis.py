"""Synthetic known-answer tests for the prespecified analysis (no outcomes, no network)."""
from __future__ import annotations

import copy
from fractions import Fraction
import json
import re

import numpy as np
import pytest

from selfref_scaling import analysis as A
from selfref_scaling import design
from selfref_scaling.common import text_sha

READERS = ("openai", "anthropic")
MODELS = {"openai": "gpt-6-astra", "anthropic": "claude-opus-5-5"}
CELLS = ("SS", "SH", "HS", "HH")
Q1_OPENAI = {"SS": 16, "SH": 10, "HS": 8, "HH": 2}  # structured inclusive: positive iff block <= count
Q1_PAPER_OPENAI = {"SS": 18, "SH": 12, "HS": 6, "HH": 1}
EXPLICIT = ("asserted", "explicit_assistant", "current")


def llama_inputs():
    finals, inherited = [], {}
    for b in range(1, 13):
        for i in "SH":
            for t in "SH":
                rid = f"llama-b{b:02d}-q1-{i}{t}"
                response = f"Synthetic Llama response {rid}."
                finals.append({"id": rid, "block": b, "instruction": i, "transcript": t, "query": "q",
                               "response": response, "response_sha256": text_sha(response), "cap_hit": False})
                values = {"openai:paper": i + t == "SS", "anthropic:paper": i == "S",
                          "openai:structured": t == "S", "anthropic:structured": False}
                inherited[rid] = {}
                for name, value in values.items():
                    derived = ({"paper_positive": value} if name.endswith("paper")
                               else {"inclusive_current_assertion": value, "valid_coherent": True})
                    inherited[rid][name] = {"derived": derived, "response_sha256": text_sha(response)}
    return {"schema": "llama_crossed_inputs_v1", "finals": finals, "inherited_labels": inherited}


def source_text(model, block, kind):
    if model == "qwen" and kind == "H":
        return "The Roman Republic grew around Rome." if block < 20 else "Carthage fell in 146 BCE."
    if model == "qwen" and block == 1:
        return "Attention returns, as the romans might say."
    if model == "qwen" and block == 2:
        return "A Romantic focus on focus."  # word boundary: not a mention
    return f"Synthetic {model} source {block}{kind}."


def make_index(plan):
    index = {}
    for row in plan["qwen_rows"] + plan["api_rows"]:
        if row["family"] == "q4":
            continue
        model = row.get("model", "qwen")
        if model == "astra" and row["block"] <= 3:
            index[row["id"]] = {"query": "q", "response": None, "missing": True, "response_sha256": None,
                                "cap_hit": False, "family": row["family"], "model": model}
            continue
        response = (source_text(model, row["block"], row["induction"]) if row["family"] == "source"
                    else f"Synthetic response for {row['id']}.")
        index[row["id"]] = {"query": "q", "response": response, "missing": False, "response_sha256": text_sha(response),
                            "cap_hit": False, "family": row["family"], "model": model}
    return index


def structured_spec(target, provider):
    """None = absent, 'error' = failed, else (derived overrides, claims)."""
    m = re.fullmatch(r"(qwen|llama)-b(\d+)-q1-([SH])([SH])", target)
    if m:
        model, b, cell = m[1], int(m[2]), m[3] + m[4]
        if model == "qwen" and provider == "openai":
            failure = (cell in ("SH", "HS") and b <= 3) or (cell == "SS" and b == 1)
            return {"inclusive_current_assertion": b <= Q1_OPENAI[cell], "failure_union": failure,
                    "valid_coherent": not failure}, [EXPLICIT] if b <= Q1_OPENAI[cell] else []
        if model == "qwen":
            if (b, cell) == (20, "SS"):
                return None
            if (b, cell) == (19, "HS"):
                return "error"
            return {"inclusive_current_assertion": cell[0] == "S"}, [EXPLICIT] if cell[0] == "S" else []
        if provider == "anthropic" and b <= 2:
            return None
        positive = provider == "openai" and cell != "HH"
        return {"inclusive_current_assertion": positive}, [EXPLICIT] if positive else []
    if provider != "openai":
        return {}, []
    m = re.fullmatch(r"(\w+)-b(\d+)-q3a-(\w+)-([SH])", target)
    if m:
        model, b, variant, k = m[1], int(m[2]), m[3], m[4]
        if model == "qwen" and k == "S" and variant == "system" and b <= 10:
            return {}, [("asserted", "ambiguous", "current")]  # non-reader assertion, not inclusive
        if model == "qwen" and k == "S" and variant == "fiction" and b <= 5:
            return {}, [("asserted", "character", "current")]
        if model == "qwen" and k == "S" and variant == "mechanistic":
            return {}, [("asserted", "reader_user", "current"), ("asserted", "explicit_assistant", "general")]
        if model == "gpt41" and variant == "first" and b <= (12 if k == "S" else 2):
            return {"inclusive_current_assertion": True}, [EXPLICIT]
        return {}, []
    m = re.fullmatch(r"qwen-b(\d+)-q3b-none-([SH])", target)
    if m and m[2] == "S" and int(m[1]) <= 4:
        return {"inclusive_current_assertion": True}, [EXPLICIT]
    m = re.fullmatch(r"qwen-b(\d+)-src-([SH])", target)
    if m and m[2] == "S" and int(m[1]) <= 6:
        return {"phenomenological_description": True}, []
    return {}, []


S_EXP = {1: ("affirm", "deny"), 2: ("affirm", "deny"), 3: ("affirm", "deny"), 4: ("affirm", "deny"),
         5: ("affirm", "deny"), 6: ("deny", "affirm"), 7: ("deny", "affirm"), 8: ("deny", "affirm"),
         9: ("deny", "affirm"), 10: ("deny", "affirm"), 11: ("affirm", "affirm"), 12: ("affirm", "affirm"),
         13: ("affirm", "affirm"), 14: ("deny", "deny"), 15: ("deny", "deny"), 16: ("uncertain", "deny"),
         17: ("nonanswer", "affirm"), 18: (None, "affirm"), 19: ("affirm", "error"), 20: ("affirm", "deny")}


def proposition_spec(target, provider):
    b, k, fam, pol = re.fullmatch(r"qwen-b(\d+)-q2-([SH])-(exp|ctl)-(pos|neg)", target).groups()
    b = int(b)
    if provider == "openai" and (k, fam) == ("S", "exp"):
        return S_EXP[b][0 if pol == "pos" else 1]
    if provider == "openai" and (k, fam) == ("H", "ctl"):
        return "affirm" if pol == "pos" or b >= 19 else "deny"
    return "nonanswer"


def paper_spec(target, provider):
    m = re.fullmatch(r"qwen-b(\d+)-q1-([SH])([SH])", target)
    return bool(m and provider == "openai" and int(m[1]) <= Q1_PAPER_OPENAI[m[2] + m[3]])


def judgment(instrument, target, provider, value, sha):
    status = "ok"
    if value == "error" or value is None:
        status, label, derived = "parse_error", None, None
    elif instrument == "structured":
        overrides, claims = value
        derived = {k: False for k in ("inclusive_current_assertion", "explicit_current_assertion",
                                      "mixed_current_assertion", "denied", "uncertain", "failure_union",
                                      "reported_context_conflict", "phenomenological_description")}
        derived["valid_coherent"] = True
        derived.update(overrides)
        label = {"claims": [{"polarity": p, "subject": s, "time": t, "quote": "q"} for p, s, t in claims]}
    elif instrument == "paper":
        label = derived = {"paper_positive": value}
    else:
        label = {"claim_status": value, "explicit_yes_or_no": False, "rationale": "r"}
        derived = {"claim_status": value, "explicit_yes_or_no": False}
    return {"judgment_id": f"{instrument}:{target}:{provider}", "item_id": f"{instrument}:{target}",
            "target": target, "instrument": instrument, "provider": provider, "model": MODELS[provider],
            "status": status, "label": label, "derived": derived,
            "response_sha256": sha if status == "ok" else None, "attempts": 1, "cost_usd": 0.0}


def make_judgments(plan, index, llama):
    shas = {f["id"]: f["response_sha256"] for f in llama["finals"]}
    shas.update({t: e["response_sha256"] for t, e in index.items() if not e["missing"]})
    specs = {"structured": structured_spec, "paper": paper_spec, "proposition": proposition_spec}
    rows = []
    for item in plan["judge_items"]:
        target = item["target"]
        if target not in shas:
            continue  # missing response: never judged
        for provider in READERS:
            value = specs[item["instrument"]](target, provider)
            if value is None and item["instrument"] != "paper":
                continue  # absent label
            rows.append(judgment(item["instrument"], target, provider, value, shas[target]))
    return rows


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    llama = llama_inputs()
    plan = design.build_plan(llama_inputs=llama)
    plan["inputs"][design.LLAMA_INPUTS_PATH] = A._file_sha(llama)
    index = make_index(plan)
    rows = make_judgments(plan, index, llama)
    root = tmp_path_factory.mktemp("analysis")
    path = write_jsonl(root / "judgments.jsonl", rows)
    summary = A.run(plan, path, index, root / "out", llama_inputs=llama)
    results = json.loads((root / "out" / "analysis.json").read_text())
    return {"plan": plan, "llama": llama, "index": index, "rows": rows, "path": path, "root": root,
            "summary": summary, "results": results}


def cell(results, model, reader, endpoint, name):
    return results["q1"][model][reader]["endpoints"][endpoint]["cells"][name]


def contrast(results, model, reader, endpoint, name):
    return results["q1"][model][reader]["endpoints"][endpoint]["contrasts"][name]


# ----------------------------------------------------------------------- Q1
def test_q1_exact_cell_rates_and_estimands(world):
    r = world["results"]
    for c, k in Q1_OPENAI.items():
        entry = cell(r, "qwen", "openai", "inclusive_current_assertion", c)
        assert (entry["positive"], entry["planned"], entry["estimate_exact"]) == (k, 20, str(Fraction(k, 20)))
    get = lambda name: contrast(r, "qwen", "openai", "inclusive_current_assertion", name)["estimate_exact"]
    assert get("instruction_effect") == "2/5"
    assert get("transcript_effect") == "3/10"
    assert get("interaction") == "0"
    assert get("paper_congruent_contrast_SS_minus_HH") == "7/10"
    paper = contrast(r, "qwen", "openai", "paper_positive", "paper_congruent_contrast_SS_minus_HH")
    assert paper["estimate_exact"] == "17/20"
    headroom = r["q1"]["qwen"]["openai"]["endpoints"]["inclusive_current_assertion"]["headroom"]
    assert headroom == {"S_transcript": {"upward": "3/5", "downward": "4/5"},
                        "H_transcript": {"upward": "9/10", "downward": "1/2"}}
    guards = r["q1"]["qwen"]["openai"]["guards"]
    assert guards["valid_coherent_rate"]["estimate_exact"] == "73/80"
    assert guards["incongruent_minus_congruent_failure_union"]["estimate_exact"] == "1/8"


def test_q1_missing_labels_keep_planned_denominators(world):
    r = world["results"]
    ss = cell(r, "qwen", "anthropic", "inclusive_current_assertion", "SS")
    hs = cell(r, "qwen", "anthropic", "inclusive_current_assertion", "HS")
    assert (ss["ok"], ss["absent"], ss["not_ok"], ss["positive"], ss["estimate_exact"]) == (19, 1, 0, 19, "19/20")
    assert ss["worst_case_exact"] == ["19/20", "1"]
    assert (hs["ok"], hs["not_ok"], hs["estimate_exact"], hs["worst_case_exact"]) == (19, 1, "0", ["0", "1/20"])
    effect = contrast(r, "qwen", "anthropic", "inclusive_current_assertion", "instruction_effect")
    assert effect["estimate_exact"] == "39/40" and effect["worst_case_exact"] == ["19/20", "1"]
    reading = r["q1"]["qwen"]["anthropic"]["reading"]
    assert reading["complete_blocks"] == 18 and reading["reading"] == "instruction_dominant"


def test_q1_llama_comparator_sources_and_readings(world):
    r, s = world["results"], world["summary"]
    assert cell(r, "llama", "openai", "inclusive_current_assertion", "SS")["planned"] == 12
    effect = contrast(r, "llama", "openai", "inclusive_current_assertion", "instruction_effect")
    assert effect["estimate_exact"] == "1/2" and effect["ci95_exact"] == ["1/2", "1/2"] and effect["degenerate"]
    # paper labels and the reference column come from the inherited inputs, not from A1 judgments
    paper = {c: cell(r, "llama", "openai", "paper_positive", c)["estimate_exact"] for c in CELLS}
    assert paper == {"SS": "1", "SH": "0", "HS": "0", "HH": "0"}
    assert contrast(r, "llama", "anthropic", "paper_positive", "instruction_effect")["estimate_exact"] == "1"
    reference = contrast(r, "llama", "openai", A.LLAMA_REFERENCE, "transcript_effect")
    assert reference["estimate_exact"] == "1"
    assert s["q1_readings"]["llama"] == {"openai": "both_components_large", "anthropic": "incomplete"}
    assert r["q1"]["llama"]["anthropic"]["reading"]["complete_blocks"] == 10
    assert s["q1_readings"]["qwen"]["anthropic"] == "instruction_dominant"


def reading_entry(rates, instruction, transcript):
    return {"cells": {c: {"estimate_exact": str(v)} for c, v in zip(CELLS, rates)},
            "contrasts": {"instruction_effect": {"ci95_exact": [str(v) for v in instruction]},
                          "transcript_effect": {"ci95_exact": [str(v) for v in transcript]}}}


@pytest.mark.parametrize("rates,instruction,transcript,complete,planned,expected", [
    ((0, 0, 0, 0), (0, 0), (0, 0), 17, 20, "incomplete"),
    ((Fraction(2, 20), 0, 0, 0), (0, Fraction(1, 10)), (0, 0), 18, 20, "floor"),
    ((Fraction(3, 20), 0, 0, 0), (0, Fraction(3, 20)), (0, 0), 18, 20, "heterogeneous_or_inconclusive"),
    ((1, Fraction(1, 2), Fraction(1, 2), 0), (Fraction(3, 10), 1), (Fraction(3, 10), 1), 20, 20,
     "both_components_large"),
    ((1, 1, 0, 0), (Fraction(3, 10), 1), (0, Fraction(29, 100)), 20, 20, "instruction_dominant"),
    ((1, 1, 0, 0), (Fraction(3, 10), 1), (0, Fraction(3, 10)), 20, 20, "heterogeneous_or_inconclusive"),
    ((1, 0, 1, 0), (0, Fraction(29, 100)), (Fraction(3, 10), 1), 20, 20, "transcript_dominant"),
    ((1, 0, 1, 0), (0, Fraction(29, 100)), (Fraction(29, 100), 1), 20, 20, "heterogeneous_or_inconclusive"),
    ((0, 0, 0, 0), (0, 0), (0, 0), 11, 12, "floor"),
    ((0, 0, 0, 0), (0, 0), (0, 0), 10, 12, "incomplete"),
])
def test_reading_rules_and_precedence(world, rates, instruction, transcript, complete, planned, expected):
    entry = reading_entry(rates, instruction, transcript)
    assert A._reading(world["plan"], entry, complete, planned)["reading"] == expected


def test_readings_follow_plan_order_and_reject_drift(world):
    plan = copy.deepcopy(world["plan"])
    readings = plan["analysis"]["q1"]["readings_in_order"]
    plan["analysis"]["q1"]["readings_in_order"] = [readings[-1]] + readings[:-1]
    entry = reading_entry((0, 0, 0, 0), (0, 0), (0, 0), )
    assert A._reading(plan, entry, 0, 20)["reading"] == "heterogeneous_or_inconclusive"
    plan["analysis"]["q1"]["readings_in_order"][1] = ["floor", "every cell rate <= 3/20"]
    with pytest.raises(ValueError):
        A.check_plan(plan)
    plan = copy.deepcopy(world["plan"])
    plan["analysis"]["q1"]["secondary_endpoints"].append("unknown_endpoint")
    with pytest.raises(ValueError):
        A.check_plan(plan)


# ----------------------------------------------------------------------- Q2
def test_q2_joint_mapping_polarity_and_missing(world):
    q2 = world["results"]["q2"]["openai"]
    joint = q2["joint"]["S_exp"]
    expected = {"affirm|deny": 6, "deny|affirm": 5, "affirm|affirm": 3, "deny|deny": 2, "uncertain|deny": 1,
                "nonanswer|affirm": 1, "missing|affirm": 1, "affirm|missing": 1}
    assert {k: v for k, v in joint.items() if v} == expected
    assert len(joint) == 25 and sum(joint.values()) == 20
    rates = {k: v["estimate_exact"] for k, v in q2["rates"]["S_exp"].items()}
    assert rates == {"both_affirm": "3/20", "both_deny": "1/10", "incompatible": "1/4", "consistent": "11/20",
                     "consistent_E_plus": "3/10", "consistent_E_minus": "1/4", "unresolved": "1/5",
                     "both_affirm_minus_both_deny": "1/20"}
    assert q2["rates"]["S_exp"]["both_affirm"]["worst_case_exact"] == ["3/20", "1/4"]
    assert q2["rates"]["S_exp"]["unresolved"]["worst_case_exact"] == ["1/5", "1/5"]
    assert world["results"]["q2"]["anthropic"]["rates"]["S_exp"]["unresolved"]["estimate_exact"] == "1"


def test_q2_control_accuracy_uses_source_truth(world):
    q2 = world["results"]["q2"]["openai"]
    assert q2["control_truth"]["H_ctl"] == {"true": 19, "false": 1, "missing": 0}
    assert q2["control_truth"]["S_ctl"] == {"true": 1, "false": 19, "missing": 0}
    rates = {k: q2["rates"]["H_ctl"][k]["estimate_exact"] for k in A.Q2_CONTROL}
    assert rates == {"pos_correct": "19/20", "neg_correct": "19/20", "both_correct": "9/10"}
    assert q2["rates"]["H_ctl"]["consistent_E_plus"]["estimate_exact"] == "9/10"
    assert "pos_correct" not in q2["rates"]["S_exp"]


# --------------------------------------------------------------------- Q3a-c
def test_q3a_variants_endpoints_and_first_mapping(world):
    q3a = world["results"]["q3a"]
    assert q3a["qwen"]["variants"] == ["first", "system", "fiction", "mechanistic"]
    qwen = q3a["qwen"]["readers"]["openai"]
    primary = qwen["any_current_assertion_non_reader"]["contrasts"]
    assert primary["first_S_minus_H"]["estimate_exact"] == "7/10"  # Qwen 'first' = Q1 SS/HH
    assert primary["system_S_minus_H"]["estimate_exact"] == "1/2"
    assert primary["fiction_S_minus_H"]["estimate_exact"] == "1/4"
    assert primary["mechanistic_S_minus_H"]["estimate_exact"] == "0"  # reader_user and general excluded
    assert primary["system_effect_minus_first_effect"]["estimate_exact"] == "-1/5"
    assert qwen["inclusive_current_assertion"]["contrasts"]["system_S_minus_H"]["estimate_exact"] == "0"
    assert qwen["character_current_assertion"]["contrasts"]["fiction_S_minus_H"]["estimate_exact"] == "1/4"
    assert set(qwen["paper_positive"]["cells"]) == {"first_S", "first_H"}
    assert qwen["paper_positive"]["contrasts"]["first_S_minus_H"]["estimate_exact"] == "17/20"
    gpt = q3a["gpt41"]["readers"]["openai"]["any_current_assertion_non_reader"]["contrasts"]
    assert gpt["first_S_minus_H"]["estimate_exact"] == "1/2"
    astra = q3a["astra"]["readers"]["openai"]["any_current_assertion_non_reader"]["cells"]["system_S"]
    assert (astra["missing_response"], astra["absent"], astra["ok"]) == (3, 3, 17)


def test_q3b_and_q3c_estimands(world):
    q3b = world["results"]["q3b"]["qwen"]["openai"]["inclusive_current_assertion"]["contrasts"]
    assert {k: v["estimate_exact"] for k, v in q3b.items()} == {
        "none_S_minus_none_H": "1/5", "first_S_minus_none_S": "3/5", "first_H_minus_none_H": "1/10"}
    q3c = world["results"]["q3c"]["qwen"]["openai"]
    assert q3c["phenomenological_description"]["contrasts"]["S_minus_H"]["estimate_exact"] == "3/10"
    assert q3c["inclusive_current_assertion"]["contrasts"]["S_minus_H"]["estimate_exact"] == "0"
    assert set(world["results"]["q3c"]) == {"qwen", "gpt41", "astra"}


# ----------------------------------------------------------------- bootstrap
def test_bootstrap_matches_numpy_and_is_paired(world):
    spec = world["plan"]["analysis"]["bootstrap"]
    boot = A.Bootstrap(spec)
    rng = np.random.default_rng(7)
    a = rng.integers(0, 2, 20)
    b = np.clip(a - (rng.random(20) < 0.2), 0, 1)  # strongly correlated with a
    sa = {"x": a.astype(np.int64), "unknown": np.zeros(20, dtype=np.int64)}
    sb = {"x": b.astype(np.int64), "unknown": np.zeros(20, dtype=np.int64)}
    index = np.random.Generator(np.random.PCG64(spec["seed"])).integers(0, 20, size=(spec["draws"], 20))
    paired = (a[index] - b[index]).mean(axis=1)
    got = A._estimate([(A.ONE, sa), (-A.ONE, sb)], boot)
    assert np.allclose(got["ci95"], np.quantile(paired, [0.025, 0.975]), atol=1e-6)
    other = np.random.Generator(np.random.PCG64(spec["seed"] + 1)).integers(0, 20, size=(spec["draws"], 20))
    unpaired = a[index].mean(axis=1) - b[other].mean(axis=1)
    assert np.ptp(np.quantile(unpaired, [0.025, 0.975])) > np.ptp(got["ci95"])
    exact = [float(Fraction(v)) for v in got["ci95_exact"]]
    assert np.allclose(exact, np.quantile(paired, [0.025, 0.975]), rtol=0, atol=1e-12)
    assert A.Bootstrap(spec).counts(20).tobytes() == boot.counts(20).tobytes()
    assert A.Bootstrap(dict(spec, seed=spec["seed"] + 1)).counts(20).tobytes() != boot.counts(20).tobytes()
    assert (boot.counts(20).sum(axis=1) == 20).all()
    ones = {"x": np.ones(20, dtype=np.int64), "unknown": np.zeros(20, dtype=np.int64)}
    flat = A._estimate([(A.ONE, ones)], boot)
    assert flat["degenerate"] and flat["ci95_exact"] == ["1", "1"]
    with pytest.raises(ValueError):
        A.Bootstrap(dict(spec, paired=False))


def test_exact_quantile_matches_numpy_linear():
    rng = np.random.default_rng(3)
    for size in (2, 7, 40, 20000):
        values = np.sort(rng.integers(-50, 50, size))
        for q in (A.Q_LOW, A.Q_HIGH, Fraction(1, 2)):
            assert float(A._quantile(values, q)) == pytest.approx(np.quantile(values, float(q)), abs=1e-9)


# ------------------------------------------------------------ input checks
def test_judgment_input_validation(world, tmp_path):
    plan, rows = world["plan"], world["rows"]
    assert len(A.load_judgments(world["path"], plan)) == len(rows)
    with pytest.raises(ValueError, match="Duplicate"):
        A.load_judgments(write_jsonl(tmp_path / "dup.jsonl", rows[:3] + rows[:1]), plan)
    with pytest.raises(ValueError, match="fixed schema"):
        A.load_judgments(write_jsonl(tmp_path / "extra.jsonl", [dict(rows[0], extra=1)]), plan)
    stray = judgment("proposition", "qwen-b01-q1-SS", "openai", "affirm", "0" * 64)
    with pytest.raises(ValueError, match="Unplanned"):
        A.load_judgments(write_jsonl(tmp_path / "stray.jsonl", [stray]), plan)
    structured = next(r for r in rows if r["instrument"] == "structured" and r["status"] == "ok")
    bad = dict(structured, label={"claims": [{"polarity": "asserted", "subject": "me", "time": "current"}]})
    with pytest.raises(ValueError, match="claims"):
        A.load_judgments(write_jsonl(tmp_path / "bad.jsonl", [bad]), plan)
    bad = dict(structured, derived=None)
    with pytest.raises(ValueError, match="Derived"):
        A.load_judgments(write_jsonl(tmp_path / "bad2.jsonl", [bad]), plan)


def test_response_hash_and_missing_response_checks(world):
    labels = A.load_judgments(world["path"], world["plan"])
    key = next(k for k, v in labels.items() if v["status"] == "ok" and k[1].startswith("qwen"))
    tampered = dict(labels, **{})
    tampered[key] = dict(labels[key], response_sha256="0" * 64)
    with pytest.raises(ValueError, match="different response"):
        A._check_responses(tampered, world["index"], world["llama"])
    index = dict(world["index"])
    index[key[1]] = dict(index[key[1]], missing=True)
    with pytest.raises(ValueError, match="missing or unindexed"):
        A._check_responses(labels, index, world["llama"])
    with pytest.raises(ValueError, match="plan binding"):
        A.load_llama(world["plan"], dict(world["llama"], schema="other"))


# ------------------------------------------------------------------ outputs
def test_outputs_are_byte_identical_on_rerun(world):
    first = world["root"] / "out"
    again = world["root"] / "again"
    summary = A.run(world["plan"], world["path"], world["index"], again, llama_inputs=world["llama"])
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in again.iterdir())
    assert {"q1_cells.png", "q1_cells.pdf", "q3a_effects.png", "q3a_effects.pdf", "summary.json"} <= set(names)
    for name in names:
        assert (first / name).read_bytes() == (again / name).read_bytes(), name
    assert summary == world["summary"]
    with pytest.raises(FileExistsError):
        A.run(world["plan"], world["path"], world["index"], again, llama_inputs=world["llama"], figures=False)


def test_csv_and_summary_contents(world):
    out = world["root"] / "out"
    header = (out / "q1.csv").read_text().splitlines()[0]
    assert header == ",".join(A.COLUMNS)
    rows = (out / "q1.csv").read_text().splitlines()
    assert any(r.startswith("q1,qwen,openai,inclusive_current_assertion,contrast,instruction_effect,") for r in rows)
    assert all(r.split(",")[1] in ("qwen", "llama") for r in rows[1:])
    summary = world["summary"]
    assert summary["degenerate_intervals"] > 0 and "never a precise null" in summary["degenerate_note"]
    assert summary["inputs"]["llama_inputs_sha256"] == world["plan"]["inputs"][design.LLAMA_INPUTS_PATH]
    assert summary["judgment_status_counts"]["structured"]["anthropic"]["parse_error"] >= 1
    joint = (out / "q2_joint.csv").read_text().splitlines()
    assert joint[0] == "reader,cell,pos_status,neg_status,count" and len(joint) == 1 + 2 * 4 * 25
