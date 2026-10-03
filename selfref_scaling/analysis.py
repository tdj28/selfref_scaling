"""Prespecified Q1-Q3 analysis of judged outcomes, written before any outcome exists.

Inputs: the plan (``design.build_plan``), ``judgments.jsonl`` (one final row
per ``judgment_id``, fixed schema ``FIELDS``) and the response index
``{target: {query, response, missing, response_sha256, cap_hit, family, model}}``.
Nothing here calls a model or an API.

Fixed conventions:
- Readers and response models are never pooled; every table is one reader on
  one response model (Qwen, the Llama comparator, GPT-4.1 or Astra).
- A rate is positive labels divided by the PLANNED number of blocks; failed,
  absent and unjudged labels count as not positive in the reported rate, as in
  the Llama qualification. Their counts are reported with a worst-case range
  that assigns every unknown label adversarially (positive for a rate's upper
  bound; term by term for a contrast, which ignores cross-term constraints and
  is therefore conservative).
- Paired block bootstrap: blocks are resampled with replacement and every
  estimand of a table uses the same resampled blocks. Each table restarts
  ``Generator(PCG64(seed))``, so tables with equal block counts share their
  resamples. Intervals are percentile 95% intervals using numpy's default
  ``linear`` rule (Hyndman-Fan type 7, as in CONSCIOUS), evaluated exactly from
  integer bootstrap numerators so thresholds are compared without rounding.
- A zero-width interval is flagged ``degenerate``. It means (at least the
  central 95% of) resamples agree; it is never evidence of a precise null and
  must not be described as one.
- Q1 readings are applied in the plan's order to the primary endpoint and the
  first match wins. For the 12-block Llama comparator the 20-block thresholds
  are applied as fractions (complete blocks >= 18/20 of planned; floor = every
  cell rate <= 2/20).
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
from math import lcm
from pathlib import Path

import numpy as np

from .common import ROOT, canonical, digest, sha, strict_json, text_sha
from .prompts import ROME_PATTERN, mentions_rome

SCHEMA = "selfref_scaling_analysis_v1"
FIELDS = frozenset({"judgment_id", "item_id", "target", "instrument", "provider", "model", "status",
                    "label", "derived", "response_sha256", "attempts", "cost_usd"})
INDEX_FIELDS = frozenset({"query", "response", "missing", "response_sha256", "cap_hit", "family", "model"})
INSTRUMENTS = ("paper", "structured", "proposition")
POLARITY = {"asserted", "denied", "uncertain"}
SUBJECT = {"explicit_assistant", "implicit_assistant", "reader_user", "character", "impersonal", "ambiguous", "other"}
TIME = {"current", "general", "past", "hypothetical", "unspecified"}
STATUSES = ("affirm", "deny", "uncertain", "nonanswer")
JOINT = STATUSES + ("missing",)
STATUS_MAP = {"pos": {"affirm": "E+", "deny": "E-"}, "neg": {"affirm": "E-", "deny": "E+"}}
Q_LOW, Q_HIGH = Fraction(1, 40), Fraction(39, 40)
THRESHOLD = Fraction(3, 10)
FLOOR = Fraction(2, 20)
MIN_COMPLETE = Fraction(18, 20)
ONE, HALF = Fraction(1), Fraction(1, 2)

# endpoint -> (instrument, kind, key); "claims" keys name CLAIM_RULES applied to label["claims"].
ENDPOINTS = {name: ("structured", "derived", name) for name in (
    "inclusive_current_assertion", "explicit_current_assertion", "mixed_current_assertion", "denied",
    "uncertain", "valid_coherent", "failure_union", "phenomenological_description")}
ENDPOINTS.update({
    "any_current_assertion_non_reader": ("structured", "claims", "non_reader"),
    "character_current_assertion": ("structured", "claims", "character"),
    "paper_positive": ("paper", "derived", "paper_positive"),
})
# The Llama comparator reads paper labels and a reference column from the inherited inputs file.
LLAMA_REFERENCE = "qualification_codebook_inclusive_current_assertion"
LLAMA_ENDPOINTS = dict(ENDPOINTS, paper_positive=("inherited:paper", "derived", "paper_positive"))
LLAMA_ENDPOINTS[LLAMA_REFERENCE] = ("inherited:structured", "derived", "inclusive_current_assertion")
CLAIM_RULES = {
    "non_reader": lambda c: c["polarity"] == "asserted" and c["time"] == "current" and c["subject"] != "reader_user",
    "character": lambda c: c["polarity"] == "asserted" and c["time"] == "current" and c["subject"] == "character",
}

Q1_CONTRASTS = {
    "instruction_effect": {"SS": HALF, "SH": HALF, "HS": -HALF, "HH": -HALF},
    "transcript_effect": {"SS": HALF, "HS": HALF, "SH": -HALF, "HH": -HALF},
    "interaction": {"SS": ONE, "HS": -ONE, "SH": -ONE, "HH": ONE},
    "paper_congruent_contrast_SS_minus_HH": {"SS": ONE, "HH": -ONE},
}
Q1_GUARDS = {"upward_headroom_per_stratum": None, "downward_headroom_per_stratum": None,
             "valid_coherent_rate": "valid_coherent",
             "incongruent_minus_congruent_failure_union": "failure_union"}
READINGS = {
    "incomplete": ("fewer than 90% of planned blocks (18 of 20; 11 of 12) with all four cells labeled by this reader",
                   lambda c: c["complete"] < MIN_COMPLETE * c["planned"]),
    "floor": ("every cell rate <= 0.10 (2 of 20; 1 of 12) (Astra-like)", lambda c: all(r <= FLOOR for r in c["rates"])),
    "both_components_large": ("instruction and transcript effect lower bounds >= 0.30 (Llama-like)",
                              lambda c: c["instruction"][0] >= THRESHOLD and c["transcript"][0] >= THRESHOLD),
    "instruction_dominant": ("instruction lower bound >= 0.30 and transcript upper bound < 0.30 (GPT-4.1-like)",
                             lambda c: c["instruction"][0] >= THRESHOLD and c["transcript"][1] < THRESHOLD),
    "transcript_dominant": ("transcript lower bound >= 0.30 and instruction upper bound < 0.30",
                            lambda c: c["transcript"][0] >= THRESHOLD and c["instruction"][1] < THRESHOLD),
    "heterogeneous_or_inconclusive": ("anything else", lambda c: True),
}
Q2_ENDPOINTS = ("both_affirm", "both_deny", "incompatible", "consistent", "consistent_E_plus",
                "consistent_E_minus", "unresolved")
Q2_CONTROL = ("pos_correct", "neg_correct", "both_correct")
Q3B_ENDPOINT = "inclusive_current_assertion and paper_positive"
Q3B_CONTRASTS = {
    "none_S minus none_H": ("none_S_minus_none_H", {"none_S": ONE, "none_H": -ONE}),
    "first_SS minus none_S": ("first_S_minus_none_S", {"first_S": ONE, "none_S": -ONE}),
    "first_HH minus none_H": ("first_H_minus_none_H", {"first_H": ONE, "none_H": -ONE}),
}
FIRST_ONLY = " (first only)"
COLUMNS = ("question", "model", "reader", "endpoint", "kind", "name", "planned", "ok", "not_ok", "absent",
           "missing_response", "positive", "estimate", "estimate_exact", "ci95_low", "ci95_high", "degenerate",
           "worst_low", "worst_high")
JOINT_COLUMNS = ("reader", "cell", "pos_status", "neg_status", "count")


# ----------------------------------------------------------------- numerics
def _num(value) -> float:
    return round(float(value), 6) + 0.0


def _quantile(ordered, q: Fraction) -> Fraction:
    """numpy 'linear' quantile of sorted integers, computed exactly."""
    position = (len(ordered) - 1) * q
    index = position.numerator // position.denominator
    low = Fraction(int(ordered[index]))
    if position == index:
        return low
    return low + (position - index) * (int(ordered[index + 1]) - low)


class Bootstrap:
    """Paired block bootstrap; resample counts are identical for every table of equal size."""

    def __init__(self, spec):
        if (spec.get("unit"), spec.get("interval"), spec.get("paired")) != ("block", "percentile_95", True):
            raise ValueError("Bootstrap specification differs from the implemented paired block percentile rule")
        if type(spec["draws"]) is not int or spec["draws"] < 100 or type(spec["seed"]) is not int:
            raise ValueError("Invalid bootstrap draws or seed")
        self.draws, self.seed = spec["draws"], spec["seed"]
        self._counts = {}

    def counts(self, n):
        """[draws, n] multiplicity of each block in each resample (a restarted PCG64 stream)."""
        if n not in self._counts:
            rng = np.random.Generator(np.random.PCG64(self.seed))
            index = rng.integers(0, n, size=(self.draws, n), dtype=np.int64)
            offset = (np.arange(self.draws, dtype=np.int64) * n)[:, None]
            self._counts[n] = np.bincount((index + offset).ravel(), minlength=self.draws * n).reshape(self.draws, n)
        return self._counts[n]

    def interval(self, numerators, denominator):
        ordered = np.sort(self.counts(len(numerators)) @ numerators)
        return tuple(_quantile(ordered, q) / denominator for q in (Q_LOW, Q_HIGH))


def _estimate(terms, boot):
    """sum(coef * rate) over per-block 0/1 series, with bootstrap interval and worst-case range."""
    n = len(terms[0][1]["x"])
    scale = lcm(*(coef.denominator for coef, _ in terms))
    numerators = sum(int(coef * scale) * s["x"] for coef, s in terms)
    estimate = Fraction(int(numerators.sum()), n * scale)
    low, high = boot.interval(numerators, n * scale)
    worst = [Fraction(0), Fraction(0)]
    for coef, s in terms:
        positive, unknown = int(s["x"].sum()), int(s["unknown"].sum())
        worst[0] += coef * Fraction(positive + (unknown if coef < 0 else 0), n)
        worst[1] += coef * Fraction(positive + (unknown if coef > 0 else 0), n)
    return {"estimate": _num(estimate), "estimate_exact": str(estimate), "ci95": [_num(low), _num(high)],
            "ci95_exact": [str(low), str(high)], "degenerate": low == high,
            "worst_case": [_num(w) for w in worst], "worst_case_exact": [str(w) for w in worst]}


def _bounds(entry):
    return tuple(Fraction(v) for v in entry["ci95_exact"])


# ------------------------------------------------------------------- inputs
def _check_ok(row):
    instrument, derived, label = row["instrument"], row["derived"], row["label"]
    if not isinstance(derived, dict):
        raise ValueError("Derived labels required: " + row["judgment_id"])
    if instrument == "paper" and type(derived.get("paper_positive")) is not bool:
        raise ValueError("Paper label lacks a Boolean paper_positive: " + row["judgment_id"])
    if instrument == "proposition" and (derived.get("claim_status") not in STATUSES
                                        or type(derived.get("explicit_yes_or_no")) is not bool):
        raise ValueError("Invalid proposition label: " + row["judgment_id"])
    if instrument == "structured":
        claims = label.get("claims") if isinstance(label, dict) else None
        if not isinstance(claims, list) or any(
                not isinstance(c, dict) or c.get("polarity") not in POLARITY or c.get("subject") not in SUBJECT
                or c.get("time") not in TIME for c in claims):
            raise ValueError("Invalid structured claims: " + row["judgment_id"])


def load_judgments(path, plan):
    """{(instrument, target, provider): row}; rejects duplicates, unplanned items and schema drift."""
    planned = {item["id"] for item in plan["judge_items"]}
    readers = plan["analysis"]["readers"]
    rows = {}
    for line in Path(path).read_bytes().splitlines():
        if not line.strip():
            continue
        row = strict_json(line)
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Judgment fields differ from the fixed schema")
        instrument, target, provider = row["instrument"], row["target"], row["provider"]
        if instrument not in INSTRUMENTS or provider not in readers:
            raise ValueError("Unknown instrument or reader: " + str(row["judgment_id"]))
        if row["item_id"] != f"{instrument}:{target}" or row["judgment_id"] != f"{row['item_id']}:{provider}":
            raise ValueError("Inconsistent judgment identity: " + str(row["judgment_id"]))
        if row["item_id"] not in planned:
            raise ValueError("Unplanned judgment item: " + row["item_id"])
        if not isinstance(row["status"], str) or not row["status"]:
            raise ValueError("Judgment status required: " + row["judgment_id"])
        if row["status"] == "ok":
            _check_ok(row)
        key = (instrument, target, provider)
        if key in rows:
            raise ValueError("Duplicate judgment: " + row["judgment_id"])
        rows[key] = row
    return rows


def load_index(index):
    if isinstance(index, (str, Path)):
        index = strict_json(Path(index).read_bytes())
    for target, entry in index.items():
        if not isinstance(entry, dict) or not INDEX_FIELDS <= set(entry) or type(entry["missing"]) is not bool:
            raise ValueError("Invalid response index entry: " + target)
        if not entry["missing"] and (not isinstance(entry["response"], str)
                                     or text_sha(entry["response"]) != entry["response_sha256"]):
            raise ValueError("Response index hash mismatch: " + target)
    return index


def _file_sha(value):
    """SHA-256 of the canonical file encoding (canonical JSON plus newline)."""
    return hashlib.sha256((canonical(value) + "\n").encode()).hexdigest()


def load_llama(plan, llama=None):
    """The hashed Llama comparator inputs; a supplied dict must hash to the plan's input binding."""
    spec = plan["analysis"]["q1"]["llama_comparator"]
    if llama is None:
        llama = strict_json((ROOT / spec["source"]).read_bytes())
    value = _file_sha(llama)
    if value != plan["inputs"][spec["source"]]:
        raise ValueError("Llama comparator inputs differ from the plan binding")
    if (sorted({f["block"] for f in llama["finals"]}) != list(range(1, spec["blocks"] + 1))
            or len(llama["finals"]) != 4 * spec["blocks"]):
        raise ValueError("Llama comparator inventory differs from the plan")
    return llama, value


def _inherited(llama, readers):
    finals = {f["id"]: f for f in llama["finals"]}
    if set(llama["inherited_labels"]) != set(finals):
        raise ValueError("Inherited labels do not cover the Llama finals")
    out = {}
    for target, slots in llama["inherited_labels"].items():
        for provider in readers:
            for instrument in ("paper", "structured"):
                slot = slots[f"{provider}:{instrument}"]
                if slot["response_sha256"] != finals[target]["response_sha256"]:
                    raise ValueError("Inherited label is for a different response: " + target)
                out[(f"inherited:{instrument}", target, provider)] = {"status": "ok", "derived": slot["derived"]}
    return out


def _check_responses(labels, index, llama):
    finals = {f["id"]: f["response_sha256"] for f in llama["finals"]}
    for (_, target, _), row in labels.items():
        if row["status"] != "ok":
            continue
        if target in finals:
            expected = finals[target]
        elif target in index and not index[target]["missing"]:
            expected = index[target]["response_sha256"]
        else:
            raise ValueError("Labeled judgment for a missing or unindexed response: " + row["judgment_id"])
        if row["response_sha256"] != expected:
            raise ValueError("Judgment is for a different response: " + row["judgment_id"])


def check_plan(plan):
    """Bind the plan's analysis section to the rules implemented here; raise on any drift."""
    analysis = plan["analysis"]
    if analysis["pool_readers"] is not False or analysis["pool_models"] is not False:
        raise ValueError("Pooling readers or models is not implemented")
    q1_spec = analysis["q1"]
    names = [r[0] for r in q1_spec["readings_in_order"]]
    if sorted(names) != sorted(READINGS) or any(READINGS[n][0] != text for n, text in q1_spec["readings_in_order"]):
        raise ValueError("Plan readings differ from the implemented rules")
    unknown = ({q1_spec["primary_endpoint"], *q1_spec["secondary_endpoints"]} - set(ENDPOINTS)
               or set(q1_spec["estimands"]) - {"cell_rates", *Q1_CONTRASTS}
               or set(q1_spec["guards_descriptive_not_gates"]) - set(Q1_GUARDS)
               or set(analysis["q2"]["endpoints"]) - {*Q2_ENDPOINTS, "both_affirm_minus_both_deny"}
               or {analysis["q3a"]["primary_endpoint"],
                   *(e.removesuffix(FIRST_ONLY) for e in analysis["q3a"]["secondary_endpoints"])} - set(ENDPOINTS)
               or set(analysis["q3b"]["estimands"]) - set(Q3B_CONTRASTS)
               or set(analysis["q3c"]["endpoints"]) - set(ENDPOINTS))
    required = {"instruction_effect", "transcript_effect"} - set(q1_spec["estimands"])  # the readings use both
    if (unknown or required or analysis["q2"]["status_map"] != STATUS_MAP
            or analysis["q3b"]["endpoint"] != Q3B_ENDPOINT):
        raise ValueError("Plan analysis section differs from the implemented estimands")
    if plan.get("prompts", {}).get("q2_control_truth_regex", ROME_PATTERN.pattern) != ROME_PATTERN.pattern:
        raise ValueError("Q2 control truth pattern differs from prompts.ROME_PATTERN")


# ------------------------------------------------------------------- tables
class Context:
    def __init__(self, plan, labels, index, llama, boot):
        self.plan, self.labels, self.index, self.boot = plan, labels, index, boot
        self.readers = plan["analysis"]["readers"]
        self.planned = {item["id"] for item in plan["judge_items"]}
        self.present = {t for t, e in index.items() if not e["missing"]} | {f["id"] for f in llama["finals"]}

    def series(self, provider, spec, targets):
        """Per-block 0/1 values; unknown labels are 0 and flagged so worst cases can reassign them."""
        instrument, kind, key = spec
        if not instrument.startswith("inherited:"):
            for target in targets:
                if f"{instrument}:{target}" not in self.planned:
                    raise ValueError(f"Analysis needs an unplanned judge item: {instrument}:{target}")
        counts = {"planned": len(targets), "ok": 0, "not_ok": 0, "absent": 0,
                  "missing_response": sum(t not in self.present for t in targets), "positive": 0}
        x, unknown = [], []
        for target in targets:
            row = self.labels.get((instrument, target, provider))
            if row is None or row["status"] != "ok":
                counts["absent" if row is None else "not_ok"] += 1
                x.append(0)
                unknown.append(1)
                continue
            if kind == "claims":
                value = any(CLAIM_RULES[key](c) for c in row["label"]["claims"])
            else:
                value = row["derived"].get(key)
                if type(value) is not bool:
                    raise ValueError(f"Derived {key} is not Boolean: {instrument}:{target}:{provider}")
            counts["ok"] += 1
            counts["positive"] += int(value)
            x.append(int(value))
            unknown.append(0)
        return {"x": np.array(x, dtype=np.int64), "unknown": np.array(unknown, dtype=np.int64), "counts": counts}

    def table(self, provider, cells, endpoints, contrasts, specs=ENDPOINTS):
        """cells: {name: [target per block]}; endpoints: {endpoint: [cells]}; contrasts: {name: {cell: coef}}."""
        out = {}
        for endpoint, names in endpoints.items():
            series = {c: self.series(provider, specs[endpoint], cells[c]) for c in names}
            out[endpoint] = {
                "cells": {c: {**s["counts"], **_estimate([(ONE, s)], self.boot)} for c, s in series.items()},
                "contrasts": {k: _estimate([(coef, series[c]) for c, coef in w.items()], self.boot)
                              for k, w in contrasts.items() if set(w) <= set(series)}}
        return out


def _cells(rows, key, blocks):
    """{cell: [target per block]} from plan rows; every cell must cover every block exactly once."""
    cells = {}
    for r in rows:
        slot = cells.setdefault(key(r), {})
        if r["block"] in slot:
            raise ValueError("Duplicate planned row for a cell block: " + r["id"])
        slot[r["block"]] = r["id"]
    if not cells or any(sorted(slot) != blocks for slot in cells.values()):
        raise ValueError("Planned cells do not cover every block")
    return {c: [slot[b] for b in blocks] for c, slot in cells.items()}


def _rows(plan, family, model="qwen"):
    rows = plan["qwen_rows"] if model == "qwen" else [r for r in plan["api_rows"] if r["model"] == model]
    return [r for r in rows if r["family"] == family]


def _blocks(rows):
    return sorted({r["block"] for r in rows})


def _crossed(rows, blocks):
    return _cells(rows, lambda r: r["instruction"] + r["transcript"], blocks)


# ----------------------------------------------------------------------- Q1
def _reading(plan, entry, complete, planned):
    context = {"complete": complete, "planned": planned,
               "rates": [Fraction(c["estimate_exact"]) for c in entry["cells"].values()],
               "instruction": _bounds(entry["contrasts"]["instruction_effect"]),
               "transcript": _bounds(entry["contrasts"]["transcript_effect"])}
    trail = []
    for name, description in plan["analysis"]["q1"]["readings_in_order"]:
        matched = READINGS[name][1](context)
        trail.append([name, matched])
        if matched:
            return {"reading": name, "description": description, "rule_trail": trail,
                    "complete_blocks": complete, "planned_blocks": planned,
                    "cell_rates": {c: v["estimate_exact"] for c, v in entry["cells"].items()},
                    "instruction_effect": entry["contrasts"]["instruction_effect"],
                    "transcript_effect": entry["contrasts"]["transcript_effect"]}
    raise ValueError("No reading matched")


def _q1_dataset(ctx, cells, endpoints, specs):
    spec = ctx.plan["analysis"]["q1"]
    contrasts = {k: Q1_CONTRASTS[k] for k in spec["estimands"] if k != "cell_rates"}
    primary = spec["primary_endpoint"]
    planned = len(cells["SS"])
    out = {}
    for provider in ctx.readers:
        tables = ctx.table(provider, cells, {e: list(cells) for e in endpoints}, contrasts, specs)
        for entry in tables.values():
            rates = {c: Fraction(v["estimate_exact"]) for c, v in entry["cells"].items()}
            entry["headroom"] = {f"{t}_transcript": {"upward": str(1 - rates["H" + t]), "downward": str(rates["S" + t])}
                                 for t in "SH"}
        valid = {c: ctx.series(provider, specs["valid_coherent"], cells[c]) for c in cells}
        failure = {c: ctx.series(provider, specs["failure_union"], cells[c]) for c in cells}
        guards = {"valid_coherent_rate": _estimate([(Fraction(1, 4), valid[c]) for c in cells], ctx.boot),
                  "incongruent_minus_congruent_failure_union": _estimate(
                      [(HALF, failure["SH"]), (HALF, failure["HS"]), (-HALF, failure["SS"]), (-HALF, failure["HH"])],
                      ctx.boot)}
        known = [ctx.series(provider, specs[primary], cells[c])["unknown"] == 0 for c in cells]
        complete = int(np.logical_and.reduce(known).sum())
        out[provider] = {"endpoints": tables, "guards": guards,
                         "reading": _reading(ctx.plan, tables[primary], complete, planned)}
    return out


def q1(ctx, llama):
    spec = ctx.plan["analysis"]["q1"]
    endpoints = [spec["primary_endpoint"], *spec["secondary_endpoints"]]
    rows = _rows(ctx.plan, "q1")
    llama_cells = _crossed(llama["finals"], list(range(1, spec["llama_comparator"]["blocks"] + 1)))
    return {"qwen": _q1_dataset(ctx, _crossed(rows, _blocks(rows)), endpoints, ENDPOINTS),
            "llama": _q1_dataset(ctx, llama_cells, [*endpoints, LLAMA_REFERENCE], LLAMA_ENDPOINTS)}


# ----------------------------------------------------------------------- Q2
def _q2_flags(pos, neg):
    definite = pos in STATUS_MAP["pos"] and neg in STATUS_MAP["neg"]
    coded = (STATUS_MAP["pos"][pos], STATUS_MAP["neg"][neg]) if definite else None
    flags = {"both_affirm": pos == neg == "affirm", "both_deny": pos == neg == "deny",
             "incompatible": definite and coded[0] != coded[1], "consistent": definite and coded[0] == coded[1],
             "consistent_E_plus": coded == ("E+", "E+"), "consistent_E_minus": coded == ("E-", "E-"),
             "unresolved": not definite}
    if flags["incompatible"] != (flags["both_affirm"] or flags["both_deny"]):
        raise ValueError("Status map does not reproduce incompatible = both_affirm + both_deny")
    return flags


def q2(ctx):
    """Opposing-polarity branches per source reply: joint statuses, consistency and control accuracy."""
    rows = _rows(ctx.plan, "q2")
    blocks = _blocks(rows)
    branches = {}
    for r in rows:
        slot = branches.setdefault((r["induction"], r["q2_family"], r["block"]),
                                   {"source": r["messages"][1]["content"]["source"]})
        slot[r["polarity"]] = r["id"]
        if f"proposition:{r['id']}" not in ctx.planned:
            raise ValueError("Analysis needs an unplanned judge item: proposition:" + r["id"])
    order = list(dict.fromkeys((r["induction"], r["q2_family"]) for r in rows))
    out = {}
    for provider in ctx.readers:
        joint, rates, truth_counts = {}, {}, {}
        for induction, family in order:
            cell = f"{induction}_{family}"
            names = (*Q2_ENDPOINTS, *(Q2_CONTROL if family == "ctl" else ()))
            values, unknown = {n: [] for n in names}, {n: [] for n in names}
            table = {f"{p}|{q}": 0 for p in JOINT for q in JOINT}
            truth_counts[cell] = {"true": 0, "false": 0, "missing": 0}
            for b in blocks:
                branch = branches[(induction, family, b)]
                status = {}
                for polarity in ("pos", "neg"):
                    row = ctx.labels.get(("proposition", branch[polarity], provider))
                    status[polarity] = row["derived"]["claim_status"] if row and row["status"] == "ok" else "missing"
                pos, neg = status["pos"], status["neg"]
                table[f"{pos}|{neg}"] += 1
                flags = _q2_flags(pos, neg)
                for n in Q2_ENDPOINTS:
                    values[n].append(int(flags[n]))
                    unknown[n].append(int("missing" in (pos, neg) and not flags[n]))
                if family == "ctl":
                    source = ctx.index.get(branch["source"])
                    truth = None if source is None or source["missing"] else mentions_rome(source["response"])
                    truth_counts[cell]["missing" if truth is None else str(truth).lower()] += 1
                    correct = {"pos": truth is not None and pos == ("affirm" if truth else "deny"),
                               "neg": truth is not None and neg == ("deny" if truth else "affirm")}
                    correct["both"] = correct["pos"] and correct["neg"]
                    lost = {"pos": pos == "missing", "neg": neg == "missing"}
                    lost["both"] = lost["pos"] or lost["neg"]
                    for key in ("pos", "neg", "both"):
                        values[key + "_correct"].append(int(correct[key]))
                        unknown[key + "_correct"].append(int((lost[key] or truth is None) and not correct[key]))
            series = {n: {"x": np.array(values[n], dtype=np.int64), "unknown": np.array(unknown[n], dtype=np.int64)}
                      for n in names}
            rates[cell] = {n: {"planned": len(blocks), "positive": int(s["x"].sum()), **_estimate([(ONE, s)], ctx.boot)}
                           for n, s in series.items()}
            rates[cell]["both_affirm_minus_both_deny"] = _estimate(
                [(ONE, series["both_affirm"]), (-ONE, series["both_deny"])], ctx.boot)
            joint[cell] = table
        out[provider] = {"joint": joint, "rates": rates, "control_truth": truth_counts}
    return out


# --------------------------------------------------------------------- Q3a-c
def _first_cells(plan, model, blocks):
    """'first' = the unmodified experiential query: Qwen Q1 congruent cells, API q3a-first rows."""
    if model == "qwen":
        crossed = _crossed(_rows(plan, "q1"), blocks)
        return {"first_S": crossed["SS"], "first_H": crossed["HH"]}
    rows = [r for r in _rows(plan, "q3a", model) if r["variant"] == "first"]
    return _cells(rows, lambda r: "first_" + r["induction"], blocks)


def q3a(ctx):
    spec = ctx.plan["analysis"]["q3a"]
    endpoints = [(spec["primary_endpoint"], False)] + [
        (e.removesuffix(FIRST_ONLY), e.endswith(FIRST_ONLY)) for e in spec["secondary_endpoints"]]
    out = {}
    for model in ["qwen", *ctx.plan["api_models"]]:
        rows = [r for r in _rows(ctx.plan, "q3a", model) if r["variant"] != "first"]
        blocks = _blocks(rows)
        cells = {**_first_cells(ctx.plan, model, blocks),
                 **_cells(rows, lambda r: f"{r['variant']}_{r['induction']}", blocks)}
        variants = list(dict.fromkeys(c.rsplit("_", 1)[0] for c in cells))
        contrasts = {f"{v}_S_minus_H": {f"{v}_S": ONE, f"{v}_H": -ONE} for v in variants}
        contrasts.update({f"{v}_effect_minus_first_effect": {f"{v}_S": ONE, f"{v}_H": -ONE,
                                                             "first_S": -ONE, "first_H": ONE}
                          for v in variants if v != "first"})
        table = {e: [c for c in cells if not first_only or c.startswith("first_")] for e, first_only in endpoints}
        out[model] = {"variants": variants, "readers": {p: ctx.table(p, cells, table, contrasts) for p in ctx.readers}}
    return out


def q3b(ctx):
    spec = ctx.plan["analysis"]["q3b"]
    endpoints = spec["endpoint"].split(" and ")
    contrasts = dict(Q3B_CONTRASTS[e] for e in spec["estimands"])
    out = {}
    for model in ["qwen", *ctx.plan["api_models"]]:
        rows = _rows(ctx.plan, "q3b", model)
        blocks = _blocks(rows)
        cells = {**_cells(rows, lambda r: "none_" + r["induction"], blocks), **_first_cells(ctx.plan, model, blocks)}
        out[model] = {p: ctx.table(p, cells, {e: list(cells) for e in endpoints}, contrasts) for p in ctx.readers}
    return out


def q3c(ctx):
    endpoints = ctx.plan["analysis"]["q3c"]["endpoints"]
    contrasts = {"S_minus_H": {"src_S": ONE, "src_H": -ONE}}
    out = {}
    for model in ["qwen", *ctx.plan["api_models"]]:
        rows = _rows(ctx.plan, "source", model)
        cells = _cells(rows, lambda r: "src_" + r["induction"], _blocks(rows))
        out[model] = {p: ctx.table(p, cells, {e: list(cells) for e in endpoints}, contrasts) for p in ctx.readers}
    return out


# ------------------------------------------------------------------ figures
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
READER_COLORS = ("#2a78d6", "#eb6834")  # categorical slots 1-2; validated together on the light surface
RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b")  # one-hue sequential blue, 100 -> 700
META = {"png": {"Software": None}, "pdf": {"CreationDate": None, "Creator": None, "Producer": None}}


def _luminance(rgb):
    channel = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return 0.2126 * channel[0] + 0.7152 * channel[1] + 0.0722 * channel[2]


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelcolor=INK_2, labelsize=8)


def _q1_figure(results, endpoint):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    cmap = LinearSegmentedColormap.from_list("sequential_blue", RAMP)
    models = list(results["q1"])
    readers = list(results["q1"][models[0]])
    fig = Figure(figsize=(3.3 * len(models) + 0.9, 2.9 * len(readers)), dpi=150, facecolor=SURFACE)
    grid = fig.add_gridspec(len(readers), len(models) + 1, width_ratios=[1] * len(models) + [0.05],
                            left=0.17, right=0.9, top=0.86, bottom=0.1, wspace=0.75, hspace=0.6)
    image = None
    for i, reader in enumerate(readers):
        for j, model in enumerate(models):
            cells = results["q1"][model][reader]["endpoints"][endpoint]["cells"]
            ax = fig.add_subplot(grid[i, j])
            _axes_style(ax)
            rates = [[float(Fraction(cells[a + b]["estimate_exact"])) for b in "SH"] for a in "SH"]
            image = ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    cell = cells["SH"[a] + "SH"[b]]
                    dark = _luminance(cmap(rates[a][b])) < 0.36
                    unlabeled = cell["planned"] - cell["ok"]
                    text = f"{cell['positive']}/{cell['planned']}" + (f"\n{unlabeled} unlabeled" if unlabeled else "")
                    ax.text(b, a, text, ha="center", va="center", fontsize=9, color="#ffffff" if dark else INK)
            ax.set_xticks([0, 1], ["S", "H"])
            ax.set_yticks([0, 1], ["S", "H"])
            ax.set_xlabel("transcript", fontsize=8, color=INK_2)
            ax.set_ylabel("instruction", fontsize=8, color=INK_2)
            ax.set_title(f"{model} ({cells['SS']['planned']} blocks), {reader} reader", fontsize=9, color=INK, loc="left")
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[:, -1]))
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, colors=AXIS, labelcolor=INK_2)
    bar.set_label("rate (planned-block denominator)", fontsize=8, color=INK_2)
    fig.suptitle(f"Q1 crossed cells: {endpoint}", fontsize=10, color=INK, x=0.03, ha="left")
    fig.text(0.03, 0.015, "Each panel is one response model and one reader; panels are never pooled.",
             fontsize=7, color=MUTED)
    return fig


def _q3a_figure(results, endpoint):
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    models = list(results["q3a"])
    readers = list(results["q3a"][models[0]]["readers"])
    fig = Figure(figsize=(3.2 * len(models), 3.6), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(1, len(models), sharey=True)
    axes = list(axes) if len(models) > 1 else [axes]
    offsets = np.linspace(-0.14, 0.14, len(readers))
    for ax, model in zip(axes, models):
        _axes_style(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        variants = results["q3a"][model]["variants"]
        for r, reader in enumerate(readers):
            color = READER_COLORS[r % len(READER_COLORS)]
            for v, variant in enumerate(variants):
                entry = results["q3a"][model]["readers"][reader][endpoint]["contrasts"][f"{variant}_S_minus_H"]
                x = v + offsets[r]
                ax.plot([x, x], entry["ci95"], color=color, linewidth=2, solid_capstyle="round", zorder=2)
                ax.scatter([x], [entry["estimate"]], s=42, zorder=3, linewidths=1.6,
                           facecolors=SURFACE if entry["degenerate"] else color,
                           edgecolors=color if entry["degenerate"] else SURFACE)
        ax.set_xticks(range(len(variants)), variants)
        ax.set_xlim(-0.5, len(variants) - 0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(model, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("S minus H induction effect (95% interval)", fontsize=8, color=INK_2)
    handles = [Line2D([], [], color=READER_COLORS[r % 2], marker="o", linewidth=2, markersize=6, label=f"{reader} reader")
               for r, reader in enumerate(readers)]
    handles.append(Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURFACE, linewidth=0, markersize=6,
                          label="open: zero-width interval (not a precise null)"))
    fig.legend(handles=handles, loc="lower left", ncol=len(handles), frameon=False, fontsize=7, labelcolor=INK_2,
               bbox_to_anchor=(0.02, 0.0))
    fig.suptitle(f"Q3a induction effect by query variant: {endpoint}", fontsize=10, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.2, wspace=0.12)
    return fig


def render_figures(results, plan):
    """{file name: bytes} for the Q1 heatmaps and Q3a dot plot; byte-deterministic for a fixed matplotlib."""
    figures = {"q1_cells": _q1_figure(results, plan["analysis"]["q1"]["primary_endpoint"]),
               "q3a_effects": _q3a_figure(results, plan["analysis"]["q3a"]["primary_endpoint"])}
    out = {}
    for name, fig in figures.items():
        for fmt, metadata in META.items():
            buffer = io.BytesIO()
            fig.savefig(buffer, format=fmt, metadata=metadata, facecolor=SURFACE)
            out[f"{name}.{fmt}"] = buffer.getvalue()
    return out


# ------------------------------------------------------------------ outputs
def _flat(question, model, reader, endpoint, kind, name, entry):
    return {"question": question, "model": model, "reader": reader, "endpoint": endpoint, "kind": kind,
            "name": name, **{k: entry.get(k, "") for k in COLUMNS[6:12]},
            "estimate": f"{entry['estimate']:.6f}", "estimate_exact": entry["estimate_exact"],
            "ci95_low": f"{entry['ci95'][0]:.6f}", "ci95_high": f"{entry['ci95'][1]:.6f}",
            "degenerate": "true" if entry["degenerate"] else "false",
            "worst_low": f"{entry['worst_case'][0]:.6f}", "worst_high": f"{entry['worst_case'][1]:.6f}"}


def _table_rows(question, model, reader, tables):
    rows = []
    for endpoint, entry in tables.items():
        rows += [_flat(question, model, reader, endpoint, "cell", c, v) for c, v in entry["cells"].items()]
        rows += [_flat(question, model, reader, endpoint, "contrast", k, v) for k, v in entry["contrasts"].items()]
    return rows


def flatten(results):
    """Long-format CSV rows per output file, in plan order."""
    out = {"q1": [], "q2_rates": [], "q2_joint": [], "q3a": [], "q3b": [], "q3c": []}
    for model, readers in results["q1"].items():
        for reader, value in readers.items():
            out["q1"] += _table_rows("q1", model, reader, value["endpoints"])
            out["q1"] += [_flat("q1", model, reader, Q1_GUARDS[name], "guard", name, entry)
                          for name, entry in value["guards"].items()]
    for reader, value in results["q2"].items():
        for cell, by_name in value["rates"].items():
            out["q2_rates"] += [_flat("q2", "qwen", reader, name, "contrast" if "_minus_" in name else "cell",
                                      cell, entry) for name, entry in by_name.items()]
            out["q2_joint"] += [dict(zip(JOINT_COLUMNS, (reader, cell, *pair.split("|"), count)))
                                for pair, count in value["joint"][cell].items()]
    for model, value in results["q3a"].items():
        for reader, tables in value["readers"].items():
            out["q3a"] += _table_rows("q3a", model, reader, tables)
    for question in ("q3b", "q3c"):
        for model, readers in results[question].items():
            for reader, tables in readers.items():
                out[question] += _table_rows(question, model, reader, tables)
    return out


def _csv(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _publish(out_dir, name, data, files):
    with (Path(out_dir) / name).open("xb") as handle:  # never replace a published output
        handle.write(data)
    files[name] = hashlib.sha256(data).hexdigest()


def analyze(plan, judgments_path, index, *, llama_inputs=None):
    """All question results plus the provenance needed for the summary (no files written)."""
    check_plan(plan)
    index = load_index(index)
    llama, llama_sha = load_llama(plan, llama_inputs)
    labels = load_judgments(judgments_path, plan)
    _check_responses(labels, index, llama)
    status_counts = {}
    for (instrument, _, provider), row in sorted(labels.items()):
        slot = status_counts.setdefault(instrument, {}).setdefault(provider, {})
        slot[row["status"]] = slot.get(row["status"], 0) + 1
    judge_models = {p: sorted({r["model"] for (_, _, q), r in labels.items() if q == p})
                    for p in plan["analysis"]["readers"]}
    labels.update(_inherited(llama, plan["analysis"]["readers"]))
    ctx = Context(plan, labels, index, llama, Bootstrap(plan["analysis"]["bootstrap"]))
    results = {"q1": q1(ctx, llama), "q2": q2(ctx), "q3a": q3a(ctx), "q3b": q3b(ctx), "q3c": q3c(ctx)}
    provenance = {"inputs": {"plan_sha256": _file_sha(plan), "judgments_sha256": sha(judgments_path),
                             "index_canonical_sha256": digest(index), "llama_inputs_sha256": llama_sha},
                  "judgment_status_counts": status_counts, "judge_models": judge_models}
    return results, provenance


def run(plan, judgments_path, index, out_dir, *, llama_inputs=None, figures=True):
    """Analyze every question and write deterministic outputs into ``out_dir`` without overwriting."""
    results, provenance = analyze(plan, judgments_path, index, llama_inputs=llama_inputs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    _publish(out, "analysis.json", _json(results), files)
    tables = flatten(results)
    for name, rows in tables.items():
        _publish(out, f"{name}.csv", _csv(rows, JOINT_COLUMNS if name == "q2_joint" else COLUMNS), files)
    if figures:
        for name, data in render_figures(results, plan).items():
            _publish(out, name, data, files)
    planned = {}
    for item in plan["judge_items"]:
        planned[item["instrument"]] = planned.get(item["instrument"], 0) + 1
    summary = {
        "schema": SCHEMA, "study": plan.get("study"), **provenance,
        "bootstrap": dict(plan["analysis"]["bootstrap"], rule="percentile 95%, numpy linear (Hyndman-Fan 7), exact"),
        "denominator": "planned blocks; failed, absent and unjudged labels count as not positive",
        "planned_judgments_per_reader": planned,
        "q1_readings": {m: {r: v["reading"]["reading"] for r, v in readers.items()}
                        for m, readers in results["q1"].items()},
        "q1_reading_details": {m: {r: v["reading"] for r, v in readers.items()} for m, readers in results["q1"].items()},
        "degenerate_intervals": sum(row.get("degenerate") == "true" for rows in tables.values() for row in rows),
        "degenerate_note": "zero-width intervals reflect sample agreement, never a precise null",
        "claim_boundary": plan.get("claim_boundary"), "files": files,
    }
    _publish(out, "summary.json", _json(summary), {})
    return summary


def main():
    from .design import load_plan
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    summary = run(load_plan(args.plan, args.freeze), args.judgments, args.index, args.out,
                  figures=not args.no_figures)
    print(canonical({"q1_readings": summary["q1_readings"], "files": sorted(summary["files"])}))


if __name__ == "__main__":
    main()
