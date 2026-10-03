"""POST-HOC, EXPLORATORY (defined after outcomes were seen; not part of the frozen analysis).

The frozen Q3a endpoints count only claims coded with time "current". The base
codebook codes explicitly imagined or roleplayed claims as "hypothetical", so
the frozen endpoints cannot register experience claims voiced by the fictional
character. This script counts, per response model, Q3a variant, induction and
reader, the answers containing at least one ASSERTED claim attributed to the
character at ANY time, and separately to any non-reader subject at any time.
It reads the frozen judgments and plan; it changes nothing frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", type=Path, default=Path("data/plan_20261003/PLAN.json"))
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    labels = {}
    for line in args.judgments.read_text().splitlines():
        row = json.loads(line)
        if row["instrument"] == "structured" and row["status"] == "ok":
            labels[(row["target"], row["provider"])] = row["label"]["claims"]
    rows = [dict(r, model="qwen") for r in plan["qwen_rows"] if r["family"] == "q3a"]
    rows += [r for r in plan["api_rows"] if r["family"] == "q3a"]
    # Qwen's first-person variant is Q1's congruent cells, as in the frozen analysis.
    rows += [dict(r, model="qwen", variant="first", induction=r["instruction"]) for r in plan["qwen_rows"]
             if r["family"] == "q1" and r["instruction"] == r["transcript"]]
    counts = {}
    for r in rows:
        for provider in ("openai", "anthropic"):
            key = (r["model"], r["variant"], r["induction"], provider)
            slot = counts.setdefault(key, {"planned": 0, "labeled": 0, "character_assertion_any_time": 0,
                                           "non_reader_assertion_any_time": 0})
            slot["planned"] += 1
            claims = labels.get((r["id"], provider))
            if claims is None:
                continue
            slot["labeled"] += 1
            asserted = [c for c in claims if c["polarity"] == "asserted"]
            slot["character_assertion_any_time"] += any(c["subject"] == "character" for c in asserted)
            slot["non_reader_assertion_any_time"] += any(c["subject"] != "reader_user" for c in asserted)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model", "variant", "induction", "reader", "planned", "labeled",
                         "character_assertion_any_time", "non_reader_assertion_any_time"])
        for (model, variant, induction, reader), slot in sorted(counts.items()):
            writer.writerow([model, variant, induction, reader, slot["planned"], slot["labeled"],
                             slot["character_assertion_any_time"], slot["non_reader_assertion_any_time"]])
    print(args.out.read_text())


if __name__ == "__main__":
    main()
