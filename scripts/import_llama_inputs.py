"""Copy the public Llama crossed-qualification finals into a hashed input file.

Reads CONSCIOUS at a pinned commit with ``git show`` (no working-tree access),
verifies each record's response hash, and keeps the inherited paper-rubric and
structured (qualification codebook) labels for the comparator tables.
Run once before the freeze; the output is bound by the plan's input hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfref_scaling.common import canonical, text_sha  # noqa: E402
from selfref_scaling.design import CONSCIOUS_COMMIT, LLAMA_INPUTS_PATH  # noqa: E402
from selfref_scaling.prompts import EXPERIENTIAL_QUERY, HISTORY, SELF  # noqa: E402

RELEASE = "data/instruction_state_qualification/crossed_v1_20261001"
NAMES = {"self": "S", "history": "H"}
KEEP = ("inclusive_current_assertion", "explicit_current_assertion", "paper_positive",
        "mixed_current_assertion", "denied", "uncertain", "reported_context_conflict",
        "valid_coherent", "failure_union")


def show(repo, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{CONSCIOUS_COMMIT}:{path}"])


def blob(repo, path):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse",
                                    f"{CONSCIOUS_COMMIT}:{path}"], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conscious", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(LLAMA_INPUTS_PATH))
    args = parser.parse_args()
    finals = []
    for block in range(1, 13):
        for instruction in ("self", "history"):
            for transcript in ("self", "history"):
                path = f"{RELEASE}/raw/generations/block-{block:02d}-{instruction}-{transcript}.json"
                raw = show(args.conscious, path)
                record = json.loads(raw)
                messages = record["messages"]
                expected = [SELF if instruction == "self" else HISTORY, None, EXPERIENTIAL_QUERY]
                if (len(messages) != 3 or messages[0]["content"] != expected[0]
                        or messages[2]["content"] != expected[2]
                        or text_sha(record["response"]) != record["response_sha256"]
                        or record["status"] != "complete"):
                    raise ValueError("Unexpected Llama record: " + path)
                finals.append({
                    "id": f"llama-b{block:02d}-q1-{NAMES[instruction]}{NAMES[transcript]}",
                    "block": block, "instruction": NAMES[instruction], "transcript": NAMES[transcript],
                    "query": messages[2]["content"], "response": record["response"],
                    "response_sha256": record["response_sha256"], "cap_hit": record["cap_hit"],
                    "source_path": path, "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_git_blob": blob(args.conscious, path)})
    judgments_path = f"{RELEASE}/judges/judgments.jsonl"
    raw = show(args.conscious, judgments_path)
    by_item = {}
    for line in raw.decode().splitlines():
        row = json.loads(line)
        if row.get("phase") != "target" or row.get("status") != "ok":
            continue
        item = row["item_id"]
        if not item.startswith("block-"):
            continue
        _, number, instruction, transcript = item.split("-")
        key = f"llama-b{int(number):02d}-q1-{NAMES[instruction]}{NAMES[transcript]}"
        derived = {k: row["derived"][k] for k in KEEP if k in row["derived"]}
        slot = by_item.setdefault(key, {})
        name = f"{row['provider']}:{row['instrument']}"
        if name in slot:
            raise ValueError("Duplicate inherited judgment: " + key + " " + name)
        slot[name] = {"derived": derived, "response_sha256": row["response_sha256"]}
    ids = {f["id"] for f in finals}
    if set(by_item) != ids or any(len(v) != 4 for v in by_item.values()):
        raise ValueError("Inherited judgments do not cover every final with four labels")
    for final in finals:
        for label in by_item[final["id"]].values():
            if label["response_sha256"] != final["response_sha256"]:
                raise ValueError("Inherited judgment is for a different response")
    value = {"schema": "llama_crossed_inputs_v1", "conscious_commit": CONSCIOUS_COMMIT,
             "release": RELEASE, "model": "meta-llama/Llama-3.3-70B-Instruct",
             "finals": finals, "inherited_labels": {k: by_item[k] for k in sorted(by_item)},
             "judgments_source": {"path": judgments_path, "sha256": hashlib.sha256(raw).hexdigest(),
                                  "git_blob": blob(args.conscious, judgments_path)},
             "note": "Inherited structured labels use the qualification codebook; this study re-scores "
                     "the same responses with the A1 codebook for the comparator."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(canonical(value) + "\n")
    print(canonical({"finals": len(finals), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
