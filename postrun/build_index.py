"""Post-run helper: write the frozen ``outcomes.build_index`` result as JSON.

No new logic: it loads the frozen plan (verified at the given commit), the API
results through the frozen loader and the hashed Llama inputs, calls the
frozen index builder and writes canonical JSON to a new file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfref_scaling import api_generate, design, outcomes  # noqa: E402
from selfref_scaling.common import ROOT, canonical, sha, strict_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", type=Path, default=ROOT / design.PLAN_PATH)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True, help="final retrieval directory of the main pod")
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = design.load_plan(args.plan, args.freeze)
    llama = strict_json((ROOT / design.LLAMA_INPUTS_PATH).read_bytes())
    index = outcomes.build_index(plan, args.qwen_dir, api_generate.load_api_results(args.api_root, plan), llama)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(canonical(index) + "\n")
    missing = sum(1 for v in index.values() if v["missing"])
    print(canonical({"targets": len(index), "missing": missing, "sha256": sha(args.out)}))


if __name__ == "__main__":
    main()
