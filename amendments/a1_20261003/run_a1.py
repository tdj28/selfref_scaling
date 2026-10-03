"""Amendment A1 launcher (see docs/AMENDMENT_A1_20261003.md).

Uses the frozen ``selfref_scaling.controller`` unchanged except for two
documented substitutions:

1. The cheap pod runs ``tests/test_qwen_backend.py`` plus the corrected
   ``amendments/a1_20261003/test_cuda_smoke_a1.py`` instead of the frozen
   ``tests/test_cuda_smoke.py`` (whose tiny-model setup offloaded to disk).
2. Every A1 pod uses its own run root ``out/scaling-qwen-20261003/a1`` so the
   failed original cheap ledger stays intact, and the main pod's prior GPU
   spend includes that original attempt (read from its own ledger at the
   original freeze) in addition to the A1 cheap pod (the frozen cheap gate).

The plan, pod runner, controller logic and every bound file are those of the
scientific freeze; ``design.load_plan`` verifies them at the A1 commit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from selfref_scaling import common, controller as C, design  # noqa: E402
from selfref_scaling.budget import EventLedger, _number

ORIGINAL_FREEZE = "d4b7d8b01d29417c9ad1a595f82517dfa66dee06"
ORIGINAL_ROOT = C.OUT_ROOT
A1_ROOT = C.OUT_ROOT / "a1"
A1_TESTS = ("tests/test_qwen_backend.py", "amendments/a1_20261003/test_cuda_smoke_a1.py")
_frozen_worker_script, _frozen_checked_budget = C.worker_script, C.checked_budget


def original_cheap_cost(plan_hash):
    """The failed original cheap attempt: closed (GET 404) and its compute bound."""
    ledger = EventLedger(ORIGINAL_ROOT / "controller" / "cheap" / "events.jsonl", plan_hash, ORIGINAL_FREEZE, [])
    closed = next((r for r in ledger.read() if r["id"] == "closed"), None)
    if closed is None or closed["data"].get("get_status") != 404:
        raise ValueError("Original cheap attempt is not verifiably closed")
    return _number(closed["data"]["compute_upper_bound_usd"])


def a1_worker_script(kind, plan_relative, freeze, deadline, hardware=None):
    script = _frozen_worker_script(kind, plan_relative, freeze, deadline, hardware)
    if kind != "cheap":
        return script
    python = C.REMOTE + "/venv/bin/python"
    frozen = "SELFREF_REQUIRE_CUDA=1 " + shlex.join([python, "-m", "pytest", "-q", "-rs", *C.CHEAP_TESTS])
    amended = ("SELFREF_REQUIRE_CUDA=1 SELFREF_RECEIPT_DIR=" + C.REMOTE + "/out "
               + shlex.join([python, "-m", "pytest", "-q", "-rs", *A1_TESTS]))
    if script.count(frozen) != 1:
        raise ValueError("Frozen cheap command not found exactly once")
    return script.replace(frozen, amended)


def install(plan_hash):
    """Patch the two documented substitutions into the frozen controller module."""
    carried = original_cheap_cost(plan_hash)

    def checked_budget(budget):
        prior, reserve = _frozen_checked_budget(budget)
        return prior + carried, reserve
    C.worker_script = a1_worker_script
    C.checked_budget = checked_budget
    return carried


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kind", choices=C.KINDS, required=True)
    parser.add_argument("--action", choices=("launch", "status", "retrieve", "monitor", "terminate", "reconcile"),
                        required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.freeze) or args.freeze == ORIGINAL_FREEZE:
        parser.error("--freeze must be the full A1 commit, not the original freeze")
    plan = common.ROOT / design.PLAN_PATH
    carried = install(common.sha(plan))
    if not args.launch:
        print(json.dumps({"dry_run": True, "kind": args.kind, "action": args.action, "root": str(A1_ROOT),
                          "carried_original_cheap_usd": str(carried), "cheap_tests": list(A1_TESTS)}))
        return
    api = C.RunPodV2(os.environ.get("RUNPOD_API_KEY"), writable=True)
    controller = C.Controller(plan, args.freeze, A1_ROOT, args.kind, api)
    print(json.dumps(getattr(controller, args.action)(), sort_keys=True))


if __name__ == "__main__":
    main()
