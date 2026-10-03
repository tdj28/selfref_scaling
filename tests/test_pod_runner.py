from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from selfref_scaling import design
from selfref_scaling.common import canonical
from selfref_scaling.pod_runner import Worker

HARDWARE = design.HARDWARE["main"][0]


def small_plan(blocks=(1, 2), cap=5, batch=4):
    plan = design.build_plan()
    keep = [r for r in plan["qwen_rows"] if r["block"] in blocks]
    for r in keep:
        if "cap" in r:
            r["cap"] = cap
    for phase in ("sources", "q1", "q2", "q3b", "q3a", "q4"):
        members = [r for r in keep if r["phase"] == phase]
        for i, r in enumerate(members):
            r["batch"] = f"{phase}-{i // batch + 1:02d}"
    plan["qwen_rows"] = keep
    plan["generation"]["batch_size"] = batch
    plan["stage0"]["neutral_cap"] = 4
    plan["stage0"]["coherence_min_correct"] = 0  # a random tiny model cannot answer the neutral questions
    return plan


def make_worker(tmp_path, backend, plan, *, deadline_hours=2, clock=None):
    path = tmp_path / "PLAN.json"
    path.write_text(canonical(plan) + "\n")
    deadline = (datetime.now(timezone.utc) + timedelta(hours=deadline_hours)).isoformat()
    kwargs = {"clock": clock} if clock else {}
    return Worker(plan, path, "0" * 40, tmp_path / "out", deadline, None, HARDWARE,
                  factory=lambda: backend, allow_test=True, heartbeat_seconds=3600, **kwargs)


def test_full_small_run_resolves_sources_and_captures(tmp_path, tiny_backend):
    plan = small_plan()
    worker = make_worker(tmp_path, tiny_backend, plan)
    summary = worker.execute()
    assert summary["stage0_pass"] and summary["result"] == "complete"
    out = tmp_path / "out"
    planned = {r["id"] for r in plan["qwen_rows"]}
    assert set(worker.completed()) == planned | {"stage0"}
    final = json.loads((out / "generations" / "qwen-b01-q1-SH.json").read_text())
    source = json.loads((out / "generations" / "qwen-b01-src-H.json").read_text())
    assert final["messages"][1] == {"role": "assistant", "content": source["response"]}
    assert final["messages"][0]["content"] == plan["prompts"]["S"]
    assert final["seed"] == json.loads((out / "generations" / "qwen-b01-q1-HH.json").read_text())["seed"]
    noinstr = json.loads((out / "generations" / "qwen-b02-q3b-none-S.json").read_text())
    assert noinstr["messages"][0]["role"] == "assistant" and len(noinstr["messages"]) == 2
    meta = json.loads((out / "states" / "qwen-b01-q4-SS.json").read_text())
    assert meta["shape"][0] == len(tiny_backend.text.layers) + 1 and meta["positions"][0] == "boundary"
    assert worker.execute() == summary  # idempotent: DONE short-circuits, nothing regenerated


def test_empty_source_blocks_dependents(tmp_path, tiny_backend):
    class EmptySource:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def generate_batch(self, rows, **kw):
            records = self.inner.generate_batch(rows, **kw)
            for r in records:
                if r["id"] == "qwen-b01-src-S":
                    r["response"] = ""
            return records
    plan = small_plan(blocks=(1,))
    worker = make_worker(tmp_path, EmptySource(tiny_backend), plan)
    worker.execute()
    done = worker.completed()
    assert done["qwen-b01-q1-SS"]["missing"] == "blocked_empty_source"
    assert done["qwen-b01-q1-HS"]["missing"] == "blocked_empty_source"
    assert "path" in done["qwen-b01-q1-SH"] and "missing" not in done["qwen-b01-q1-SH"]
    assert done["qwen-b01-q4-SS"]["missing"] == "blocked_missing_generation"


def test_unresolved_dispatch_is_never_regenerated(tmp_path, tiny_backend):
    plan = small_plan(blocks=(1,))
    worker = make_worker(tmp_path, tiny_backend, plan)
    worker.stage0()
    worker.step_seconds, worker.prefill_per_token = 0.0, 0.0
    rows = [r for r in plan["qwen_rows"] if r["batch"] == "sources-01"]
    worker.ledger.bind("dispatch:sources-01", {"kind": "dispatch", "rows": [r["id"] for r in rows]})
    with pytest.raises(RuntimeError, match="never regenerate"):
        worker.run_batch("sources-01", rows)


def test_time_budget_stops_later_batches(tmp_path, tiny_backend):
    plan = small_plan(blocks=(1,))
    worker = make_worker(tmp_path, tiny_backend, plan)
    worker.stage0()
    worker.step_seconds, worker.prefill_per_token = 10_000.0, 0.0  # any batch now exceeds the deadline
    summary = worker._execute()
    assert summary["result"] == "partial_time_budget"
    assert summary["rows_missing"]["not_run_time_budget"] == len(plan["qwen_rows"])
