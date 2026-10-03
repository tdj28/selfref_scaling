"""Append-only GPU worker for the frozen Qwen plan. No provider calls.

Order: load and verify the pinned model, stage 0 (token bindings, neutral
coherence, repeat determinism, timing probe), then the frozen batches in
phase-priority order, then Q4 state captures. Before every batch the worst-case
time is projected from measured step times; work that cannot finish before the
deadline (minus the retrieval reserve) is recorded as not run, never rushed or
silently resampled. A dispatched batch without complete outputs blocks the run
(no regeneration). An out-of-memory batch that produced no tokens is retried
once as two deterministic halves; this fallback is part of the frozen plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
import time

from .budget import EventLedger
from .common import canonical, digest, seed, sha
from . import design

PROBE_FILLER = "The quick brown fox jumps over the lazy dog. " * 40
PROBE_PROMPT = "Here is a neutral passage: " + PROBE_FILLER + "Summarize it in one sentence."


def utc():
    return datetime.now(timezone.utc).isoformat()


def publish(path, value):
    """Fsync then atomic no-clobber publication of canonical JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical(value) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to replace immutable artifact: " + path.name)
        return
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.exists():
        raise RuntimeError("Unresolved pending publication: " + pending.name)
    with pending.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(pending, path)
    pending.unlink()


class Worker:
    def __init__(self, plan, plan_path, freeze, out, deadline_utc, cache, hardware, *,
                 factory=None, clock=time.time, allow_test=False, heartbeat_seconds=30):
        self.plan, self.plan_hash, self.freeze = plan, sha(plan_path), freeze
        self.out, self.cache, self.hardware = Path(out), cache, hardware
        parsed = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("Deadline must be an aware UTC timestamp")
        self.deadline, self.clock, self.allow_test = parsed.timestamp(), clock, allow_test
        self.reserve = plan["budget"]["worker_margin_seconds"]  # controller already reserved retrieval time
        if factory is None:
            from .qwen_backend import QwenBackend
            factory = lambda: QwenBackend(plan, cache, hardware)  # noqa: E731
        self.factory, self.backend = factory, None
        self.rows = [r for r in plan["qwen_rows"] if r["family"] != "q4"]
        self.captures = [r for r in plan["qwen_rows"] if r["family"] == "q4"]
        self.out.mkdir(parents=True, exist_ok=True)
        publish(self.out / "PLAN.json", json.loads(Path(plan_path).read_text()))
        ids = ["stage0"] + [r["id"] for r in plan["qwen_rows"]]
        self.ledger = EventLedger(self.out / "receipts.jsonl", self.plan_hash, freeze, ids)
        self.ledger.bind("runtime", {"kind": "runtime", "deadline_utc": deadline_utc, "hardware": hardware,
                                     "test_only_allowed": allow_test, "study": plan["study"],
                                     "env": {k: os.environ.get(k) for k in ("PYTORCH_CUDA_ALLOC_CONF",
                                                                            "HF_XET_HIGH_PERFORMANCE")}})
        self.stage, self.step_seconds, self.prefill_per_token = "init", None, None
        self._beat, self._stop = heartbeat_seconds, threading.Event()

    # --------------------------------------------------------------- utilities
    def events(self):
        return {e["id"]: e["data"] for e in self.ledger.read()}

    def completed(self):
        return {e["data"]["row_id"]: e["data"]["payload"] for e in self.ledger.read()
                if e["data"].get("kind") == "row"}

    def check_time(self, seconds=0):
        if self.clock() + seconds + self.reserve >= self.deadline:
            raise TimeoutError("Deadline reserve reached")
        if (self.out / "STOP").exists():
            raise RuntimeError("Controller requested a technical stop")

    def heartbeat(self):
        while not self._stop.wait(self._beat):
            value = {"utc": utc(), "stage": self.stage, "rows": len(self.completed())}
            tmp = self.out / "heartbeat.json.tmp"
            tmp.write_text(canonical(value) + "\n")
            os.replace(tmp, self.out / "heartbeat.json")

    def model(self):
        if self.backend is None:
            self.stage = "loading"
            self.check_time(1800)
            self.backend = self.factory()
            if self.backend.test_only and not self.allow_test:
                raise ValueError("A test backend cannot produce production evidence")
            publish(self.out / "metadata.json", self.backend.metadata)
            if not self.backend.test_only:
                publish(self.out / "token-verification.json",
                        self.backend.verify_token_bindings(self.plan["token_bindings"]))
        return self.backend

    # ----------------------------------------------------------------- stage 0
    def stage0(self):
        done = self.completed()
        if "stage0" in done:
            return json.loads((self.out / done["stage0"]["path"]).read_text())
        if "dispatch:stage0" in self.events():
            raise RuntimeError("Unresolved stage-0 dispatch; never rerun")
        backend = self.model()
        self.stage = "stage0"
        self.ledger.bind("dispatch:stage0", {"kind": "dispatch", "row": "stage0"})
        spec = self.plan["stage0"]
        neutral = [{"id": f"neutral-{i:02d}", "messages": [{"role": "user", "content": q}],
                    "seed": seed("stage0", i), "cap": spec["neutral_cap"]}
                   for i, (q, _) in enumerate(self.plan["prompts"]["neutral_checks"])]
        first = backend.generate_batch(neutral, check=self.check_time)
        second = backend.generate_batch(neutral, check=self.check_time)
        correct = [any(a in r["response"].lower() for a in answers)
                   for r, (_, answers) in zip(first, self.plan["prompts"]["neutral_checks"])]
        repeat = [a["output_token_ids"] == b["output_token_ids"] for a, b in zip(first, second)]
        size = self.plan["generation"]["batch_size"]
        probe_rows = [{"id": f"probe-{i:02d}", "messages": [{"role": "user", "content": PROBE_PROMPT}],
                       "seed": seed("probe", i), "cap": 16} for i in range(size)]
        probe = backend.generate_batch(probe_rows, check=self.check_time)[0]
        self.step_seconds = probe["batch_decode_seconds"] / max(1, probe["batch_decode_steps"] - 1)
        self.prefill_per_token = probe["batch_prefill_seconds"] / (size * probe["input_tokens"])
        value = {"id": "stage0", "neutral": first, "repeat_equal": repeat, "correct": correct,
                 "coherent_count": sum(correct), "coherence_min": spec["coherence_min_correct"],
                 "determinism_pass": all(repeat),
                 "think_token_rows": sum(r["think_token_present"] for r in first),
                 "probe": {k: probe[k] for k in ("batch_size", "input_tokens", "batch_prefill_seconds",
                                                  "batch_decode_seconds", "batch_decode_steps")},
                 "step_seconds_b40": self.step_seconds, "prefill_seconds_per_token_b40": self.prefill_per_token}
        value["pass"] = value["determinism_pass"] and value["coherent_count"] >= spec["coherence_min_correct"]
        path = self.out / "stage0.json"
        publish(path, value)
        self.ledger.append_row("stage0", {"path": path.name, "sha256": sha(path)})
        return value

    # ------------------------------------------------------------- generation
    def resolve(self, row, done):
        messages = []
        for m in row["messages"]:
            if "text" in m["content"]:
                messages.append({"role": m["role"], "content": m["content"]["text"]})
                continue
            source = m["content"]["source"]
            if source not in done or "missing" in done[source]:
                return None, "blocked_missing_source"
            record = json.loads((self.out / done[source]["path"]).read_text())
            if not record["response"].strip():
                return None, "blocked_empty_source"
            messages.append({"role": m["role"], "content": record["response"]})
        return messages, None

    def mark_missing(self, row, reason):
        path = self.out / "missing" / (row["id"] + ".json")
        publish(path, {"id": row["id"], "status": reason, "plan_row": {k: v for k, v in row.items() if k != "messages"}})
        self.ledger.append_row(row["id"], {"missing": reason, "path": path.relative_to(self.out).as_posix(),
                                           "sha256": sha(path)})

    def projected_seconds(self, rows, lengths):
        if self.step_seconds is None:
            raise RuntimeError("Stage-0 timing is required before projecting")
        prefill = self.prefill_per_token * sum(lengths) * 1.5
        return prefill + max(r["cap"] for r in rows) * self.step_seconds * 1.25

    def run_batch(self, batch, rows):
        done = self.completed()
        pending = [r for r in rows if r["id"] not in done]
        if not pending:
            return True
        ready = []
        for row in pending:
            messages, reason = self.resolve(row, done)
            if reason:
                self.mark_missing(row, reason)
            else:
                ready.append({"id": row["id"], "messages": messages, "seed": row["seed"], "cap": row["cap"]})
        if not ready:
            return True
        backend = self.model()
        lengths = [len(backend.serialize(r["messages"])[1]) for r in ready]
        try:
            self.check_time(self.projected_seconds(ready, lengths))
        except TimeoutError:
            return False
        events = self.events()
        groups = [(batch, ready)]
        if "dispatch:" + batch in events:
            if "failed:" + batch not in events:
                raise RuntimeError("Unresolved dispatch for batch " + batch + "; never regenerate")
            half = (len(ready) + 1) // 2
            groups = [(batch + "-a", ready[:half]), (batch + "-b", ready[half:])]
        for name, group in groups:
            if not group:
                continue
            if "dispatch:" + name in self.events():
                if all(r["id"] in self.completed() for r in group):
                    continue
                raise RuntimeError("Unresolved dispatch for batch " + name + "; never regenerate")
            self.ledger.bind("dispatch:" + name, {"kind": "dispatch", "rows": [r["id"] for r in group],
                                                   "messages_sha256": digest([r["messages"] for r in group]),
                                                   "seeds": [r["seed"] for r in group],
                                                   "caps": [r["cap"] for r in group], "utc": utc()})
            self.stage = name
            try:
                records = backend.generate_batch(group, check=self.check_time)
            except Exception as exc:  # torch.OutOfMemoryError subclasses RuntimeError
                if type(exc).__name__ == "OutOfMemoryError" and name == batch and len(group) > 1:
                    self.ledger.bind("failed:" + name, {"kind": "dispatch_failed", "error": "OutOfMemoryError",
                                                        "tokens_returned": 0})
                    import torch
                    torch.cuda.empty_cache()
                    return self.run_batch(batch, rows)
                raise
            observed = max(r["batch_decode_seconds"] / max(1, r["batch_decode_steps"] - 1) for r in records)
            self.step_seconds = max(self.step_seconds, observed)
            plan_rows = {r["id"]: r for r in rows}
            for record in records:
                record = {**record, "batch": name,
                          "plan_row": {k: v for k, v in plan_rows[record["id"]].items() if k != "messages"}}
                path = self.out / "generations" / (record["id"] + ".json")
                publish(path, record)
                self.ledger.append_row(record["id"], {"path": path.relative_to(self.out).as_posix(),
                                                      "sha256": sha(path)})
        return True

    def capture(self, batch, rows):
        from safetensors.torch import save_file
        done = self.completed()
        pending = [r for r in rows if r["id"] not in done]
        ready = []
        for row in pending:
            source = done.get(row["from_generation"])
            if source is None or "missing" in source:
                self.mark_missing(row, "blocked_missing_generation")
                continue
            record = json.loads((self.out / source["path"]).read_text())
            ready.append({"id": row["id"], "input_token_ids": record["input_token_ids"],
                          "answer_token_ids": record["output_token_ids"]})
        if not ready:
            return True
        try:
            self.check_time(self.prefill_per_token * sum(len(r["input_token_ids"]) + 4 for r in ready) * 2)
        except TimeoutError:
            return False
        if "dispatch:" + batch in self.events():
            raise RuntimeError("Unresolved capture dispatch " + batch + "; never rerun")
        self.ledger.bind("dispatch:" + batch, {"kind": "dispatch", "rows": [r["id"] for r in ready], "utc": utc()})
        self.stage = batch
        for result in self.model().capture_batch(ready):
            path = self.out / "states" / (result["id"] + ".safetensors")
            path.parent.mkdir(parents=True, exist_ok=True)
            pending_path = path.with_suffix(".pending")
            save_file({"states": result["states"]}, str(pending_path),
                      metadata={"positions": ",".join(result["positions"]), "index_0": "embeddings",
                                "index_i": "output of decoder layer i-1"})
            os.link(pending_path, path)
            pending_path.unlink()
            sidecar = self.out / "states" / (result["id"] + ".json")
            publish(sidecar, {"id": result["id"], "positions": result["positions"],
                              "shape": list(result["states"].shape), "dtype": "bfloat16",
                              "safetensors_sha256": sha(path)})
            self.ledger.append_row(result["id"], {"path": path.relative_to(self.out).as_posix(),
                                                  "sha256": sha(path)})
        return True

    # ------------------------------------------------------------------ driver
    def execute(self):
        descriptor = os.open(self.out / ".worker.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        beat = threading.Thread(target=self.heartbeat, daemon=True)
        beat.start()
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another worker owns this run") from None
            return self._execute()
        finally:
            self._stop.set()
            os.close(descriptor)

    def _execute(self):
        if (self.out / "DONE-all.json").exists():
            return json.loads((self.out / "DONE-all.json").read_text())
        stage0 = self.stage0()
        summary = {"plan_sha256": self.plan_hash, "freeze_commit": self.freeze, "stage0_pass": stage0["pass"],
                   "phases": {}, "test_only": self.allow_test}
        if not stage0["pass"]:
            summary["result"] = "stage0_failed_no_experimental_rows"
        else:
            if self.step_seconds is None:
                self.step_seconds = stage0["step_seconds_b40"]
                self.prefill_per_token = stage0["prefill_seconds_per_token_b40"]
            stopped = False
            for phase in self.plan["phase_priority"]:
                rows = self.captures if phase == "q4" else [r for r in self.rows if r["phase"] == phase]
                for batch in dict.fromkeys(r["batch"] for r in rows):
                    members = [r for r in rows if r["batch"] == batch]
                    if stopped:
                        ok = False
                    else:
                        ok = (self.capture if phase == "q4" else self.run_batch)(batch, members)
                    if not ok:
                        stopped = True
                        for row in members:
                            if row["id"] not in self.completed():
                                self.mark_missing(row, "not_run_time_budget")
                summary["phases"][phase] = "stopped_for_time" if stopped else "complete"
            summary["result"] = "partial_time_budget" if stopped else "complete"
        done = self.completed()
        summary["rows_completed"] = sum(1 for v in done.values() if "missing" not in v)
        summary["rows_missing"] = {}
        for value in done.values():
            if "missing" in value:
                summary["rows_missing"][value["missing"]] = summary["rows_missing"].get(value["missing"], 0) + 1
        self.stage = "done"
        publish(self.out / "DONE-all.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "freeze", "out", "cache", "deadline-utc", "hardware-json"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    # Set before CUDA initialization or any Hub transfer; recorded in metadata.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    plan = design.load_plan(args.plan, args.freeze)
    hardware = json.loads(args.hardware_json)
    if hardware not in plan["hardware"]["main"]:
        raise ValueError("Hardware is not a frozen alternative")
    worker = Worker(plan, args.plan, args.freeze, args.out, args.deadline_utc, args.cache, hardware)
    try:
        print(canonical(worker.execute()), flush=True)
    except Exception as exc:
        publish(Path(args.out) / "failed.json", {"error_type": type(exc).__name__, "message": str(exc)[:500],
                                                 "rows": len(worker.completed())})
        raise
    finally:
        if worker.backend is not None:
            worker.backend.close()


if __name__ == "__main__":
    main()
