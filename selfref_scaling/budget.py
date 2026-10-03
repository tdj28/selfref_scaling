# Adapted from CONSCIOUS experiments/sae_assay_diagnostic/budget.py at commit
# fe4b831b508ec7c7c7fd9a0476f0f50fccad252e (https://github.com/tdj28/llm_selfref_pre).
# Only the study constants changed: total $235, GPU compute $100 (cheap test,
# main pod, failed startups and storage), $5 retrieval reserve. The CONSCIOUS
# Pro-review cost and per-provider API caps were removed from the arithmetic.
"""Local-only GPU reservations, row receipts and owned-pod deletion permits.

One ledger per controller or worker; API spending is accounted by the separate
API-generation and judge ledgers, so the API argument here is validated (an
unknown or invalid amount still fails closed) and recorded, never capped.
No network calls. The parent must verify the public freeze, quotes and
conservative cost inputs.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

TOTAL_CAP = Decimal("235")
COMPUTE_CAP = Decimal("100")
RETRIEVAL_RESERVE = Decimal("5")


def _fail(message):
    raise ValueError(message)


def _number(value):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        _fail("Unknown cost/time; a finite nonnegative number is required")
    if isinstance(value, bool) or not number.is_finite() or number < 0:
        _fail("Unknown or invalid cost/time")
    return number


def _utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.utcoffset() != timedelta(0):
        _fail("An aware UTC timestamp is required")
    return result


def _sha(value, width=64):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % width, value):
        _fail("Invalid SHA binding")
    return value


def _canonical(value):
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            _fail("JSON keys must be strings")
        for item in value.values():
            _canonical(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _canonical(item)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


class EventLedger:
    """Canonical append-only JSONL; transact callbacks run under an OS file lock.

    row_ids is the complete frozen inventory, not just the current shard.
    Keep returned receipts externally to detect deletion of a whole ledger.
    """
    def __init__(self, path, plan_sha256, freeze_commit, row_ids):
        self.path = Path(path).absolute()
        self.plan, self.freeze = _sha(plan_sha256), _sha(freeze_commit, 40)
        ids = list(row_ids)
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            _fail("Invalid or duplicate planned row IDs")
        self.ids, self.anchor = frozenset(ids), None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bind("binding", {"kind": "binding", "row_ids": sorted(ids)})

    @contextmanager
    def _locked(self):
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                _fail("Ledger must be a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd
        finally:
            os.close(fd)

    def _read(self, fd):
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while chunk := os.read(fd, 1048576):
            chunks.append(chunk)
        raw = b"".join(chunks)
        if raw and not raw.endswith(b"\n"):
            _fail("Truncated ledger; manual recovery required")
        rows, seen, row_ids, previous = [], set(), set(), None
        for line in raw.splitlines(keepends=True):
            row = json.loads(line, parse_constant=lambda _: _fail("Nonfinite JSON"))
            if not isinstance(row, dict) or set(row) != {
                "id", "seq", "data", "plan_sha256", "freeze_commit", "previous_sha256", "sha256"
            } or _canonical(row) + b"\n" != line:
                _fail("Noncanonical or malformed ledger row")
            body = {k: v for k, v in row.items() if k != "sha256"}
            if (row["sha256"] != hashlib.sha256(_canonical(body)).hexdigest()
                    or row["previous_sha256"] != previous or type(row["seq"]) is not int
                    or row["seq"] != len(rows) or row["plan_sha256"] != self.plan
                    or row["freeze_commit"] != self.freeze):
                _fail("Ledger hash/sequence/plan/freeze mismatch")
            identifier, data = row["id"], row["data"]
            if not isinstance(identifier, str) or not identifier or identifier in seen or not isinstance(data, dict):
                _fail("Invalid or duplicate event ID")
            if data.get("kind") == "row":
                rid = data.get("row_id")
                if not isinstance(rid, str) or rid not in self.ids or rid in row_ids or identifier != "row:" + rid:
                    _fail("Unknown or duplicate row ID")
                row_ids.add(rid)
            seen.add(identifier)
            rows.append(row)
            previous = row["sha256"]
        if rows and (rows[0]["id"] != "binding" or rows[0]["data"] != {
            "kind": "binding", "row_ids": sorted(self.ids)
        }):
            _fail("Frozen row inventory binding mismatch")
        if self.anchor and (len(rows) < self.anchor[0] or rows[self.anchor[0] - 1]["sha256"] != self.anchor[1]):
            _fail("Ledger was truncated or replaced")
        if rows:
            self.anchor = len(rows), rows[-1]["sha256"]
        return rows

    def transact(self, event_id, build, *, identical_ok=False):
        """Durable check-and-append. A failed/uncertain append never permits dispatch."""
        if not isinstance(event_id, str) or not event_id:
            _fail("Invalid event ID")
        with self._locked() as fd:
            rows = self._read(fd)
            existing = next((r for r in rows if r["id"] == event_id), None)
            if existing and not identical_ok:
                _fail("Duplicate event ID")
            data = json.loads(_canonical(build(rows)))
            if not isinstance(data, dict):
                _fail("Event payload must be an object")
            if data.get("kind") == "row":
                rid = data.get("row_id")
                if not isinstance(rid, str) or rid not in self.ids or event_id != "row:" + rid:
                    _fail("Unknown or mismatched row ID")
            if existing:
                if data != existing["data"]:
                    _fail("Frozen configuration changed")
                return existing
            row = {"id": event_id, "seq": len(rows), "data": data, "plan_sha256": self.plan,
                   "freeze_commit": self.freeze, "previous_sha256": rows[-1]["sha256"] if rows else None}
            row["sha256"] = hashlib.sha256(_canonical(row)).hexdigest()
            encoded = _canonical(row) + b"\n"
            if os.write(fd, encoded) != len(encoded):
                raise OSError("Partial ledger write; do not dispatch or retry")
            os.fsync(fd)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            if self._read(fd)[-1] != row:
                _fail("Ledger readback mismatch")
            return row

    def bind(self, identifier, data):
        return self.transact(identifier, lambda _: data, identical_ok=True)

    def append_row(self, row_id, payload):
        if row_id not in self.ids:
            _fail("Unknown row ID")
        return self.transact("row:" + row_id, lambda _: {"kind": "row", "row_id": row_id, "payload": payload})

    def read(self):
        with self._locked() as fd:
            return self._read(fd)

    def assert_complete(self):
        actual = {r["data"]["row_id"] for r in self.read() if r["data"].get("kind") == "row"}
        if actual != self.ids:
            _fail("Incomplete planned row inventory")


class BudgetGuard:
    """Rates include storage per hour. Deadline is an absolute batch-stop deadline.

    Controller compute totals cover actual spend plus external reservations,
    excluding this guard's pending batches (added here). finish_batch requires
    updated accounting plus its evidence SHA, never releases on timeout alone.
    """
    def __init__(self, ledger, quote, created_utc, deadline_utc, *, max_compute=COMPUTE_CAP,
                 retrieval_reserve=RETRIEVAL_RESERVE, clock=lambda: datetime.now(timezone.utc),
                 monotonic=time.monotonic):
        if not isinstance(quote, dict) or set(quote) != {"hourly_rate_usd", "storage_hourly_usd"}:
            _fail("Missing complete compute/storage quote")
        rate, storage = _number(quote["hourly_rate_usd"]), _number(quote["storage_hourly_usd"])
        self.cap, self.reserve = _number(max_compute), _number(retrieval_reserve)
        self.created, self.deadline = _utc(created_utc), _utc(deadline_utc)
        if rate <= 0 or self.cap > COMPUTE_CAP or self.reserve < RETRIEVAL_RESERVE or self.deadline <= self.created:
            _fail("Invalid quote, cap, reserve or frozen deadline")
        self.ledger, self.rate, self.clock, self.monotonic = ledger, rate + storage, clock, monotonic
        self.wall0, self.mono0 = _utc(clock()), _number(monotonic())
        self.last_wall, self.last_mono = self.wall0, self.mono0
        ledger.bind("budget:config", {"kind": "budget_config", "rate": str(rate), "storage": str(storage),
                    "created_utc": self.created.isoformat(), "deadline_utc": self.deadline.isoformat(),
                    "compute_cap": str(self.cap), "retrieval_reserve": str(self.reserve),
                    "total_cap": str(TOTAL_CAP), "api_accounting": "separate ledgers; recorded, not capped here"})

    def _state(self, rows, compute, api):
        compute, api = _number(compute), _number(api)
        now, mono = _utc(self.clock()), _number(self.monotonic())
        events = [r["data"] for r in rows if r["data"].get("kind") in {"reserve", "finish"}]
        if now < max([self.created, self.last_wall] + [_utc(r["utc"]) for r in events]) or mono < self.last_mono:
            _fail("Clock moved backward")
        elapsed = max(_number((now - self.created).total_seconds()),
                      _number((self.wall0 - self.created).total_seconds()) + mono - self.mono0,
                      max((_number(e["elapsed_seconds"]) for e in events), default=Decimal(0)))
        accounted = max(compute, self._cost(elapsed),
                        max((_number(e["compute_accounted_usd"]) for e in events), default=Decimal(0)))
        finished = {e["batch_id"] for e in events if e["kind"] == "finish"}
        pending = [e for e in events if e["kind"] == "reserve" and e["batch_id"] not in finished]
        self.last_wall, self.last_mono = now, mono
        return {"utc": now.isoformat(), "elapsed_seconds": str(elapsed),
                "compute_accounted_usd": str(accounted), "api_actual_reserved_usd": str(api)}, pending

    def _cost(self, seconds):
        return (seconds * self.rate / 3600).quantize(Decimal("0.000001"), rounding=ROUND_CEILING)

    def before_batch(self, batch_id, worst_case_seconds, compute_actual_reserved_usd, api_actual_reserved_usd):
        seconds = _number(worst_case_seconds)
        if not isinstance(batch_id, str) or not batch_id or seconds <= 0:
            _fail("A batch ID and positive worst-case duration are required")
        def build(rows):
            state, pending = self._state(rows, compute_actual_reserved_usd, api_actual_reserved_usd)
            duration = seconds + sum((_number(p["seconds"]) for p in pending), Decimal(0))
            projected = (_number(state["compute_accounted_usd"]) + self._cost(seconds)
                         + sum((_number(p["reserved_usd"]) for p in pending), Decimal(0)))
            effective_now = max(_utc(state["utc"]), self.created + timedelta(seconds=float(state["elapsed_seconds"])))
            if (projected + self.reserve > min(self.cap, TOTAL_CAP)
                    or effective_now + timedelta(seconds=float(duration)) > self.deadline):
                _fail("Next batch breaches cost/retrieval reserve or frozen deadline")
            return {**state, "kind": "reserve", "batch_id": batch_id, "seconds": str(seconds),
                    "reserved_usd": str(self._cost(seconds)), "projected_compute_usd": str(projected)}
        return self.ledger.transact("budget:reserve:" + batch_id, build)

    def finish_batch(self, batch_id, compute_actual_reserved_usd, api_actual_reserved_usd, receipt_sha256):
        _sha(receipt_sha256)
        def build(rows):
            state, pending = self._state(rows, compute_actual_reserved_usd, api_actual_reserved_usd)
            if batch_id not in {p["batch_id"] for p in pending}:
                _fail("Unknown or already finished batch")
            return {**state, "kind": "finish", "batch_id": batch_id, "accounting_receipt_sha256": receipt_sha256}
        return self.ledger.transact("budget:finish:" + batch_id, build)


class PodRegistry:
    """Local evidence only. Registration attests a newly created pod, not discovery."""
    def __init__(self, ledger, preexisting_ids):
        self.ledger, self.blocked = ledger, frozenset(preexisting_ids)
        if any(not isinstance(p, str) or not p for p in self.blocked):
            _fail("Invalid blocked pod IDs")
        ledger.bind("pods:config", {"kind": "pod_config", "preexisting_ids": sorted(self.blocked)})

    def register_created(self, pod_id, creation_receipt_sha256, required_artifacts):
        if not isinstance(pod_id, str) or not pod_id or pod_id in self.blocked:
            _fail("Pod is not a newly owned ID")
        paths = [str(Path(p)) for p in required_artifacts]
        if not paths or len(set(paths)) != len(paths) or any(not Path(p).is_absolute() for p in paths):
            _fail("Required artifact paths must be nonempty, unique and absolute")
        return self.ledger.transact("pod:created:" + pod_id, lambda _: {
            "kind": "pod_created", "pod_id": pod_id, "required_artifacts": sorted(paths),
            "creation_receipt_sha256": _sha(creation_receipt_sha256)})

    def authorize_delete(self, pod_id, artifact_sha256):
        def build(rows):
            owned = next((r["data"] for r in rows if r["id"] == "pod:created:" + pod_id), None)
            if pod_id in self.blocked or owned is None:
                _fail("Unknown or preexisting pod; deletion forbidden")
            if not isinstance(artifact_sha256, dict) or set(artifact_sha256) != set(owned["required_artifacts"]):
                _fail("Complete expected artifact manifest required")
            for name, expected in artifact_sha256.items():
                path, digest = Path(name), hashlib.sha256()
                if path.is_symlink() or not path.is_file():
                    _fail("Missing local regular artifact")
                with path.open("rb") as handle:
                    while chunk := handle.read(1048576):
                        digest.update(chunk)
                if digest.hexdigest() != _sha(expected):
                    _fail("Artifact SHA256 mismatch")
            return {"kind": "delete_authorized", "pod_id": pod_id, "artifacts": artifact_sha256}
        return self.ledger.transact("pod:delete:" + pod_id, build)
