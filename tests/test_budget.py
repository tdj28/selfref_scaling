"""Offline fault, accounting and ownership tests. No providers or model imports.

Ported from CONSCIOUS tests/test_sae_assay_budget.py at commit
fe4b831b508ec7c7c7fd9a0476f0f50fccad252e with this study's caps ($100 GPU,
$5 reserve, $235 total; API spending is recorded but capped elsewhere).
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from selfref_scaling import budget as b

PLAN, FREEZE, RECEIPT = "a" * 64, "b" * 40, "c" * 64
START = datetime(2026, 10, 3, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.wall, self.ticks = START, 1000

    def advance(self, seconds):
        self.wall += timedelta(seconds=seconds)
        self.ticks += seconds


class LocalCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.clock = Clock()

    def ledger(self, path="events.jsonl", ids=("r1", "r2")):
        return b.EventLedger(self.base / path, PLAN, FREEZE, ids)

    def guard(self, ledger=None, rate="100", storage="0", **kwargs):
        return b.BudgetGuard(ledger or self.ledger(),
                             {"hourly_rate_usd": rate, "storage_hourly_usd": storage},
                             START, START + timedelta(hours=24),
                             clock=lambda: self.clock.wall, monotonic=lambda: self.clock.ticks,
                             **kwargs)


class TestLedger(LocalCase):
    def test_binding_exact_rows_replay_and_completion(self):
        ledger = self.ledger()
        with self.assertRaises(ValueError):
            ledger.assert_complete()
        receipt = ledger.append_row("r1", {"result": [1, True, None]})
        self.assertEqual(receipt["plan_sha256"], PLAN)
        self.assertEqual(receipt["freeze_commit"], FREEZE)
        self.assertEqual(receipt["seq"], 1)
        self.assertEqual(self.ledger().read(), ledger.read())
        with self.assertRaises(ValueError):
            self.ledger().append_row("r1", {})
        ledger.append_row("r2", {})
        ledger.assert_complete()
        self.assertEqual(len(ledger.read()), 3)

    def test_fsync_and_readback_before_acknowledgement(self):
        ledger = self.ledger()
        with patch.object(b.os, "fsync", wraps=os.fsync) as sync:
            row = ledger.append_row("r1", {"n": 1})
        self.assertEqual(sync.call_count, 2)  # File plus directory.
        self.assertEqual(row, self.ledger().read()[-1])

    def test_unknown_duplicate_and_nonfinite_payloads_leave_file_unchanged(self):
        ledger = self.ledger()
        original = ledger.path.read_bytes()
        invalid = [float("nan"), float("inf"), {"a": [float("-inf")]}, {1: "bad"}]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises((ValueError, TypeError)):
                ledger.append_row("r1", payload)
            self.assertEqual(original, ledger.path.read_bytes())
        with self.assertRaises(ValueError):
            ledger.append_row("unknown", {})
        with self.assertRaises(ValueError):
            ledger.transact("row:unknown", lambda _: {"kind": "row", "row_id": "unknown"})
        self.assertEqual(original, ledger.path.read_bytes())

    def test_changed_plan_freeze_inventory_and_duplicate_inventory_rejected(self):
        ledger = self.ledger()
        for plan, freeze, ids in [("d" * 64, FREEZE, ["r1", "r2"]),
                                   (PLAN, "d" * 40, ["r1", "r2"]),
                                   (PLAN, FREEZE, ["r1"]), (PLAN, FREEZE, ["r1", "r1"])]:
            with self.subTest(plan=plan, freeze=freeze, ids=ids), self.assertRaises(ValueError):
                b.EventLedger(ledger.path, plan, freeze, ids)

    def test_partial_write_is_retained_and_resume_refuses_it(self):
        ledger = self.ledger()
        write = os.write

        def partial(fd, data):
            return write(fd, data[:len(data) // 2])
        with patch.object(b.os, "write", side_effect=partial), self.assertRaises(OSError):
            ledger.append_row("r1", {"text": "not a complete receipt"})
        damaged = ledger.path.read_bytes()
        self.assertFalse(damaged.endswith(b"\n"))
        with self.assertRaisesRegex(ValueError, "Truncated"):
            self.ledger()
        self.assertEqual(ledger.path.read_bytes(), damaged)

    def test_failed_fsync_never_acknowledges_or_retries_complete_row(self):
        ledger = self.ledger()
        with patch.object(b.os, "fsync", side_effect=OSError("disk failure")), self.assertRaises(OSError):
            ledger.append_row("r1", {})
        resumed = self.ledger()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            resumed.append_row("r1", {})

    def test_tampering_and_noncanonical_json_rejected(self):
        ledger = self.ledger()
        receipt = ledger.append_row("r1", {"n": 1})
        original = ledger.path.read_bytes()
        bad = original.replace(b'"n":1', b'"n":2')
        ledger.path.write_bytes(bad)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.ledger()
        prefix = original.splitlines(keepends=True)[0]
        variants = [json.dumps(receipt).encode() + b"\n", b'{"n":NaN}\n',
                    b'{"n":1e999}\n', b'{"id":"a","id":"b"}\n', b"\n"]
        for raw in variants:
            with self.subTest(raw=raw):
                ledger.path.write_bytes(prefix + raw)
                with self.assertRaises((ValueError, TypeError)):
                    self.ledger()

    def test_in_process_tail_removal_is_not_silently_repaired(self):
        ledger = self.ledger()
        ledger.append_row("r1", {})
        ledger.path.write_bytes(ledger.path.read_bytes().splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, "truncated or replaced"):
            ledger.append_row("r2", {})

    def test_symlink_ledger_cannot_overwrite_another_file(self):
        other = self.base / "other"
        other.write_bytes(b"private content")
        (self.base / "events.jsonl").symlink_to(other)
        with self.assertRaises(OSError):
            self.ledger()
        self.assertEqual(other.read_bytes(), b"private content")

    def test_two_instances_do_not_fork_hash_chain(self):
        left, right = self.ledger(), self.ledger()
        barrier = threading.Barrier(2)

        def append(pair):
            ledger, rid = pair
            barrier.wait()
            return ledger.append_row(rid, {})
        with ThreadPoolExecutor(2) as pool:
            receipts = list(pool.map(append, [(left, "r1"), (right, "r2")]))
        self.assertEqual({r["seq"] for r in receipts}, {1, 2})
        self.ledger().assert_complete()

    def test_nonfinite_canonical_record_and_duplicate_event_replay_fail(self):
        ledger = self.ledger()
        ledger.append_row("r1", {"n": 1})
        original = ledger.path.read_bytes()
        for replacement in (b"NaN", b"Infinity", b"1e999"):
            ledger.path.write_bytes(original.replace(b'"n":1', b'"n":' + replacement))
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.ledger()
        last = json.loads(original.splitlines()[-1])
        last["seq"], last["previous_sha256"] = 2, last["sha256"]
        last["sha256"] = hashlib.sha256(b._canonical({k: v for k, v in last.items() if k != "sha256"})).hexdigest()
        ledger.path.write_bytes(original + b._canonical(last) + b"\n")
        with self.assertRaisesRegex(ValueError, "duplicate event"):
            self.ledger()


class TestBudget(LocalCase):
    def test_policy_caps_and_retrieval_reserve(self):
        self.assertEqual(b.TOTAL_CAP, Decimal("235"))
        self.assertEqual(b.COMPUTE_CAP, Decimal("100"))
        self.assertEqual(b.RETRIEVAL_RESERVE, Decimal("5"))
        self.assertFalse(hasattr(b, "PRO_COST") or hasattr(b, "API_CAPS"))
        guard = self.guard()
        row = guard.before_batch("one", 900, 70, 0)
        self.assertEqual(Decimal(row["data"]["projected_compute_usd"]), Decimal("95"))
        with self.assertRaisesRegex(ValueError, "reserve"):
            guard.before_batch("two", 1, 70, 0)
        self.assertEqual(len([r for r in guard.ledger.read() if r["data"].get("kind") == "reserve"]), 1)
        config = guard.ledger.read()[1]["data"]
        self.assertEqual((config["compute_cap"], config["retrieval_reserve"], config["total_cap"]), ("100", "5", "235"))

    def test_individually_rounded_reservations_cannot_be_rounded_down_together(self):
        guard = self.guard(rate="0.000001")
        row = guard.before_batch("first", 1, "94.999999", 0)
        self.assertEqual(row["data"]["reserved_usd"], "0.000001")
        with self.assertRaises(ValueError):
            guard.before_batch("second", 1, "94.999999", 0)

    def test_positive_quote_storage_and_numeric_validation(self):
        for quote in (None, {}, {"hourly_rate_usd": 1},
                      {"hourly_rate_usd": 0, "storage_hourly_usd": 0},
                      {"hourly_rate_usd": -1, "storage_hourly_usd": 0},
                      {"hourly_rate_usd": 1, "storage_hourly_usd": None}):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                b.BudgetGuard(self.ledger(), quote, START, START + timedelta(hours=1))
        guard = self.guard(rate="99", storage="1")
        self.assertEqual(guard.before_batch("b", 360, 0, 0)["data"]["reserved_usd"], "10.000000")

    def test_unknown_costs_and_duration_fail_closed_without_reservation(self):
        guard = self.guard()
        initial = guard.ledger.path.read_bytes()
        for bad in (None, "unknown", float("nan"), "Infinity", -1, True):
            for args in ((bad, 0, 0), (1, bad, 0), (1, 0, bad)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    guard.before_batch("b", *args)
        self.assertEqual(guard.ledger.path.read_bytes(), initial)

    def test_caps_cannot_be_relaxed_and_compute_overrun_blocks(self):
        for kwargs in ({"max_compute": "100.000001"}, {"retrieval_reserve": "4.99"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.guard(**kwargs)
        guard = self.guard()
        with self.assertRaises(ValueError):
            guard.before_batch("b", 1, 95, 0)

    def test_api_spending_is_recorded_not_capped_here(self):
        guard = self.guard()
        row = guard.before_batch("b", 1, 0, "150")
        self.assertEqual(row["data"]["api_actual_reserved_usd"], "150")
        self.assertEqual(Decimal(row["data"]["projected_compute_usd"]), Decimal("0.027778"))

    def test_wall_cost_storage_and_external_spend_are_conservative(self):
        guard = self.guard(rate="45", storage="5")
        self.clock.advance(3600)
        with self.assertRaises(ValueError):
            guard.before_batch("b", 3600, 0, 0)
        row = guard.before_batch("short", 1, 60, 7)
        self.assertEqual(row["data"]["compute_accounted_usd"], "60")

    def test_backward_wall_or_monotonic_clock_blocks(self):
        for field in ("wall", "ticks"):
            with self.subTest(field=field):
                guard = self.guard(ledger=self.ledger(path=field + ".jsonl"))
                if field == "wall":
                    self.clock.wall -= timedelta(seconds=1)
                else:
                    self.clock.ticks -= 1
                with self.assertRaisesRegex(ValueError, "Clock"):
                    guard.before_batch("b", 1, 0, 0)
                self.clock = Clock()

    def test_forward_jump_and_frozen_deadline_fail_closed(self):
        guard = self.guard(rate="1")
        self.clock.advance(24 * 3600)
        with self.assertRaisesRegex(ValueError, "deadline"):
            guard.before_batch("b", 1, 0, 0)
        self.clock = Clock()
        self.clock.ticks += 25 * 3600
        # A fresh guard cannot infer earlier monotonic time; use the existing guard.
        self.clock.wall = START + timedelta(hours=24)
        with self.assertRaises(ValueError):
            guard.before_batch("c", 1, 0, 0)

    def test_monotonic_elapsed_charges_when_wall_clock_stalls(self):
        guard = self.guard()
        self.clock.ticks += 7200
        with self.assertRaises(ValueError):
            guard.before_batch("b", 1, 0, 0)

    def test_resume_rejects_wall_time_before_last_durable_observation(self):
        guard = self.guard()
        self.clock.advance(60)
        guard.before_batch("b", 1, 0, 0)
        self.clock.wall -= timedelta(seconds=30)
        with self.assertRaisesRegex(ValueError, "Clock"):
            self.guard().before_batch("c", 1, 0, 0)

    def test_timestamp_unknown_future_creation_and_naive_times(self):
        ledger = self.ledger()
        for created, deadline in ((START.replace(tzinfo=None), START + timedelta(hours=1)),
                                  (START, START), (None, START + timedelta(hours=1))):
            with self.subTest(created=created), self.assertRaises(ValueError):
                b.BudgetGuard(ledger, {"hourly_rate_usd": 1, "storage_hourly_usd": 0}, created, deadline)
        self.clock.wall -= timedelta(seconds=1)
        guard = self.guard()
        with self.assertRaises(ValueError):
            guard.before_batch("b", 1, 0, 0)

    def test_resume_keeps_reservations_and_absolute_creation_time(self):
        first = self.guard()
        first.before_batch("b", 900, 70, 0)
        resumed = self.guard()
        with self.assertRaises(ValueError):
            resumed.before_batch("c", 1, 70, 0)
        with self.assertRaises(ValueError):
            resumed.before_batch("b", 900, 70, 0)
        with self.assertRaisesRegex(ValueError, "configuration"):
            b.BudgetGuard(self.ledger(), {"hourly_rate_usd": 100, "storage_hourly_usd": 0},
                          START, START + timedelta(hours=25))

    def test_finish_requires_accounting_receipt_and_releases_only_its_reservation(self):
        guard = self.guard()
        guard.before_batch("b", 900, 60, 0)
        with self.assertRaises(ValueError):
            guard.finish_batch("b", 60, 0, None)
        with self.assertRaises(ValueError):
            guard.finish_batch("unknown", 60, 0, RECEIPT)
        guard.finish_batch("b", 61, 0, RECEIPT)
        with self.assertRaises(ValueError):
            guard.finish_batch("b", 61, 0, RECEIPT)
        row = guard.before_batch("c", 900, 61, 0)
        self.assertEqual(row["data"]["projected_compute_usd"], "86.000000")

    def test_atomic_check_and_reserve_across_controller_instances(self):
        guards = [self.guard(), self.guard()]
        barrier = threading.Barrier(2)

        def reserve(pair):
            i, guard = pair
            barrier.wait()
            try:
                guard.before_batch(str(i), 900, 70, 0)
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(2) as pool:
            self.assertEqual(sorted(pool.map(reserve, enumerate(guards))), [False, True])
        rows = self.ledger().read()
        self.assertEqual(sum(r["data"].get("kind") == "reserve" for r in rows), 1)

    def test_no_budget_reservation_ack_on_partial_write(self):
        guard = self.guard()
        write = os.write
        with patch.object(b.os, "write", side_effect=lambda fd, data: write(fd, data[:10])):
            with self.assertRaises(OSError):
                guard.before_batch("b", 1, 0, 0)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            self.guard()

    def test_accounting_cannot_erase_previously_recorded_spend(self):
        guard = self.guard()
        guard.before_batch("b", 1, 94, 0)
        guard.finish_batch("b", 94, 0, RECEIPT)
        with self.assertRaises(ValueError):
            self.guard().before_batch("c", 100, 0, 0)

    def test_cross_process_reservations_share_one_cap(self):
        self.guard()
        root = str(Path(__file__).resolve().parents[1])
        script = """
import sys
from datetime import datetime, timedelta, timezone
sys.path.insert(0, sys.argv[1])
from selfref_scaling.budget import EventLedger, BudgetGuard
start = datetime(2026, 10, 3, tzinfo=timezone.utc)
ledger = EventLedger(sys.argv[2], 'a'*64, 'b'*40, ['r1', 'r2'])
guard = BudgetGuard(ledger, {'hourly_rate_usd': '100', 'storage_hourly_usd': '0'},
                    start, start + timedelta(hours=24), clock=lambda: start, monotonic=lambda: 1000)
try:
    guard.before_batch(sys.argv[3], 900, 70, 0)
except ValueError:
    print('blocked')
else:
    print('reserved')
"""
        processes = [subprocess.Popen([sys.executable, "-B", "-I", "-S", "-c", script, root,
                                       str(self.base / "events.jsonl"), str(i)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for i in range(2)]
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stderr)
            results.append(stdout.strip())
        self.assertEqual(sorted(results), ["blocked", "reserved"])


class TestOwnership(LocalCase):
    def registry(self):
        return b.PodRegistry(self.ledger(), ["preexisting"])

    def artifacts(self):
        result = {}
        for name in ("raw.jsonl", "manifest.json", "runtime.log"):
            path = self.base / name
            path.write_bytes((name + " verified content").encode())
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def test_unknown_and_preexisting_pods_never_authorized(self):
        registry = self.registry()
        with self.assertRaises(ValueError):
            registry.register_created("preexisting", RECEIPT, [str(self.base / "raw")])
        for pod in ("unknown", "preexisting"):
            with self.subTest(pod=pod), self.assertRaises(ValueError):
                registry.authorize_delete(pod, {})

    def test_verified_artifacts_receipt_and_persistent_ownership(self):
        files = self.artifacts()
        registry = self.registry()
        registry.register_created("owned", RECEIPT, files)
        resumed = self.registry()
        receipt = resumed.authorize_delete("owned", files)
        self.assertEqual(receipt["data"]["artifacts"], files)
        self.assertEqual(receipt["data"]["kind"], "delete_authorized")
        self.assertEqual(self.ledger().read()[-1], receipt)
        self.assertTrue(all(Path(path).exists() for path in files))
        with self.assertRaises(ValueError):
            resumed.authorize_delete("owned", files)

    def test_required_manifest_cannot_omit_artifacts(self):
        registry, files = self.registry(), self.artifacts()
        registry.register_created("owned", RECEIPT, files)
        for manifest in ({}, dict(list(files.items())[:1]), {**files, "/unknown": RECEIPT}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                registry.authorize_delete("owned", manifest)

    def test_missing_corrupt_or_symlink_artifact_blocks_delete(self):
        for damage in ("missing", "corrupt", "symlink"):
            with self.subTest(damage=damage):
                files = self.artifacts()
                registry = self.registry()
                registry.register_created(damage, RECEIPT, files)
                path = Path(next(iter(files)))
                if damage == "missing":
                    path.unlink()
                elif damage == "corrupt":
                    path.write_bytes(b"changed")
                else:
                    other = self.base / "target"
                    other.write_bytes(path.read_bytes())
                    path.unlink()
                    path.symlink_to(other)
                with self.assertRaises(ValueError):
                    registry.authorize_delete(damage, files)
                if path.is_symlink():
                    path.unlink()

    def test_blocked_inventory_is_frozen_and_creation_is_not_idempotent(self):
        registry, files = self.registry(), self.artifacts()
        registry.register_created("owned", RECEIPT, files)
        with self.assertRaises(ValueError):
            registry.register_created("owned", RECEIPT, files)
        with self.assertRaisesRegex(ValueError, "configuration"):
            b.PodRegistry(self.ledger(), [])


if __name__ == "__main__":
    unittest.main()
