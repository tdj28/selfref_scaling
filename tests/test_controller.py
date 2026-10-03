"""CPU-only controller tests. Every HTTP, SSH, rsync and paid action is faked; no network.

Ported from CONSCIOUS tests/test_sae_assay_controller.py at commit
fe4b831b508ec7c7c7fd9a0476f0f50fccad252e, plus tests for hardware alternatives,
deadline arithmetic, the cheap gate, reconciliation, deletion and monitoring.
Remote scripts run only against temporary directories and a synthetic /proc.
"""
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch
import urllib.parse

from selfref_scaling import common, controller as c, design
from selfref_scaling.budget import EventLedger

FREEZE = "a" * 40
UTC = datetime(2026, 10, 3, tzinfo=timezone.utc)
PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest test-only"
H100, H200, B200 = "NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA B200"
PLAN = design.build_plan()
REQUIRE_IGNORED = c.require_ignored


def item(gpu, price, memory, availability="HIGH"):
    return {"id": gpu, "memory": memory, "secure": True, "price": {"secure": price}, "availability": availability}


class FakeAPI:
    """In-memory RunPod v2 double; records every call. POST creates one pod."""
    writable = True

    def __init__(self):
        self.calls, self.pod = [], None
        self.lost_response, self.deleted, self.hidden = False, False, False
        self.visible_after_delete, self.post_id = 0, "newowned1"
        self.preexisting = [{"id": c.BLOCKED, "name": "otherroom-sam2-pro6000"},
                            {"id": "foreign2", "name": "someone-else"}]
        self.catalog = {H100: item(H100, 3.49, 80, "LOW"), H200: item(H200, 4.59, 141),
                        B200: item(B200, 6.79, 180)}

    def inventory(self):
        self.calls.append(("GET", "/pods", None))
        visible = self.pod and not self.deleted and not self.hidden
        return list(self.preexisting) + ([self.pod] if visible else [])

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path.startswith("/catalog/gpus/"):
            return 200, copy.deepcopy(self.catalog[urllib.parse.unquote(path.split("/")[3].split("?")[0])])
        if method == "POST":
            price = Decimal(str(self.catalog[body["gpu"]["id"]]["price"]["secure"]))
            self.pod = {**copy.deepcopy(body), "id": self.post_id, "createdAt": UTC.isoformat(),
                        "cost": float(price * body["gpu"]["count"]), "status": "RUNNING",
                        "ssh": {"direct": {"host": "192.0.2.1", "port": 12345, "username": "root"}}}
            if self.lost_response:
                raise RuntimeError("simulated uncertain POST")
            return 201, self.pod
        if method == "DELETE":
            self.deleted = True
            return 204, None
        if self.deleted:
            if self.visible_after_delete:
                self.visible_after_delete -= 1
                return 200, self.pod
            raise c.ApiError(404)
        return 200, self.pod

    def count(self, method):
        return sum(m == method for m, _, _ in self.calls)


class ControllerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.plan_path = self.root / design.PLAN_PATH
        self.plan_path.parent.mkdir(parents=True)
        self.plan = copy.deepcopy(PLAN)
        self.write_plan()
        self.key = self.root / "key"
        self.key.write_text("test private material")
        self.key.chmod(0o600)
        self.key.with_suffix(".pub").write_text(PUBLIC_KEY)
        self.verify = Mock()
        self.ignored = Mock()
        for target, value in (("selfref_scaling.common.ROOT", self.root),
                              ("selfref_scaling.design.load_plan", Mock(side_effect=lambda *a, **k: copy.deepcopy(self.plan))),
                              ("selfref_scaling.controller.KEY", self.key),
                              ("selfref_scaling.controller.shutil.disk_usage", Mock(return_value=Mock(free=20 * 1024 ** 3))),
                              ("selfref_scaling.controller.verify_public", self.verify),
                              ("selfref_scaling.controller.require_ignored", self.ignored)):
            item_patch = patch(target, value)
            item_patch.start()
            self.addCleanup(item_patch.stop)
        env = patch.dict(os.environ, {"HF_TOKEN": "hf_dummy_test_only", "RUNPOD_API_KEY": "rp_dummy_test_only"})
        env.start()
        self.addCleanup(env.stop)
        self.api = FakeAPI()
        self.out = self.root / "out"
        self.ctrl = self.make("cheap")

    def write_plan(self):
        self.plan_path.write_text(common.canonical(self.plan) + "\n")

    def make(self, kind, out=None):
        return c.Controller(self.plan_path, FREEZE, out or self.out, kind, self.api, clock=lambda: UTC, sleep=Mock())

    def launched(self, ctrl=None):
        ctrl = ctrl or self.ctrl
        with patch.object(ctrl, "start_worker"):
            ctrl.launch()
        return ctrl

    def ready_main(self, cost="1.5", out=None, approval=True, status=404):
        out = out or self.out
        plan_hash = common.sha(self.plan_path)
        ledger = EventLedger(out / "controller" / "cheap" / "events.jsonl", plan_hash, FREEZE, [])
        ledger.bind("closed", {"pod_id": "cheapowned1", "compute_upper_bound_usd": cost,
                               "utc": UTC.isoformat(), "get_status": status})
        if approval:
            (out / "APPROVE-cheap").write_text(plan_hash + "\n")
        return self.make("main", out)

    def snapshots(self, ctrl=None, *, fail=False, corrupt=False):
        ctrl = ctrl or self.ctrl
        ctrl.ledger.bind("worker-intent", {"script_sha256": "b" * 64, "seconds": 120})
        files = {"generations/row1.json": b'{"id":"row1"}\n', "controller.log": b"progress\n"}
        actions = []

        def ssh(pod, command, **kwargs):
            self.assertEqual(pod["id"], "newowned1")
            if c.SIGNAL_SCRIPT in shlex.split(command):
                action = command.rsplit(" ", 1)[1]
                self.assertEqual(kwargs["data"], b"b" * 64 + b"\n")
                actions.append(action)
                if action == "stop":
                    files["controller-stopped.json"] = b'{"stopped":true}\n'
                return json.dumps({"verified": True, "action": action}).encode()
            self.assertIn(c.MANIFEST_SCRIPT, shlex.split(command))
            self.assertGreaterEqual(kwargs["timeout"], 300)
            return json.dumps({name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}).encode()

        def run(argv, **kwargs):
            self.assertEqual(argv[0], "rsync")
            self.assertEqual(kwargs["timeout"], 900)
            destination = Path(argv[-1])
            for name, raw in files.items():
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw + (b"corrupt" if corrupt else b""))
            return subprocess.CompletedProcess(argv, 1 if fail else 0, b"", b"")

        ctrl._ssh = Mock(side_effect=ssh)
        ctrl.run = Mock(side_effect=run)
        return actions


class TestOfflineContracts(ControllerCase):
    def test_default_cli_is_unpaid_even_without_valid_plan(self):
        with patch.object(c, "RunPodV2") as api, patch.object(c, "Controller") as ctrl, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            c.main(["--plan", "missing", "--freeze", FREEZE, "--kind", "main"])
        api.assert_not_called()
        ctrl.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["network_calls"], 0)
        self.assertEqual([o["gpu"] for o in result["alternatives_in_order"]], [H200, B200, B200])

    def test_quote_cli_requires_explicit_read_only_flag_and_only_gets(self):
        with patch.object(c, "RunPodV2") as api, patch("sys.stderr", new_callable=io.StringIO):
            for argv in (["--kind", "main", "--action", "quote"],
                         ["--kind", "main", "--action", "quote", "--read-only-quote", "--launch"],
                         ["--kind", "main", "--action", "status", "--read-only-quote", "--launch"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    c.main(argv)
        api.assert_not_called()
        fake = FakeAPI()
        fake.writable = False
        fake.catalog[H200]["availability"] = "NONE"
        with patch.object(c, "RunPodV2", return_value=fake) as api, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            c.main(["--kind", "main", "--action", "quote", "--read-only-quote"])
        self.assertIs(api.call_args.kwargs["writable"], False)
        self.assertEqual(len(fake.calls), 3)
        self.assertTrue(all(m == "GET" and p.startswith("/catalog/gpus/") for m, p, _ in fake.calls))
        result = json.loads(output.getvalue())
        self.assertEqual((result["would_select"]["gpu"], result["would_select"]["count"]), (B200, 6))

    def test_live_cli_requires_full_freeze(self):
        with patch.object(c, "RunPodV2") as api, patch("sys.stderr", new_callable=io.StringIO):
            for freeze in ([], ["--freeze", "abc"]):
                with self.assertRaises(SystemExit):
                    c.main(["--kind", "cheap", "--action", "status", "--launch", *freeze])
        api.assert_not_called()

    def test_schema_and_public_key_only(self):
        cheap = design.HARDWARE["cheap"][0]
        payload = c.create_payload("cheap", c.PREFIX + "cheap-012345abcdef", PUBLIC_KEY, cheap, design.HARDWARE)
        self.assertEqual(payload["gpu"], {"id": H100, "count": 2, "minCudaVersion": "12.8"})
        self.assertEqual(payload["env"], {"PUBLIC_KEY": PUBLIC_KEY})
        self.assertEqual(payload["image"], "runpod/pytorch:" + c.IMAGE_TAG + "@" + c.IMAGE_DIGEST)
        self.assertEqual(payload["mounts"], {"persistent": {"size": 40, "path": "/workspace"}})  # cheap volume
        self.assertEqual(payload["disk"], 50)
        self.assertFalse({"gpuTypeIds", "imageName"} & payload.keys())
        for key in ("private key", PUBLIC_KEY + "\nexport EVIL=1"):
            with self.assertRaises(ValueError):
                c.create_payload("cheap", payload["name"], key, cheap, design.HARDWARE)
        for name in (c.PREFIX + "main-012345abcdef", "other-cheap-012345abcdef",
                     c.PREFIX + "cheap-012345abcdeF", c.PREFIX + "cheap-012345abcde"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                c.create_payload("cheap", name, PUBLIC_KEY, cheap, design.HARDWARE)

    def test_payload_shape_for_eight_gpus(self):
        option = design.HARDWARE["main"][0]
        name = c.PREFIX + "main-0123456789ab"
        self.assertEqual(c.create_payload("main", name, PUBLIC_KEY, option, design.HARDWARE), {
            "name": name, "image": c.IMAGE, "cloud": "SECURE",
            "gpu": {"id": H200, "count": 8, "minCudaVersion": "12.8"}, "disk": 50,
            "mounts": {"persistent": {"size": 1000, "path": "/workspace"}}, "ports": ["22/tcp"],
            "startSsh": True, "startJupyter": False, "env": {"PUBLIC_KEY": PUBLIC_KEY}})
        for bad in ({**option, "count": 4}, {**option, "max_gpu_hourly_usd": "9"}, design.HARDWARE["cheap"][0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                c.create_payload("main", name, PUBLIC_KEY, bad, design.HARDWARE)

    def test_quote_fails_closed(self):
        option, quote, checked = c.select_hardware(self.api, "cheap", design.HARDWARE)
        self.assertEqual(option, design.HARDWARE["cheap"][0])
        self.assertEqual(quote, {"hourly_rate_usd": "6.98", "storage_hourly_usd": "0.20"})
        self.assertEqual(self.api.calls[0][1], "/catalog/gpus/NVIDIA%20H100%2080GB%20HBM3?include=AVAILABILITY"
                                               "&product=POD&cloud=SECURE&count=2&minCudaVersion=12.8")
        for replacement in ({"availability": "NONE"}, {"availability": None}, {"memory": 79}, {"secure": False},
                            {"price": {"secure": 3.50}}, {"price": {"secure": None}}, {"price": {}},
                            {"price": {"secure": float("nan")}}, {"price": {"secure": 0}}, {"id": "NVIDIA H100 PCIe"}):
            with self.subTest(replacement=replacement), patch.dict(self.api.catalog[H100], replacement), \
                    self.assertRaises(ValueError):
                c.select_hardware(self.api, "cheap", design.HARDWARE)
        self.assertEqual(self.api.count("POST"), 0)

    def test_first_qualifying_alternative_is_selected_in_order(self):
        option, quote, checked = c.select_hardware(self.api, "main", design.HARDWARE)
        self.assertEqual(option, design.HARDWARE["main"][0])
        self.assertEqual(quote["hourly_rate_usd"], "36.72")
        self.assertEqual([p for _, p, _ in self.api.calls],
                         ["/catalog/gpus/NVIDIA%20H200?include=AVAILABILITY&product=POD&cloud=SECURE"
                          "&count=8&minCudaVersion=12.8"])  # Stops at the first qualifying option.
        self.api.calls.clear()
        self.api.catalog[H200]["availability"] = "NONE"
        option, quote, checked = c.select_hardware(self.api, "main", design.HARDWARE)
        self.assertEqual(option, design.HARDWARE["main"][1])
        self.assertEqual(quote, {"hourly_rate_usd": "40.74", "storage_hourly_usd": "0.20"})
        self.assertEqual([x["qualifies"] for x in checked], [False, True])
        self.assertEqual([p.split("count=")[1][0] for _, p, _ in self.api.calls], ["8", "6"])
        self.api.catalog[H200].update(availability="HIGH", price={"secure": 4.60})
        self.api.catalog[B200]["price"]["secure"] = 6.80
        with self.assertRaisesRegex(ValueError, "nothing created"):
            c.select_hardware(self.api, "main", design.HARDWARE)
        self.assertEqual(self.api.count("POST"), 0)

    def test_transport_guard_route_allow_list_and_sanitized_headers(self):
        response = Mock(status=200)
        response.read.return_value = b'{}'
        opener = Mock()
        opener.open.return_value.__enter__ = Mock(return_value=response)
        opener.open.return_value.__exit__ = Mock(return_value=False)
        readonly = c.RunPodV2("secret", opener=opener)
        for method, path in (("POST", "/pods"), ("DELETE", "/pods/newowned1"),
                             ("GET", "/pods/" + c.BLOCKED), ("GET", "//evil")):
            with self.assertRaises(ValueError):
                readonly.request(method, path)
        writable = c.RunPodV2("secret", writable=True, opener=opener)
        for method, path in (("POST", "/pods/newowned1/stop"), ("POST", "/pods?x=1"), ("POST", "//pods"),
                             ("POST", "/endpoints"), ("POST", "/networkvolumes"), ("DELETE", "/pods"),
                             ("DELETE", "/pods/newowned1/x"), ("DELETE", "/pods/" + c.BLOCKED),
                             ("DELETE", "/networkvolumes/x"), ("DELETE", "/pods/a b"), ("DELETE", "/pods/../x"),
                             ("PATCH", "/pods/newowned1"), ("PUT", "/pods"), ("GET", "/pods/../x"),
                             ("GET", "pods"), ("POST", "/pods/" + c.BLOCKED)):
            with self.subTest(method=method, path=path), self.assertRaises(ValueError):
                writable.request(method, path)
        opener.open.assert_not_called()
        readonly.request("GET", "/catalog/gpus")
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(request.get_header("User-agent"), "selfref-scaling/1.0")
        self.assertNotIn("secret", request.full_url)
        writable.request("POST", "/pods", {"name": "x"})
        writable.request("DELETE", "/pods/newowned1")
        self.assertEqual([(r.args[0].get_method(), r.args[0].full_url) for r in opener.open.call_args_list[1:]],
                         [("POST", c.API + "/pods"), ("DELETE", c.API + "/pods/newowned1")])

    def test_inventory_paginates_and_rejects_repeated_cursor(self):
        api = c.RunPodV2("secret")
        api.request = Mock(side_effect=[(200, {"pods": [{"id": "a"}], "pagination": {"hasNextPage": True, "nextCursor": "next"}}),
                                       (200, {"pods": [{"id": "b"}], "pagination": {"hasNextPage": False}})])
        self.assertEqual([p["id"] for p in api.inventory()], ["a", "b"])
        self.assertIn("cursor=next", api.request.call_args.args[1])
        api.request = Mock(return_value=(200, {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": "next"}}))
        with self.assertRaises(ValueError):
            api.inventory()

    def test_strict_json(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
            with self.assertRaises(ValueError):
                c.strict_json(text)

    def test_plan_caps_and_hardware_must_match_controller(self):
        for path, value in ((("budget", "gpu_cap_usd"), "150"), (("budget", "total_cap_usd"), "300"),
                            (("budget", "cheap_gpu_cap_usd"), "6"), (("budget", "storage_retrieval_reserve_usd"), "1"),
                            (("budget", "main_retrieval_reserve_seconds"), "900"),
                            (("hardware", "cloud"), "COMMUNITY"), (("hardware", "pod_prefix"), "other-"),
                            (("hardware", "volume_gb"), "1000"), (("hardware", "storage_hourly_usd_bound"), "0"),
                            (("hardware", "cheap"), [{"gpu": H100, "count": 2, "max_gpu_hourly_usd": "3.49"}] * 2),
                            (("hardware", "main"), [{"gpu": "NVIDIA A100", "count": 8, "max_gpu_hourly_usd": "2",
                                                     "max_memory_gib": 70}]),
                            (("hardware", "main"), [{"gpu": H200, "count": 8, "max_gpu_hourly_usd": "4.59"}]),
                            (("hardware", "main"), [{"gpu": H200, "count": 8, "max_gpu_hourly_usd": "4.59",
                                                     "max_memory_gib": 141}]),
                            (("hardware", "main"), [{"gpu": H200, "count": True, "max_gpu_hourly_usd": "4.59",
                                                     "max_memory_gib": 100}])):
            with self.subTest(path=path, value=value):
                self.plan = copy.deepcopy(PLAN)
                self.plan[path[0]][path[1]] = value
                self.write_plan()
                with self.assertRaises(ValueError):
                    self.make("cheap", self.root / "other")
        self.assertEqual(self.api.calls, [])

    def test_worker_contract_exact_freeze_checkout_and_runner_command(self):
        option = design.HARDWARE["main"][0]
        prefix = ["git -c credential.helper= clone --filter=blob:none --no-checkout "
                  "https://github.com/tdj28/selfref_scaling.git /workspace/scaling/repo",
                  "cd /workspace/scaling/repo", "git fetch --depth=1 origin " + FREEZE,
                  "git checkout --detach " + FREEZE, 'test "$(git rev-parse HEAD)" = ' + FREEZE,
                  "python3 -m venv --system-site-packages /workspace/scaling/venv",
                  "/workspace/scaling/venv/bin/python -m pip install -r requirements-gpu.txt",
                  "/workspace/scaling/venv/bin/python -m pip freeze --all > /workspace/scaling/out/pip-freeze.txt",
                  ". /workspace/scaling/hf.env"]
        for kind in c.KINDS:
            script = c.worker_script(kind, design.PLAN_PATH, FREEZE, UTC.isoformat(), option)
            subprocess.run(["bash", "-n"], input=script.encode(), check=True, capture_output=True)
            lines = script.splitlines()
            self.assertEqual(lines[:3], ["set -euC", "umask 077", "mkdir -p /workspace/scaling/out"])
            self.assertTrue(lines[3].startswith("trap ") and lines[3].endswith(" EXIT"))
            self.assertEqual(lines[4:4 + len(prefix)], prefix)
            for forbidden in ("python3 -m pip install", "APPROVE", "RUNPOD_API_KEY", "HF_TOKEN", "llm_selfref_pre"):
                self.assertNotIn(forbidden, script)
        main = c.worker_script("main", design.PLAN_PATH, FREEZE, UTC.isoformat(), option).splitlines()
        self.assertEqual(main[4 + len(prefix):], [shlex.join([
            "/workspace/scaling/venv/bin/python", "-m", "selfref_scaling.pod_runner", "--plan", design.PLAN_PATH,
            "--freeze", FREEZE, "--out", "/workspace/scaling/out", "--cache", "/workspace/hf",
            "--deadline-utc", UTC.isoformat(), "--hardware-json", common.canonical(option)])])
        argv = shlex.split(main[-1])
        self.assertIn(json.loads(argv[argv.index("--hardware-json") + 1]), PLAN["hardware"]["main"])
        cheap = c.worker_script("cheap", design.PLAN_PATH, FREEZE, UTC.isoformat()).splitlines()
        self.assertEqual(cheap[4 + len(prefix):-2], [
            "rc=0", "SELFREF_REQUIRE_CUDA=1 /workspace/scaling/venv/bin/python -m pytest -q -rs tests/test_qwen_backend.py "
            "tests/test_cuda_smoke.py > /workspace/scaling/out/cheap-tests.txt || rc=$?"])
        self.assertTrue(cheap[4 + len(prefix) + 1].startswith("SELFREF_REQUIRE_CUDA=1 "))
        self.assertEqual(cheap[-1], 'exit "$rc"')
        self.assertIn("/workspace/scaling/out/DONE-all.json", cheap[-2])
        for args in (("main", "../plan.json", FREEZE, UTC.isoformat(), option),
                     ("main", "/abs/plan.json", FREEZE, UTC.isoformat(), option),
                     ("main", "-plan.json", FREEZE, UTC.isoformat(), option),
                     ("main", "plan.json\nrm -rf /", FREEZE, UTC.isoformat(), option),
                     ("main", "a b.json", FREEZE, UTC.isoformat(), option),
                     ("main", design.PLAN_PATH, "A" * 40, UTC.isoformat(), option),
                     ("main", design.PLAN_PATH, FREEZE[:39], UTC.isoformat(), option),
                     ("main", design.PLAN_PATH, FREEZE, "2026-10-03T12:00:00", option),
                     ("main", design.PLAN_PATH, FREEZE, UTC.isoformat(), None),
                     ("other", design.PLAN_PATH, FREEZE, UTC.isoformat(), option)):
            with self.subTest(args=args[:3]), self.assertRaises(ValueError):
                c.worker_script(*args)
        for script in (c.SIGNAL_SCRIPT, c.MANIFEST_SCRIPT, c.STATUS_SCRIPT):
            compile(script, "remote_script", "exec")

    def test_exit_trap_records_bootstrap_failure(self):
        with patch.object(c, "REMOTE", str(self.root / "remote")):
            script = c.worker_script("cheap", "plan.json", FREEZE, UTC.isoformat())
        setup = "\n".join(script.splitlines()[:4]) + "\nexit 3\n"
        result = subprocess.run(["bash"], input=setup.encode(), capture_output=True)
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(json.loads((self.root / "remote/out/controller-exit.json").read_text()), {"exit_code": 3})

    def test_cheap_worker_records_test_outcome_and_always_ends(self):
        for code in (1, 0):
            with self.subTest(code=code):
                remote = self.root / ("remote" + str(code))
                fake = remote / "venv" / "bin" / "python"
                fake.parent.mkdir(parents=True)
                fake.write_text('#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "pytest" ]; then echo "pytest $*"; '
                                'exit %d; fi\nexec %s "$@"\n' % (code, shlex.quote(sys.executable)))
                fake.chmod(0o755)
                with patch.object(c, "REMOTE", str(remote)):
                    lines = c.worker_script("cheap", "plan.json", FREEZE, UTC.isoformat()).splitlines()
                work = lines[lines.index(". " + str(remote) + "/hf.env") + 1:]
                result = subprocess.run(["bash"], input="\n".join(lines[:4] + work).encode(),
                                        capture_output=True, cwd=self.root)
                self.assertEqual(result.returncode, code, result.stderr)
                out = remote / "out"
                self.assertEqual(json.loads((out / "DONE-all.json").read_text()), {"pytest_exit_code": code})
                self.assertEqual(json.loads((out / "controller-exit.json").read_text()), {"exit_code": code})
                self.assertIn("-q -rs tests/test_qwen_backend.py tests/test_cuda_smoke.py",
                              (out / "cheap-tests.txt").read_text())

    def test_require_ignored_is_read_only_and_fails_closed(self):
        with patch.object(c.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            REQUIRE_IGNORED(self.root / "out/scaling-qwen-20261003/controller/main/events.jsonl")
        self.assertEqual(run.call_args.args[0], ["git", "check-ignore", "-q", "--",
                                                 "out/scaling-qwen-20261003/controller/main/events.jsonl"])
        self.assertEqual(run.call_args.kwargs["cwd"], self.root)
        with patch.object(c.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)), \
                self.assertRaises(ValueError):
            REQUIRE_IGNORED(self.root / "tracked.json")
        with patch.object(c.subprocess, "run") as run:
            REQUIRE_IGNORED(Path(tempfile.gettempdir()).resolve() / "elsewhere.json")
        run.assert_not_called()


class SignalScriptTests(unittest.TestCase):
    """Run the real remote scripts against temporary roots and a synthetic /proc.

    The exercised branches never send signals: either no worker group exists or
    the controller refuses before any os.killpg call.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name).resolve()
        self.remote, self.proc = base / "remote", base / "proc"
        (self.remote / "out").mkdir(parents=True)
        self.proc.mkdir()
        self.script = c.worker_script("cheap", design.PLAN_PATH, FREEZE, UTC.isoformat())
        self.expected = hashlib.sha256(self.script.encode()).hexdigest()

    def process(self, number, argv, pgrp):
        directory = self.proc / str(number)
        directory.mkdir()
        (directory / "cmdline").write_bytes(b"\0".join(argv) + b"\0")
        (directory / "stat").write_text(f"{number} (proc) S 1 {pgrp} {pgrp} 0 -1 0\n")

    def run_script(self, source, *args, stdin=b""):
        replaced = (source.replace("pathlib.Path('/workspace/scaling')", "pathlib.Path(%r)" % str(self.remote))
                    .replace("pathlib.Path('/workspace/scaling/out')", "pathlib.Path(%r)" % str(self.remote / "out"))
                    .replace("pathlib.Path('/proc')", "pathlib.Path(%r)" % str(self.proc)))
        self.assertNotIn("'/proc'", replaced)
        self.assertNotIn("Path('/workspace", replaced)
        return subprocess.run([sys.executable, "-c", replaced, *args], input=stdin, capture_output=True)

    def test_missing_pid_stop_without_dispatched_worker_is_recorded(self):
        self.process(4242, [b"python3", b"-c", c.SIGNAL_SCRIPT.encode(), b"stop"], 4242)
        self.process(4243, [b"bash", b"-c", b"unrelated"], 4243)
        result = self.run_script(c.SIGNAL_SCRIPT, "stop", stdin=(self.expected + "\n").encode())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"action": "stop", "verified": True, "pid": None})
        self.assertEqual(json.loads((self.remote / "out/controller-stopped.json").read_text()),
                         {"stopped": True, "pid": None, "pid_file_missing": True})

    def test_missing_pid_with_live_dispatched_worker_or_no_hash_refuses(self):
        self.process(4244, [b"timeout", b"60s", b"bash", b"-c", self.script.encode()], 4244)
        for action, stdin in (("stop", self.expected), ("pause", self.expected), ("resume", self.expected)):
            with self.subTest(action=action):
                result = self.run_script(c.SIGNAL_SCRIPT, action, stdin=stdin.encode())
                self.assertNotEqual(result.returncode, 0)
        (self.proc / "4244" / "cmdline").write_bytes(b"other\0")
        self.assertNotEqual(self.run_script(c.SIGNAL_SCRIPT, "stop", stdin=b"").returncode, 0)
        self.assertNotEqual(self.run_script(c.SIGNAL_SCRIPT, "kill", stdin=self.expected.encode()).returncode, 0)
        self.assertFalse((self.remote / "out/controller-stopped.json").exists())

    def test_existing_pid_without_live_group_stops_vacuously(self):
        pid = 999999999  # Above any kernel pid_max; never a real process.
        (self.remote / "worker.pid").write_text(f"{pid}\n")
        self.process(4245, [b"bash"], 4245)
        result = self.run_script(c.SIGNAL_SCRIPT, "stop", stdin=self.expected.encode())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["pid"], pid)
        self.assertEqual(json.loads((self.remote / "out/controller-stopped.json").read_text())["pid"], pid)

    def test_status_tolerates_transient_files_and_partial_markers(self):
        out = self.remote / "out"
        (out / "DONE-all.json").write_text("{")
        (out / "generations").mkdir()
        (out / "generations/a.json").write_text("{}")
        (out / "heartbeat.json").write_text('{"utc":"x"}')
        result = self.run_script(c.STATUS_SCRIPT)
        self.assertEqual(result.returncode, 0, result.stderr)
        status = json.loads(result.stdout)
        self.assertIsNone(status["DONE-all.json"])
        self.assertEqual(sorted(status["_progress"]), ["DONE-all.json", "generations/a.json", "heartbeat.json"])

    def test_manifest_hashes_files_and_rejects_symlinks(self):
        out = self.remote / "out"
        (out / "a.txt").write_bytes(b"abc")
        result = self.run_script(c.MANIFEST_SCRIPT)
        self.assertEqual(json.loads(result.stdout), {"a.txt": hashlib.sha256(b"abc").hexdigest()})
        (out / "link").symlink_to(out / "a.txt")
        self.assertNotEqual(self.run_script(c.MANIFEST_SCRIPT).returncode, 0)


class TestLifecycle(ControllerCase):
    def test_ssh_endpoint_and_environment(self):
        self.launched()
        args = c.ssh_args(self.api.pod, self.key, self.root / "known_hosts")
        self.assertIn("ForwardAgent=no", args)
        self.assertNotIn("SendEnv", " ".join(args))
        self.assertFalse({"HF_TOKEN", "RUNPOD_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"} & c.local_env().keys())
        for change in ({"host": "-oProxyCommand=evil"}, {"port": True}, {"username": "root;evil"}):
            with patch.dict(self.api.pod["ssh"]["direct"], change), self.assertRaises(ValueError):
                c.ssh_args(self.api.pod, self.key, self.root / "hosts")

    def test_launch_records_single_create_and_cheap_wall_cap(self):
        self.launched()
        intent = self.ctrl.event("create-intent")
        created = self.ctrl.event("created")
        self.assertLess(self.ctrl.event("pods:config")["seq"], intent["seq"])
        self.assertLess(intent["seq"], created["seq"])
        data = intent["data"]
        # Money binds before the 45-minute lifetime: ($5 - 300 s of the full rate) / rate.
        rate = Decimal("6.98") + Decimal("0.20")
        expected = UTC + timedelta(seconds=float((Decimal(5) - rate * 300 / 3600) / rate * 3600))
        self.assertEqual(c._utc(data["deadline_utc"]), expected)
        self.assertLess(expected, UTC + timedelta(seconds=c.CHEAP_LIFETIME_SECONDS - c.CHEAP_RESERVE_SECONDS))
        self.assertEqual(data["retrieval_reserve_seconds"], 300)
        self.assertEqual(data["hardware"], design.HARDWARE["cheap"][0])
        self.assertEqual(data["quote"], {"hourly_rate_usd": "6.98", "storage_hourly_usd": "0.20"})
        self.assertEqual(Decimal(data["local_cap_usd"]), Decimal("5"))
        self.assertEqual(data["payload"]["gpu"], {"id": H100, "count": 2, "minCudaVersion": "12.8"})
        self.assertTrue({c.BLOCKED, "foreign2"} <= set(data["blocked"]))
        self.assertEqual((data["plan_sha256"], data["freeze_commit"]), (common.sha(self.plan_path), FREEZE))
        self.assertEqual(self.ctrl.owned()["id"], "newowned1")
        self.ignored.assert_called_once()
        self.verify.assert_called_once_with(common.sha(self.plan_path), design.PLAN_PATH, FREEZE)
        with self.assertRaises(ValueError):
            self.ctrl.launch()
        self.assertEqual(self.api.count("POST"), 1)

    def test_bootstrap_retry_carries_prior_spending_without_new_allowance(self):
        self.plan["budget"]["prior_compute_usd"] = "0.0300157487"
        self.write_plan()
        ctrl = self.launched(self.make("cheap", self.root / "retry"))
        intent = ctrl.event("create-intent")["data"]
        self.assertEqual(Decimal(intent["prior_compute_usd"]), Decimal("0.0300157487"))
        self.assertEqual(Decimal(intent["local_cap_usd"]), Decimal("4.9699842513"))

    def test_main_deadline_arithmetic_for_eight_h200(self):
        main = self.launched(self.ready_main(cost="1.5"))
        intent = main.event("create-intent")["data"]
        self.assertEqual(intent["hardware"], design.HARDWARE["main"][0])
        self.assertEqual(intent["payload"]["gpu"], {"id": H200, "count": 8, "minCudaVersion": "12.8"})
        self.assertEqual(intent["quote"], {"hourly_rate_usd": "36.72", "storage_hourly_usd": "0.20"})
        self.assertEqual((Decimal(intent["prior_compute_usd"]), Decimal(intent["local_cap_usd"])),
                         (Decimal("1.5"), Decimal("98.5")))
        self.assertEqual(intent["retrieval_reserve_seconds"], 900)
        rate = Decimal("36.92")  # 8 x 4.59 + 0.20 storage bound.
        expected = (min(Decimal(100) - Decimal("1.5"), Decimal(100) - 5 - Decimal("1.5")) / rate) * 3600 - 900
        actual = Decimal(str((c._utc(intent["deadline_utc"]) - UTC).total_seconds()))
        self.assertLess(abs(actual - expected), Decimal("0.001"))
        self.assertAlmostEqual(float(actual), 8217.0098, places=3)
        # Deadline plus the 900 s retrieval window, the cheap spend and the $5 reserve fill exactly $100.
        self.assertLess(abs((actual + 900) * rate / 3600 + Decimal("1.5") + 5 - 100), Decimal("0.00001"))
        config = main.event("budget:config")["data"]
        self.assertEqual((config["compute_cap"], config["retrieval_reserve"], config["rate"], config["storage"]),
                         ("98.5", "5", "36.72", "0.20"))

    def test_main_requires_cheap_gate(self):
        cases = {"no_cheap_ledger": None, "not_closed": "open", "no_approval": dict(approval=False),
                 "not_404": dict(status=200), "over_cap": dict(cost="5.01"), "unknown_cost": dict(cost=None)}
        for name, case in cases.items():
            with self.subTest(case=name):
                out = self.root / name
                if case is None:
                    main = self.make("main", out)
                elif case == "open":
                    EventLedger(out / "controller/cheap/events.jsonl", common.sha(self.plan_path), FREEZE, [])
                    (out / "APPROVE-cheap").write_text(common.sha(self.plan_path))
                    main = self.make("main", out)
                else:
                    main = self.ready_main(out=out, **case)
                with self.assertRaises(ValueError):
                    main.launch()
                self.assertIsNone(main.event("create-intent"))
        for content in ("b" * 64, ""):
            with self.subTest(approval=content):
                main = self.ready_main(out=self.root / ("wrong" + content[:1]))
                (main.out / "APPROVE-cheap").write_text(content)
                with self.assertRaises(ValueError):
                    main.launch()
        main = self.ready_main(out=self.root / "symlink")
        real = self.root / "approval-elsewhere"
        real.write_text(common.sha(self.plan_path))
        (main.out / "APPROVE-cheap").unlink()
        (main.out / "APPROVE-cheap").symlink_to(real)
        with self.assertRaises(ValueError):
            main.launch()
        self.assertEqual(self.api.calls, [])
        self.verify.assert_not_called()
        self.launched(self.ready_main(out=self.root / "approved"))
        self.assertEqual(self.api.count("POST"), 1)

    def test_no_qualifying_hardware_creates_nothing_and_stays_retryable(self):
        main = self.ready_main()
        self.api.catalog[H200]["availability"] = "NONE"
        self.api.catalog[B200]["availability"] = "NONE"
        with self.assertRaisesRegex(ValueError, "nothing created"):
            main.launch()
        self.assertEqual(self.api.count("POST"), 0)
        for event in ("create-intent", "pods:config", "budget:config"):
            self.assertIsNone(main.event(event))
        self.api.catalog[B200]["availability"] = "LOW"
        self.api.preexisting.append({"id": "foreign3", "name": "appeared-later"})
        self.launched(main)
        intent = main.event("create-intent")["data"]
        self.assertEqual(intent["hardware"], design.HARDWARE["main"][1])
        self.assertEqual(intent["payload"]["gpu"]["count"], 6)
        self.assertIn("foreign3", intent["blocked"])
        self.assertEqual(self.api.count("POST"), 1)

    def test_uncertain_post_reconciles_exact_name_without_second_post(self):
        self.api.lost_response = True
        self.launched()
        self.assertEqual(self.ctrl.owned()["id"], "newowned1")
        self.assertTrue(any(r["id"].startswith("create-reconciled:") for r in self.ctrl.ledger.read()))
        self.assertEqual(self.api.count("POST"), 1)

    def test_unresolved_create_is_reconciled_by_get_only_and_never_reposted(self):
        self.api.lost_response, self.api.hidden = True, True
        with self.assertRaisesRegex(RuntimeError, "never create again"):
            self.launched()
        self.assertIsNone(self.ctrl.event("created"))
        with self.assertRaisesRegex(ValueError, "already attempted"):
            self.launched()
        self.api.hidden = False
        start = Mock()
        with patch.object(self.ctrl, "start_worker", start):
            self.assertEqual(self.ctrl.reconcile()["id"], "newowned1")
        start.assert_not_called()
        self.assertEqual(self.api.count("POST"), 1)
        self.assertEqual(self.api.count("DELETE"), 0)

    def test_post_returning_a_preexisting_pod_is_never_adopted(self):
        self.api.post_id = "foreign2"
        with self.assertRaises(ValueError):
            self.launched()
        self.assertIsNone(self.ctrl.event("created"))
        with self.assertRaises(ValueError):
            self.ctrl.owned()
        self.assertEqual(self.api.count("POST"), 1)
        self.assertEqual(self.api.count("DELETE"), 0)

    def test_unknown_pod_never_accessed(self):
        with self.assertRaises(ValueError):
            self.ctrl.get_pod()
        with self.assertRaises(ValueError):
            self.ctrl.terminate()
        self.assertEqual(self.api.calls, [])

    def test_ssh_waits_for_daemon_and_transfers_only_hf_stdin(self):
        self.launched()
        self.ctrl._ssh = Mock(side_effect=[RuntimeError("not ready"), b"", b"", b""])
        self.ctrl.start_worker()
        self.ctrl.sleep.assert_called_with(15)
        calls = self.ctrl._ssh.call_args_list
        self.assertEqual(calls[0].args[1], "true")
        transfer = calls[2]
        self.assertEqual(transfer.kwargs["data"], b"export HF_TOKEN=hf_dummy_test_only\n")
        self.assertNotIn("hf_dummy_test_only", transfer.args[1])
        self.assertIn("chmod 600", transfer.args[1])
        dispatched = shlex.split(calls[3].args[1])
        self.assertEqual(dispatched[dispatched.index("env"):dispatched.index("bash")], [
            "env", "-i", "HOME=/root",
            "PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LD_LIBRARY_PATH=/usr/local/cuda/lib64",
        ])
        self.assertEqual(dispatched[dispatched.index("timeout"):dispatched.index("timeout") + 4],
                         ["timeout", "--signal=TERM", "--kill-after=30s", "2206s"])  # money-bound cheap deadline
        script = dispatched[dispatched.index("-c") + 1]
        self.assertEqual(hashlib.sha256(script.encode()).hexdigest(),
                         self.ctrl.event("worker-intent")["data"]["script_sha256"])
        self.assertNotIn("hf_dummy_test_only", script)
        with self.assertRaises(ValueError):
            self.ctrl.start_worker()

    def test_main_dispatch_carries_deadline_and_frozen_hardware(self):
        main = self.launched(self.ready_main())
        main._ssh = Mock(side_effect=[b"", b"", b""])
        main.start_worker()
        script = shlex.split(main._ssh.call_args_list[2].args[1])
        argv = shlex.split(script[script.index("-c") + 1].splitlines()[-1])
        intent = main.event("create-intent")["data"]
        self.assertEqual(argv[argv.index("--deadline-utc") + 1], intent["deadline_utc"])
        self.assertEqual(json.loads(argv[argv.index("--hardware-json") + 1]), design.HARDWARE["main"][0])
        self.assertEqual(argv[argv.index("--plan") + 1], design.PLAN_PATH)

    def test_runtime_hardware_drift_and_cost_unknown_fail(self):
        self.launched()
        self.ctrl.cost_check(self.api.pod)
        for change in ({"cost": None}, {"cost": 0}, {"cost": 6.99}, {"image": "other"}, {"cloud": "COMMUNITY"},
                       {"gpu": {"id": H100, "count": 1, "minCudaVersion": "12.8"}},
                       {"gpu": {"id": "NVIDIA H100 PCIe", "count": 2, "minCudaVersion": "12.8"}},
                       {"gpu": None}, {"disk": 20}, {"mounts": {"persistent": {"size": 10, "path": "/workspace"}}}):
            with self.subTest(change=change), patch.dict(self.api.pod, change), self.assertRaises(ValueError):
                self.ctrl.cost_check(self.api.pod)

    def test_cost_check_stops_before_deadline_horizon(self):
        self.launched()
        clock = {"now": UTC + timedelta(minutes=26)}
        self.ctrl.clock = lambda: clock["now"]
        self.ctrl.cost_check(self.api.pod, 600)
        clock["now"] = UTC + timedelta(minutes=27)  # within 600 s of the ~36.8-minute money-bound deadline.
        with self.assertRaises(ValueError):
            self.ctrl.cost_check(self.api.pod, 600)

    def test_retrieval_pauses_resumes_and_never_approves(self):
        self.launched()
        actions = self.snapshots()
        one = self.ctrl.retrieve()
        two = self.ctrl.retrieve()
        self.assertEqual(actions, ["pause", "resume", "pause", "resume"])
        self.assertNotEqual(one["data"]["directory"], two["data"]["directory"])
        self.assertIn("--link-dest=" + one["data"]["directory"], self.ctrl.run.call_args.args[0])
        self.assertFalse(list(self.root.rglob("APPROVE*")))
        self.assertEqual(len(one["data"]["artifacts"]), 2)

    def test_retrieval_failure_always_resumes(self):
        self.launched()
        actions = self.snapshots(fail=True)
        with self.assertRaises(RuntimeError):
            self.ctrl.retrieve()
        self.assertEqual(actions, ["pause", "resume"])
        self.assertFalse(self.api.deleted)

    def test_corrupt_snapshot_blocks_delete(self):
        self.launched()
        self.snapshots(corrupt=True)
        with self.assertRaises(ValueError):
            self.ctrl.terminate()
        self.assertFalse(self.api.deleted)
        self.assertIsNone(self.ctrl.event("delete-intent"))

    def test_corruption_after_retrieval_blocks_delete(self):
        self.launched()
        self.snapshots()
        receipt = self.ctrl.retrieve(final=True)
        (Path(receipt["data"]["directory"]) / "generations/row1.json").write_text("corrupted")
        with self.assertRaises(ValueError):
            self.ctrl.terminate()
        self.assertFalse(self.api.deleted)

    def test_retrieve_before_delete_and_verified_closure(self):
        self.launched()
        actions = self.snapshots()
        closed = self.ctrl.terminate()
        self.assertEqual(actions, ["stop"])
        self.assertTrue(self.api.deleted)
        self.assertEqual(closed["data"]["get_status"], 404)
        self.assertIsNotNone(closed["data"]["compute_upper_bound_usd"])
        self.assertLess(self.ctrl.event("delete-intent")["seq"], closed["seq"])
        self.assertTrue((self.ctrl.base / "final-retrieval.json").is_file())
        self.assertEqual([(m, p) for m, p, _ in self.api.calls if m == "DELETE"], [("DELETE", "/pods/newowned1")])
        self.assertFalse(any(c.BLOCKED in path or "foreign2" in path for _, path, _ in self.api.calls))
        self.assertEqual(self.ctrl.terminate(), closed)
        self.assertEqual(self.api.count("DELETE"), 1)

    def test_deletion_requires_get_404_and_never_repeats_delete(self):
        self.launched()
        self.snapshots()
        self.api.visible_after_delete = 100
        with self.assertRaisesRegex(ValueError, "not verified"):
            self.ctrl.terminate()
        self.assertIsNone(self.ctrl.event("closed"))
        self.api.visible_after_delete = 2
        closed = self.ctrl.terminate()
        self.assertEqual(closed["data"]["get_status"], 404)
        self.ctrl.sleep.assert_called_with(5)
        self.assertEqual(self.api.count("DELETE"), 1)

    def test_no_worker_startup_failure_can_be_closed(self):
        self.launched()
        self.ctrl.terminate()
        self.assertTrue(self.api.deleted)
        receipt = c.strict_json((self.ctrl.base / "final-retrieval.json").read_bytes())
        self.assertTrue(receipt["data"]["no_worker_dispatched"])
        self.assertEqual(list(Path(receipt["data"]["directory"]).iterdir()), [])

    def test_failed_startup_after_create_is_retrieved_and_deleted(self):
        self.api.catalog[H100]["availability"] = "HIGH"
        original = self.api.request

        def exited(method, path, body=None):
            status, value = original(method, path, body)
            if method == "GET" and path == "/pods/newowned1":
                value = {**value, "status": "EXITED"}
            return status, value
        self.api.request = exited
        with self.assertRaisesRegex(ValueError, "failed startup"):
            self.ctrl.launch()
        self.assertTrue(self.api.deleted)
        self.assertEqual(self.ctrl.event("closed")["data"]["get_status"], 404)
        self.assertTrue(any(r["id"].startswith("launch-failed:") for r in self.ctrl.ledger.read()))

    def test_low_disk_blocks_create_and_pull_without_mutating_remote(self):
        with patch.object(c.shutil, "disk_usage", return_value=Mock(free=8 * 1024 ** 3 - 1)):
            with self.assertRaises(ValueError):
                self.ctrl.launch()
            with self.assertRaises(ValueError):
                self.ctrl.retrieve()
        self.assertEqual(self.api.calls, [])

    def test_lost_delete_response_can_be_confirmed_without_retry(self):
        self.launched()
        self.snapshots()
        original = self.api.request

        def lost(method, path, body=None):
            result = original(method, path, body)
            if method == "DELETE":
                raise RuntimeError("lost response")
            return result

        self.api.request = lost
        with self.assertRaises(RuntimeError):
            self.ctrl.terminate()
        self.assertTrue(self.api.deleted)
        self.assertEqual(self.ctrl.terminate()["id"], "closed")
        self.assertEqual(self.api.count("DELETE"), 1)


class TestMonitor(ControllerCase):
    """Monitor decisions with lifecycle effects replaced by one ordered Mock."""

    def setUp(self):
        super().setUp()
        self.launched()
        self.ctrl.ledger.bind("worker-intent", {"script_sha256": "b" * 64, "seconds": 120})

    def state(self, *, mtime=1, files=()):
        result = {"pod": self.api.pod, "files": {"_progress": {"heartbeat.json": [10, mtime]}}}
        result["files"].update({name: {} for name in files})
        return result

    def monitored(self, statuses, *, step=60, cost_check=None, ctrl=None):
        """Run monitor with a fake monotonic clock advancing `step` seconds per tick."""
        ctrl = ctrl or self.ctrl
        clock = {"t": 0.0}
        lifecycle = Mock()
        ctrl.status, ctrl.retrieve, ctrl.quiesce = lifecycle.status, lifecycle.retrieve, lifecycle.quiesce
        ctrl.status.side_effect = statuses
        ctrl.terminate = lifecycle.terminate
        ctrl.terminate.return_value = "closed"
        if cost_check is not None:
            ctrl.cost_check = cost_check
        ctrl.sleep = Mock(side_effect=lambda s: clock.__setitem__("t", clock["t"] + step))
        with patch.object(c.time, "monotonic", side_effect=lambda: clock["t"]):
            self.assertEqual(ctrl.monitor(), "closed")
        return lifecycle

    def test_terminal_marker_triggers_final_retrieval_and_delete(self):
        for marker in c.TERMINAL:
            with self.subTest(marker=marker):
                self.api = FakeAPI()  # A pod listed before creation can never be adopted.
                ctrl = self.launched(self.make("cheap", self.root / marker))
                lifecycle = self.monitored([self.state(), self.state(mtime=2, files=[marker])], ctrl=ctrl)
                self.assertEqual([n for n, _, _ in lifecycle.mock_calls],
                                 ["status", "retrieve", "status", "terminate"])
        self.assertFalse(list(self.root.rglob("APPROVE*")))

    def test_stalled_worker_is_stopped_before_final_retrieval(self):
        lifecycle = self.monitored([self.state()] * 3, step=901, cost_check=Mock())
        self.assertEqual([n for n, _, _ in lifecycle.mock_calls],
                         ["status", "retrieve", "status", "quiesce", "terminate"])
        stop = [r for r in self.ctrl.ledger.read() if r["id"].startswith("monitor-stop:")]
        self.assertEqual(stop[0]["data"], {"reason": "Owned worker stalled", "terminal": False})

    def test_stall_without_dispatched_worker_deletes_without_signalling(self):
        self.api = FakeAPI()
        ctrl = self.launched(self.make("cheap", self.root / "no-worker"))
        lifecycle = self.monitored([self.state()] * 3, step=901, cost_check=Mock(), ctrl=ctrl)
        self.assertEqual([n for n, _, _ in lifecycle.mock_calls], ["status", "retrieve", "status", "terminate"])

    def test_heartbeat_progress_prevents_stall_and_snapshots_every_ten_minutes(self):
        statuses = [self.state(mtime=i) for i in range(21)] + [self.state(mtime=99, files=["DONE-all.json"])]
        lifecycle = self.monitored(statuses, step=60, cost_check=Mock())
        names = [n for n, _, _ in lifecycle.mock_calls]
        self.assertEqual(names.count("retrieve"), 3)  # t = 0, 600 and 1200 s.
        self.assertNotIn("quiesce", names)
        self.assertEqual(names[-1], "terminate")

    def test_budget_or_deadline_stops_worker_before_final_retrieval(self):
        lifecycle = self.monitored([self.state()],
                                   cost_check=Mock(side_effect=ValueError("Retrieval reserve/deadline reached")))
        self.assertEqual(lifecycle.mock_calls[-2:], [call.quiesce(), call.terminate()])
        self.assertNotIn("retrieve", [n for n, _, _ in lifecycle.mock_calls])

    def test_real_cost_check_runs_with_ten_minute_horizon(self):
        with patch.object(self.ctrl, "cost_check", wraps=self.ctrl.cost_check) as check:
            self.monitored([self.state(files=["DONE-all.json"])])
        self.assertEqual(check.call_args.args[1], 600)
        rows = {r["id"] for r in self.ctrl.ledger.read()}
        reserves = [i for i in rows if i.startswith("budget:reserve:monitor-")]
        self.assertTrue(reserves and all(i.replace(":reserve:", ":finish:") in rows for i in reserves))

    def test_transient_failures_are_retried_then_close(self):
        ok = self.state()
        statuses = [RuntimeError("ssh"), subprocess.TimeoutExpired("ssh", 60), ok,
                    OSError("net"), ValueError("bad json"), c.ApiError(502)]
        lifecycle = self.monitored(statuses, cost_check=Mock())
        self.assertEqual(lifecycle.mock_calls[-2:], [call.quiesce(), call.terminate()])
        retries = [r["data"]["consecutive"] for r in self.ctrl.ledger.read() if r["id"].startswith("monitor-retry:")]
        self.assertEqual(retries, [1, 2, 1, 2, 3])
        self.assertEqual(self.ctrl.sleep.call_count, 5)

    def test_monitor_requires_launch_and_resumes_closure(self):
        self.api.writable = False
        with self.assertRaises(ValueError):
            self.ctrl.monitor()
        self.api.writable = True
        self.snapshots()
        self.ctrl.terminate()
        self.ctrl.status = Mock()
        self.assertEqual(self.ctrl.monitor()["id"], "closed")
        self.ctrl.status.assert_not_called()

if __name__ == "__main__":
    unittest.main()
