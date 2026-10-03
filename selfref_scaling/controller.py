# Adapted from CONSCIOUS experiments/sae_assay_diagnostic/controller.py at commit
# fe4b831b508ec7c7c7fd9a0476f0f50fccad252e (https://github.com/tdj28/llm_selfref_pre).
# Study changes: this repository, pod prefix, remote root and budget; hardware is
# the first qualifying frozen alternative (read-only quotes); the main worker is
# selfref_scaling.pod_runner and the cheap worker runs the CUDA tests; no
# approval barriers inside a run. Lifecycle safety is unchanged: one create
# attempt with exact-name reconciliation, owned-pod-only access, verified
# retrieval before deletion and a direct GET 404 before closure.
"""Explicit-launch RunPod v2 controller for the Qwen3.5-397B study.

Import and the default CLI are offline and unpaid. Only ``--launch`` creates,
signals or deletes, and only a pod this ledger created (unique
``scaling-qwen-20261003-<kind>-<12 hex>`` name, absent from the inventory at
creation, never the protected foreign pod). Never approves anything: the main
pod requires the cheap ledger's verified deletion plus a human-written
``APPROVE-cheap`` containing the plan hash. Keep the monitor supervised on the
local host: the remote timeout stops work at the deadline but cannot stop
billing; only deletion does. API schema: https://api.runpod.io/v2/openapi.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from . import common, design
from .budget import (COMPUTE_CAP, RETRIEVAL_RESERVE, TOTAL_CAP, BudgetGuard, EventLedger, PodRegistry,
                     _canonical, _number, _utc)

API = "https://api.runpod.io/v2"
REPO = "https://github.com/tdj28/selfref_scaling.git"
RAW = "https://raw.githubusercontent.com/tdj28/selfref_scaling/"
IMAGE_TAG = "1.0.2-cu1281-torch280-ubuntu2404"
IMAGE_DIGEST = "sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"
IMAGE = "runpod/pytorch:" + IMAGE_TAG + "@" + IMAGE_DIGEST
PREFIX = common.POD_PREFIX
BLOCKED = "6lnzutgdc6lemh"  # Protected foreign pod; also every pod listed at creation time.
KEY = Path("~/.ssh/runpod_conscious_20260708")
REMOTE = "/workspace/scaling"
HF_CACHE = "/workspace/hf"
OUT_ROOT = common.ROOT / "out" / "scaling-qwen-20261003"
USER_AGENT = "selfref-scaling/1.0"
KINDS = ("cheap", "main")
# Minimum catalog memory per GPU type; a frozen alternative with less does not qualify.
MIN_MEMORY_GIB = {"NVIDIA H200": 141, "NVIDIA B200": 180, "NVIDIA H100 80GB HBM3": 80}
AVAILABLE = frozenset({"LOW", "MEDIUM", "HIGH"})
CHEAP_CAP = Decimal("5")
# Cheap lifetime: worker deadline plus a short retrieval/deletion window <= 45 min;
# the $5 cap binds first (about 37 minutes at the 2xH100 quote).
CHEAP_LIFETIME_SECONDS, CHEAP_RESERVE_SECONDS = 2700, 300
MONITOR_TICK_SECONDS, MONITOR_HORIZON_SECONDS, SNAPSHOT_SECONDS = 60, 600, 600
STALL_SECONDS, MONITOR_FAILURE_LIMIT = 900, 3  # The runner rewrites out/heartbeat.json every ~30 s.
RSYNC_TIMEOUT, MANIFEST_TIMEOUT, SIGNAL_TIMEOUT = 900, 300, 150  # About 1 GB of outputs.
CHEAP_TESTS = ("tests/test_qwen_backend.py", "tests/test_cuda_smoke.py")
TERMINAL = ("DONE-all.json", "controller-exit.json", "controller-stopped.json")
sha, strict_json = common.sha, common.strict_json


def now():
    return datetime.now(timezone.utc)


class ApiError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"RunPod HTTP {status}; no automatic mutation retry")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class RunPodV2:
    def __init__(self, token, *, writable=False, opener=None):
        if not isinstance(token, str) or not token.strip() or any(c.isspace() for c in token):
            raise ValueError("RUNPOD_API_KEY must be supplied through the environment")
        self.token, self.writable = token, writable
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, method, path, body=None):
        if (not path.startswith("/") or ".." in path or "//" in path or BLOCKED in path
                or method not in {"GET", "POST", "DELETE"}):
            raise ValueError("Forbidden API route")
        if method != "GET" and (not self.writable or not (
            (method == "POST" and path == "/pods") or
            (method == "DELETE" and re.fullmatch(r"/pods/[A-Za-z0-9_-]+", path))
        )):
            raise ValueError("Mutation disabled or outside owned lifecycle")
        request = urllib.request.Request(API + path, method=method,
            data=None if body is None else _canonical(body), headers={
                "Authorization": "Bearer " + self.token, "User-Agent": USER_AGENT,
                "Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=30) as response:
                raw = response.read()
                return response.status, strict_json(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raise ApiError(exc.code) from None  # Never serialize response bodies/credentials.
        except urllib.error.URLError:
            raise RuntimeError("RunPod transport outcome unknown; do not retry mutation") from None

    def inventory(self):
        pods, cursor, seen = [], None, set()
        while True:
            query = {"includeClusterPods": "true", "limit": 1000}
            if cursor:
                query["cursor"] = cursor
            _, page = self.request("GET", "/pods?" + urllib.parse.urlencode(query))
            pods.extend(page["pods"])
            pagination = page["pagination"]
            if pagination["hasNextPage"] is False:
                if len({p["id"] for p in pods}) != len(pods):
                    raise ValueError("Duplicate inventory IDs")
                return pods
            cursor = pagination["nextCursor"]
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise ValueError("Incomplete inventory pagination")
            seen.add(cursor)


def checked_hardware(hardware):
    """Fail closed unless the plan's hardware block has exactly the expected shape."""
    if (not isinstance(hardware, dict) or hardware.get("cloud") != "SECURE"
            or hardware.get("pod_prefix") != PREFIX
            or type(hardware.get("container_disk_gb")) is not int or not 0 < hardware["container_disk_gb"] <= 4000
            or not isinstance(hardware.get("volume_gb"), dict) or set(hardware["volume_gb"]) != set(KINDS)
            or any(type(v) is not int or not 0 < v <= 4000 for v in hardware["volume_gb"].values())
            or _number(hardware.get("storage_hourly_usd_bound")) <= 0):
        raise ValueError("Plan hardware block missing or outside controller invariants")
    for kind in KINDS:
        options = hardware.get(kind)
        if not isinstance(options, list) or not options or (kind == "cheap" and len(options) != 1):
            raise ValueError("Frozen hardware alternatives missing")
        for option in options:
            keys = {"gpu", "count", "max_gpu_hourly_usd"} | ({"max_memory_gib"} if kind == "main" else set())
            if (not isinstance(option, dict) or set(option) != keys or option["gpu"] not in MIN_MEMORY_GIB
                    or type(option["count"]) is not int or not 1 <= option["count"] <= 8
                    or _number(option["max_gpu_hourly_usd"]) <= 0
                    or kind == "main" and (type(option["max_memory_gib"]) is not int or not
                                           0 < option["max_memory_gib"] < MIN_MEMORY_GIB[option["gpu"]])):
                raise ValueError("Unknown or unsafe hardware alternative")
    return hardware


def checked_budget(budget):
    """The plan's caps must equal the controller's; returns (prior compute, main reserve seconds)."""
    if (not isinstance(budget, dict)
            or _number(budget.get("total_cap_usd")) != TOTAL_CAP
            or _number(budget.get("gpu_cap_usd")) != COMPUTE_CAP
            or _number(budget.get("cheap_gpu_cap_usd")) != CHEAP_CAP
            or _number(budget.get("storage_retrieval_reserve_usd")) != RETRIEVAL_RESERVE
            or type(budget.get("main_retrieval_reserve_seconds")) is not int
            or not 0 < budget["main_retrieval_reserve_seconds"] < 3600):
        raise ValueError("Plan budget differs from the controller's frozen caps")
    return _number(budget.get("prior_compute_usd", 0)), budget["main_retrieval_reserve_seconds"]


def quote_option(api, option):
    """One read-only catalog GET. Malformed quotes raise; unavailable ones do not qualify."""
    gpu, count = option["gpu"], option["count"]
    _, item = api.request("GET", "/catalog/gpus/" + urllib.parse.quote(gpu, safe="")
                          + "?include=AVAILABILITY&product=POD&cloud=SECURE&count=" + str(count)
                          + "&minCudaVersion=12.8")
    if not isinstance(item, dict) or item.get("id") != gpu or not isinstance(item.get("price"), dict):
        raise ValueError("Unknown hardware quote")
    rate, memory = _number(item["price"].get("secure")), _number(item.get("memory"))
    if rate <= 0:
        raise ValueError("Unknown or invalid secure price")
    availability = item.get("availability")
    return {"gpu": gpu, "count": count, "secure_gpu_hourly_usd": str(rate), "memory_gib": str(memory),
            "availability": availability if isinstance(availability, str) else None,
            "secure": item.get("secure") is True,
            "qualifies": (item.get("secure") is True and availability in AVAILABLE
                          and rate <= _number(option["max_gpu_hourly_usd"]) and memory >= MIN_MEMORY_GIB[gpu])}


def select_hardware(api, kind, hardware):
    """Return the FIRST qualifying frozen alternative, its total hourly quote and the checks made.

    Read-only; if no alternative qualifies, raise before any ledger binding or creation.
    """
    if kind not in KINDS:
        raise ValueError("Unknown pod kind")
    storage = _number(hardware["storage_hourly_usd_bound"])
    checked = []
    for option in hardware[kind]:
        evaluation = quote_option(api, option)
        checked.append(evaluation)
        if evaluation["qualifies"]:
            rate = _number(evaluation["secure_gpu_hourly_usd"]) * option["count"]
            return dict(option), {"hourly_rate_usd": str(rate), "storage_hourly_usd": str(storage)}, checked
    raise ValueError("No frozen hardware alternative is available within its price ceiling; nothing created")


def verify_public(plan_hash, relative, freeze):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    urls = [RAW + freeze + "/" + relative,
            "https://hub.docker.com/v2/repositories/runpod/pytorch/tags/" + IMAGE_TAG]
    for index, url in enumerate(urls):
        with opener.open(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
            raw = response.read()
        if index == 0 and hashlib.sha256(raw).hexdigest() != plan_hash:
            raise ValueError("Public freeze plan differs")
        if index == 1:
            image = strict_json(raw)
            if image.get("tag_status") != "active" or image.get("digest") != IMAGE_DIGEST:
                raise ValueError("Public image tag missing or digest changed")


def require_ignored(path):
    """Read-only Git check: lifecycle state (pod IDs, endpoints, approvals) must never be stageable."""
    try:
        relative = Path(path).resolve().relative_to(common.ROOT).as_posix()
    except ValueError:
        return  # Outside the repository: cannot be committed.
    result = subprocess.run(["git", "check-ignore", "-q", "--", relative], cwd=common.ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=local_env())
    if result.returncode != 0:
        raise ValueError("Lifecycle state must live under a Git-ignored path")


def create_payload(kind, name, public_key, option, hardware):
    if kind not in KINDS or not re.fullmatch(re.escape(PREFIX) + kind + r"-[0-9a-f]{12}", name):
        raise ValueError("Unique owned-pod name required")
    if not re.fullmatch(r"(?:ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp256) [A-Za-z0-9+/=]+(?: [^\r\n]+)?", public_key):
        raise ValueError("Valid public SSH key required")
    if not isinstance(option, dict) or option not in hardware[kind]:
        raise ValueError("Hardware is not a frozen alternative for this kind")
    return {"name": name, "image": IMAGE, "cloud": "SECURE",
            "gpu": {"id": option["gpu"], "count": option["count"], "minCudaVersion": "12.8"},
            "disk": hardware["container_disk_gb"],
            "mounts": {"persistent": {"size": hardware["volume_gb"][kind], "path": "/workspace"}},
            "ports": ["22/tcp"], "startSsh": True, "startJupyter": False,
            "env": {"PUBLIC_KEY": public_key}}


def clean_pod(pod):
    return {key: pod[key] for key in ("id", "name", "status", "createdAt", "cost", "gpu", "cloud",
                                      "image", "disk", "mounts", "ports", "ssh") if key in pod}


def local_env():
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(Path.home()), "LANG": "C.UTF-8"}


def ssh_args(pod, key, known_hosts):
    direct = pod.get("ssh", {}).get("direct")
    if not isinstance(direct, dict):
        raise ValueError("Direct SSH not ready")
    host, port, user = direct["host"], direct["port"], direct["username"]
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", host) or user != "root"
            or type(port) is not int or not 1 <= port <= 65535):
        raise ValueError("Unsafe SSH endpoint")
    return ["ssh", "-F", "/dev/null", "-i", str(Path(key).expanduser()), "-p", str(port),
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ForwardAgent=no",
            "-o", "ClearAllForwardings=yes", "-o", "PermitLocalCommand=no",
            "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(known_hosts), user + "@" + host]


def _safe_relative(relative):
    path = PurePosixPath(relative) if isinstance(relative, str) else None
    if (path is None or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]*", relative)
            or path.is_absolute() or ".." in path.parts or path.as_posix() != relative):
        raise ValueError("Unsafe frozen source path")


def worker_script(kind, plan_relative, freeze, deadline, hardware=None):
    """Bootstrap at the exact freeze; main runs pod_runner, cheap runs the CUDA tests.

    The cheap pod always records its pytest exit code in DONE-all.json and exits
    with it, so failing tests end the pod like passing ones and stay visible.
    """
    if kind not in KINDS or not isinstance(freeze, str) or not re.fullmatch(r"[0-9a-f]{40}", freeze):
        raise ValueError("Exact freeze and cheap/main kind required")
    _safe_relative(plan_relative)
    deadline = _utc(deadline).isoformat()
    python = REMOTE + "/venv/bin/python"
    if kind == "main":
        if not isinstance(hardware, dict):
            raise ValueError("Main worker requires the chosen frozen hardware record")
        work = [shlex.join([python, "-m", "selfref_scaling.pod_runner", "--plan", plan_relative,
                            "--freeze", freeze, "--out", REMOTE + "/out", "--cache", HF_CACHE,
                            "--deadline-utc", deadline, "--hardware-json", common.canonical(hardware)])]
    else:
        done = ("import json,sys; json.dump({'pytest_exit_code':int(sys.argv[1])},open("
                + repr(REMOTE + "/out/DONE-all.json") + ",'x'))")
        work = ["rc=0",
                "SELFREF_REQUIRE_CUDA=1 " + shlex.join([python, "-m", "pytest", "-q", "-rs", *CHEAP_TESTS])
                + " > " + REMOTE + "/out/cheap-tests.txt || rc=$?",
                shlex.join([python, "-c", done]) + ' "$rc"',
                'exit "$rc"']
    exit_code = "import json,sys; json.dump({'exit_code':int(sys.argv[1])},open('" + REMOTE + "/out/controller-exit.json','x'))"
    return "\n".join([
        "set -euC", "umask 077", "mkdir -p " + REMOTE + "/out",
        "trap 'rc=$?; python3 -c " + shlex.quote(exit_code).replace("'", "'\"'\"'") + " \"$rc\"' EXIT",
        "git -c credential.helper= clone --filter=blob:none --no-checkout " + REPO + " " + REMOTE + "/repo",
        "cd " + REMOTE + "/repo", "git fetch --depth=1 origin " + freeze,
        "git checkout --detach " + freeze, 'test "$(git rev-parse HEAD)" = ' + freeze,
        "python3 -m venv --system-site-packages " + REMOTE + "/venv",
        python + " -m pip install -r requirements-gpu.txt",
        python + " -m pip freeze --all > " + REMOTE + "/out/pip-freeze.txt",
        ". " + REMOTE + "/hf.env", *work,
    ])


# Stdin carries only the dispatched script's SHA-256, never credentials. When
# worker.pid is absent (dispatch never ran), "stop" succeeds only after a
# /proc scan finds no process whose argv contains that exact script.
SIGNAL_SCRIPT = """import hashlib,json,os,pathlib,signal,sys,time
r=pathlib.Path('/workspace/scaling'); p=r/'worker.pid'; action=sys.argv[1]
expected=sys.stdin.read().strip()
if action not in ('pause','resume','stop'): raise RuntimeError('unknown signal action')
def dispatched():
 found=[]
 for c in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try: argv=c.read_bytes().split(b'\\0')
  except OSError: continue
  if any(hashlib.sha256(a).hexdigest()==expected for a in argv): found.append(c.parent.name)
 return found
if p.exists(): pid=int(p.read_text())
elif action=='stop' and len(expected)==64 and not dispatched(): pid=None
else: raise RuntimeError('worker PID missing; dispatch unresolved')
def members():
 result=[]
 for q in pathlib.Path('/proc').glob('[0-9]*/stat'):
  try:
   parts=q.read_text().split(') ',1)[1].split()
   if int(parts[2])==pid and parts[0]!='Z': result.append(parts[0])
  except (FileNotFoundError,ProcessLookupError): pass
 return result
if pid is not None and members():
 proc=pathlib.Path('/proc')/str(pid)
 if not proc.exists() or os.getpgid(pid)!=pid or b'/workspace/scaling' not in (proc/'cmdline').read_bytes():
  raise RuntimeError('worker identity mismatch')
 if action=='resume': os.killpg(pid,signal.SIGCONT)
 elif action=='pause':
  os.killpg(pid,signal.SIGSTOP)
  for _ in range(300):
   if all(s in ('T','t') for s in members()): break
   time.sleep(0.1)
  else: raise RuntimeError('worker pause unverified')
 else:
  os.killpg(pid,signal.SIGCONT); os.killpg(pid,signal.SIGTERM)
  for _ in range(60):
   if not members(): break
   time.sleep(0.5)
  if members(): os.killpg(pid,signal.SIGKILL)
  for _ in range(120):
   if not members(): break
   time.sleep(0.5)
  if members(): raise RuntimeError('worker did not stop')
if action=='stop':
 q=r/'out/controller-stopped.json'
 if not q.exists():
  with q.open('x') as f:
   json.dump({'stopped':True,'pid':pid,'pid_file_missing':pid is None},f); f.flush(); os.fsync(f.fileno())
print(json.dumps({'action':action,'verified':True,'pid':pid}))
"""


MANIFEST_SCRIPT = """import hashlib,json,pathlib
root=pathlib.Path('/workspace/scaling/out'); result={}
if not root.is_dir(): raise RuntimeError('output directory missing')
for p in sorted(root.rglob('*')):
 if p.is_symlink(): raise RuntimeError('symlink artifact')
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1048576),b''): h.update(block)
  result[p.relative_to(root).as_posix()]=h.hexdigest()
print(json.dumps(result,sort_keys=True))
"""
# The live runner publishes through short-lived *.pending/*.tmp files: skip any
# file that vanishes mid-scan, and report a half-written terminal marker as present.
STATUS_SCRIPT = """import json,pathlib
r=pathlib.Path('/workspace/scaling/out'); d={}
for n in ['DONE-all.json','controller-exit.json','controller-stopped.json']:
 p=r/n
 if p.is_file():
  try: d[n]=json.loads(p.read_text())
  except ValueError: d[n]=None
g={}
for p in r.rglob('*'):
 try:
  if p.is_file() and not p.is_symlink():
   s=p.stat(); g[p.relative_to(r).as_posix()]=[s.st_size,s.st_mtime_ns]
 except FileNotFoundError: pass
d['_progress']=g
print(json.dumps(d))
"""


class Controller:
    def __init__(self, plan_path, freeze, out, kind, api, *, run=subprocess.run, clock=now, sleep=time.sleep):
        if kind not in KINDS or not isinstance(freeze, str) or not re.fullmatch(r"[0-9a-f]{40}", freeze):
            raise ValueError("Exact freeze and cheap/main kind required")
        self.plan_path, self.freeze, self.kind = Path(plan_path).resolve(), freeze, kind
        self.plan = design.load_plan(self.plan_path, freeze)
        self.hardware = checked_hardware(self.plan.get("hardware"))
        checked_budget(self.plan.get("budget"))
        self.plan_hash, self.relative = sha(self.plan_path), self.plan_path.relative_to(common.ROOT).as_posix()
        _safe_relative(self.relative)
        self.out = Path(out).resolve()
        self.base = self.out / "controller" / kind
        self.api, self.run, self.clock, self.sleep = api, run, clock, sleep
        self.guard = None
        self.ledger = EventLedger(self.base / "events.jsonl", self.plan_hash, freeze, [])
        self.ledger.bind("controller:config", {"kind": kind, "plan_path": self.relative, "image": IMAGE,
                                               "prefix": PREFIX, "remote": REMOTE})

    def event(self, identifier):
        return next((r for r in self.ledger.read() if r["id"] == identifier), None)

    def record(self, prefix, payload):
        return self.ledger.transact(prefix + ":" + str(len(self.ledger.read())), lambda _: payload)

    def owned(self):
        event = self.event("created")
        if event is None or event["data"]["id"] == BLOCKED:
            raise ValueError("No owned creation receipt; refusing pod access")
        return event["data"]

    def get_pod(self):
        owned = self.owned()
        _, pod = self.api.request("GET", "/pods/" + owned["id"])
        if pod["id"] != owned["id"] or pod["name"] != owned["name"] or pod["createdAt"] != owned["createdAt"]:
            raise ValueError("Owned pod identity changed")
        return pod

    def _ssh(self, pod, command, *, data=None, timeout=60):
        result = self.run(ssh_args(pod, KEY, self.base / "known_hosts") + [command], input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=local_env())
        if result.returncode:
            raise RuntimeError("Owned-pod SSH failed; stderr withheld to avoid credential leakage")
        return result.stdout

    def cheap_gate(self):
        """Main needs the cheap pod's verified deletion and a human approval bound to this plan."""
        cheap_path = self.out / "controller" / "cheap" / "events.jsonl"
        if not cheap_path.is_file():
            raise ValueError("Cheap qualification has not completed")
        cheap = EventLedger(cheap_path, self.plan_hash, self.freeze, [])
        closed = next((r for r in cheap.read() if r["id"] == "closed"), None)
        approval = self.out / "APPROVE-cheap"
        if (closed is None or closed["data"].get("get_status") != 404 or approval.is_symlink()
                or not approval.is_file() or approval.stat().st_size > 1024
                or approval.read_text().strip() != self.plan_hash):
            raise ValueError("Audited cheap qualification and verified deletion required")
        return _number(closed["data"]["compute_upper_bound_usd"])

    def launch(self):
        if not self.api.writable:
            raise ValueError("Explicit --launch required")
        if self.event("create-intent"):
            raise ValueError("Create already attempted; never retry an uncertain creation")
        self.disk_check()
        require_ignored(self.base / "events.jsonl")
        token = os.environ.get("HF_TOKEN", "")
        if not token or any(c.isspace() for c in token):
            raise ValueError("HF_TOKEN missing/malformed before creation")
        if not KEY.expanduser().is_file():
            raise ValueError("Existing private SSH key missing")
        prior, main_reserve = checked_budget(self.plan.get("budget"))
        if self.kind == "main":
            prior += self.cheap_gate()
            if prior > CHEAP_CAP:
                raise ValueError("Cheap qualification exceeded its cap")
        verify_public(self.plan_hash, self.relative, self.freeze)
        # Read-only quotes come before any ledger binding, so "nothing available" can be retried later.
        option, quoted, checked = select_hardware(self.api, self.kind, self.hardware)
        quoted_at = self.clock()
        key = (Path(str(KEY.expanduser()) + ".pub")).read_text().strip()
        payload = create_payload(self.kind, PREFIX + self.kind + "-" + uuid.uuid4().hex[:12], key,
                                 option, self.hardware)
        created = self.clock()
        rate = _number(quoted["hourly_rate_usd"]) + _number(quoted["storage_hourly_usd"])
        reserve_seconds = main_reserve if self.kind == "main" else CHEAP_RESERVE_SECONDS
        limit = (CHEAP_CAP if self.kind == "cheap" else COMPUTE_CAP) - prior
        # Keep the $5 reserve inside the GPU cap, plus reserve_seconds of the full
        # pod rate after the worker deadline for final retrieval and verified deletion.
        worker_dollars = min(limit, COMPUTE_CAP - RETRIEVAL_RESERVE - prior) - rate * reserve_seconds / 3600
        if worker_dollars <= 0:
            raise ValueError("No funded startup/retrieval window")
        deadline = created + timedelta(seconds=float(worker_dollars / rate * 3600))
        if self.kind == "cheap":
            deadline = min(deadline, created + timedelta(seconds=CHEAP_LIFETIME_SECONDS - CHEAP_RESERVE_SECONDS))
        if deadline <= created + timedelta(seconds=MONITOR_HORIZON_SECONDS):
            raise ValueError("No funded startup/retrieval window")
        inventory = self.api.inventory()
        blocked = sorted({BLOCKED} | {p["id"] for p in inventory})
        PodRegistry(self.ledger, blocked)
        guard = BudgetGuard(self.ledger, quoted, created, deadline, max_compute=COMPUTE_CAP - prior, clock=self.clock)
        self.guard = guard
        guard.before_batch("create", 600, 0, 0)
        intent = {"payload": payload, "hardware": option, "quote": quoted, "quotes_checked": checked,
                  "created_utc": created.isoformat(), "deadline_utc": deadline.isoformat(),
                  "retrieval_reserve_seconds": reserve_seconds, "prior_compute_usd": str(prior),
                  "local_cap_usd": str(limit), "blocked": blocked, "plan_sha256": self.plan_hash,
                  "freeze_commit": self.freeze}
        self.ledger.transact("create-intent", lambda _: intent)
        if not 0 <= (self.clock() - quoted_at).total_seconds() <= 60:
            raise ValueError("Quote stale before creation; do not retry")
        try:
            status, pod = self.api.request("POST", "/pods", payload)  # Exactly one attempt.
            if status != 201 or not self._new_pod(pod, intent):
                raise ValueError("Ambiguous creation response")
        except (ApiError, RuntimeError, ValueError, OSError, KeyError, TypeError):
            pod = self.reconcile_create()  # Exact persisted unique name, never another POST.
        self._register(pod)
        try:
            self.start_worker()
        except Exception as exc:
            self.record("launch-failed", {"error_type": type(exc).__name__})
            if self.event("worker-intent"):
                self.quiesce()
            self.terminate()
            raise
        return self.owned()

    def _guard(self, intent):
        if self.guard is None:
            self.guard = BudgetGuard(self.ledger, intent["quote"], intent["created_utc"], intent["deadline_utc"],
                                     max_compute=COMPUTE_CAP - _number(intent["prior_compute_usd"]), clock=self.clock)
        return self.guard

    def _register(self, pod):
        intent = self.event("create-intent")["data"]
        receipt = self.ledger.bind("created", clean_pod(pod))
        registry = PodRegistry(self.ledger, intent["blocked"])
        if not self.event("pod:created:" + pod["id"]):
            registry.register_created(pod["id"], receipt["sha256"], [str(self.base / "final-retrieval.json")])
        if not self.event("budget:finish:create"):
            self._guard(intent).finish_batch("create", 0, 0, receipt["sha256"])

    def reconcile(self):
        """Recover persisted single-create intent by GET only; never starts a worker."""
        if not self.event("create-intent"):
            raise ValueError("No creation intent to reconcile")
        if not self.event("created"):
            self._register(self.reconcile_create())
        return self.owned()

    def _new_pod(self, pod, intent):
        return (isinstance(pod, dict) and isinstance(pod.get("id"), str)
                and re.fullmatch(r"[A-Za-z0-9_-]+", pod["id"]) and pod["id"] not in intent["blocked"]
                and pod.get("name") == intent["payload"]["name"]
                and _utc(pod["createdAt"]) >= _utc(intent["created_utc"]) - timedelta(seconds=60))

    def reconcile_create(self):
        intent = self.event("create-intent")["data"]
        for attempt in range(12):
            matches = [p for p in self.api.inventory() if p.get("name") == intent["payload"]["name"]]
            if len(matches) > 1:
                raise ValueError("Ambiguous exact-name inventory; manual reconciliation required")
            if matches:
                if not self._new_pod(matches[0], intent):
                    raise ValueError("Reconciled pod identity invalid")
                self.record("create-reconciled", {"pod": clean_pod(matches[0]), "attempt": attempt})
                return matches[0]
            self.sleep(5)
        raise RuntimeError("Create outcome unresolved: retain intent and reconcile exact name; never create again")

    def cost_check(self, pod, horizon=60):
        intent = self.event("create-intent")["data"]
        rate = _number(pod.get("cost"))
        quote_rate = _number(intent["quote"]["hourly_rate_usd"])
        storage = _number(intent["quote"]["storage_hourly_usd"])
        expected, hardware, gpu = intent["payload"], intent["hardware"], pod.get("gpu")
        if (not 0 < rate <= quote_rate or pod.get("cloud") != "SECURE"
                or not isinstance(gpu, dict) or gpu.get("id") != hardware["gpu"]
                or type(gpu.get("count")) is not int or gpu["count"] != hardware["count"]
                or any(pod.get(k) != expected[k] for k in ("image", "disk", "mounts", "ports"))):
            raise ValueError("Unknown billing rate or hardware drift")
        prior = _number(intent["prior_compute_usd"])
        tick = "monitor-" + str(len(self.ledger.read()))
        reserved = self._guard(intent).before_batch(tick, horizon, 0, 0)
        spent = _number(reserved["data"]["compute_accounted_usd"])
        self.guard.finish_batch(tick, spent, 0, reserved["sha256"])
        projected = spent + Decimal(horizon) * (quote_rate + storage) / 3600
        if (projected > _number(intent["local_cap_usd"]) or
                projected + prior + RETRIEVAL_RESERVE > COMPUTE_CAP or
                self.clock() + timedelta(seconds=horizon) >= _utc(intent["deadline_utc"])):
            raise ValueError("Retrieval reserve/deadline reached")
        return spent

    def start_worker(self):
        if self.event("worker-intent"):
            raise ValueError("Worker dispatch uncertain/already attempted; no duplicate worker")
        intent = self.event("create-intent")["data"]
        for _ in range(40):
            pod = self.get_pod()
            self.cost_check(pod, 600)
            if pod.get("status") == "RUNNING" and pod.get("ssh", {}).get("direct"):
                try:
                    self._ssh(pod, "true", timeout=15)
                    break
                except (RuntimeError, subprocess.TimeoutExpired):
                    pass
            if pod.get("status") not in {"PROVISIONING", "STARTING", "RUNNING"}:
                raise ValueError("Owned pod failed startup")
            self.sleep(15)
        else:
            raise TimeoutError("SSH startup exceeded bounded readiness window")
        token = os.environ.get("HF_TOKEN", "")
        if not token or any(c.isspace() for c in token):
            raise ValueError("HF_TOKEN missing or malformed; worker not started")
        self._ssh(pod, "umask 077; mkdir -p " + REMOTE + "/out; cat > " + REMOTE + "/hf.env; chmod 600 " + REMOTE + "/hf.env",
                  data=("export HF_TOKEN=" + shlex.quote(token) + "\n").encode())
        script = worker_script(self.kind, self.relative, self.freeze, intent["deadline_utc"], intent["hardware"])
        seconds = int((_utc(intent["deadline_utc"]) - self.clock()).total_seconds())
        if seconds <= 60:
            raise ValueError("No remaining worker window")
        self.ledger.transact("worker-intent", lambda _: {"script_sha256": hashlib.sha256(script.encode()).hexdigest(),
                                                         "seconds": seconds})
        command = ("nohup setsid timeout --signal=TERM --kill-after=30s " + str(seconds)
                   + "s env -i HOME=/root"
                   + " PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                   + " LD_LIBRARY_PATH=/usr/local/cuda/lib64 bash --noprofile --norc -c "
                   + shlex.quote(script) + " >" + REMOTE
                   + "/out/controller.log 2>&1 </dev/null & echo $! > " + REMOTE + "/worker.pid")
        self._ssh(pod, command)
        self.ledger.transact("worker-started", lambda _: {"utc": self.clock().isoformat()})

    def status(self):
        pod = self.get_pod()
        state = {"pod": clean_pod(pod), "files": {}}
        if pod.get("status") == "RUNNING" and pod.get("ssh", {}).get("direct"):
            state["files"] = strict_json(self._ssh(pod, "python3 -c " + shlex.quote(STATUS_SCRIPT)))
        return state

    def quiesce(self):
        pod = self.get_pod()
        result = self.signal_worker(pod, "stop")
        self.record("worker-stopped", result)

    def signal_worker(self, pod, action):
        if action not in {"pause", "resume", "stop"} or pod["id"] != self.owned()["id"]:
            raise ValueError("Unknown worker or signal")
        intent = self.event("worker-intent")
        expected = intent["data"]["script_sha256"] if intent else ""
        result = strict_json(self._ssh(pod, "python3 -c " + shlex.quote(SIGNAL_SCRIPT) + " " + action,
                                       data=(expected + "\n").encode(), timeout=SIGNAL_TIMEOUT))
        if result.get("verified") is not True or result.get("action") != action:
            raise ValueError("Owned worker signal unverified")
        return result

    def retrieve(self, *, final=False):
        self.disk_check()
        pod = self.get_pod()
        if final and not self.event("worker-intent"):
            destination = self.base / "retrievals" / ("no-worker-" + uuid.uuid4().hex)
            destination.mkdir(parents=True, exist_ok=False)
            receipt = self.record("retrieval", {"pod_id": pod["id"], "no_worker_dispatched": True,
                                               "directory": str(destination), "artifacts": {},
                                               "pod": clean_pod(pod)})
            self._final_receipt(receipt)
            return receipt
        if final:
            self.quiesce()
            return self._snapshot(pod, final=True)
        try:
            self.signal_worker(pod, "pause")
            return self._snapshot(pod, final=False)
        finally:
            self.signal_worker(pod, "resume")

    def _snapshot(self, pod, *, final):
        manifest_command = "python3 -c " + shlex.quote(MANIFEST_SCRIPT)
        before = strict_json(self._ssh(pod, manifest_command, timeout=MANIFEST_TIMEOUT))
        if not isinstance(before, dict):
            raise ValueError("Unsafe remote manifest")
        for name, digest in before.items():
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Unsafe remote manifest")
        destination = self.base / "retrievals" / uuid.uuid4().hex
        destination.mkdir(parents=True, exist_ok=False)
        ssh = ssh_args(pod, KEY, self.base / "known_hosts")
        previous = next((r["data"]["directory"] for r in reversed(self.ledger.read())
                         if r["id"].startswith("retrieval:") and "directory" in r["data"]), None)
        reuse = ["--link-dest=" + str(Path(previous).resolve())] if previous else []
        result = self.run(["rsync", "-rt", "--safe-links", "--no-links", *reuse, "-e", shlex.join(ssh[:-1]),
                           ssh[-1] + ":" + REMOTE + "/out/", str(destination) + "/"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=RSYNC_TIMEOUT, env=local_env())
        if result.returncode:
            raise RuntimeError("Retrieval failed; owned pod retained")
        after = strict_json(self._ssh(pod, manifest_command, timeout=MANIFEST_TIMEOUT))
        local = {p.relative_to(destination).as_posix(): sha(p) for p in destination.rglob("*") if p.is_file() and not p.is_symlink()}
        if before != after or before != local:
            raise ValueError("Live snapshot changed or local artifact hashes differ; retry read-only retrieval")
        receipt = self.record("retrieval", {"pod_id": pod["id"], "directory": str(destination), "artifacts": local,
                                          "utc": self.clock().isoformat()})
        if final:
            if self.event("worker-intent") and not any(name in local for name in TERMINAL):
                raise ValueError("Worker has not stopped; refusing destructive cleanup")
            self._final_receipt(receipt)
        return receipt

    def disk_check(self):
        if shutil.disk_usage(self.base).free < 8 * 1024 ** 3:
            raise ValueError("At least 8 GiB free local disk required; old raw snapshots are never deleted")

    def _final_receipt(self, receipt):
        with (self.base / "final-retrieval.json").open("xb") as handle:
            handle.write(_canonical(receipt) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _verify_final(self):
        receipt = strict_json((self.base / "final-retrieval.json").read_bytes())
        if self.event(receipt["id"]) != receipt:
            raise ValueError("Final retrieval receipt is not bound to this ledger")
        data = receipt["data"]
        for name, digest in data["artifacts"].items():
            path = Path(data["directory"]) / name
            if path.is_symlink() or not path.is_file() or sha(path) != digest:
                raise ValueError("Retrieved artifact missing or corrupt")

    def terminate(self):
        if not self.api.writable:
            raise ValueError("Explicit --launch required for termination")
        if self.event("closed"):
            return self.event("closed")
        if self.event("delete-intent"):
            return self._confirm_closed(self.event("delete-intent")["data"]["pod"])
        pod = self.get_pod()
        if not self.event("delete-intent"):
            if not (self.base / "final-retrieval.json").exists():
                self.retrieve(final=True)
            self._verify_final()
            manifest_path = self.base / "final-retrieval.json"
            registry = PodRegistry(self.ledger, self.event("create-intent")["data"]["blocked"])
            permit = registry.authorize_delete(pod["id"], {str(manifest_path): sha(manifest_path)})
            self.ledger.transact("delete-intent", lambda _: {"permit_sha256": permit["sha256"], "pod_id": pod["id"], "pod": clean_pod(pod)})
            status, _ = self.api.request("DELETE", "/pods/" + pod["id"])  # Exactly one attempt.
            if status != 204:
                raise ValueError("Unexpected delete response")
        else:
            raise ValueError("Deletion already attempted; reconcile read-only, never repeat blindly")
        return self._confirm_closed(pod)

    def _confirm_closed(self, pod):
        for _ in range(12):  # Read-only: the provider may list a deleted pod briefly.
            try:
                self.api.request("GET", "/pods/" + pod["id"])
            except ApiError as exc:
                if exc.status != 404:
                    raise
                break
            self.sleep(5)
        else:
            raise ValueError("Deletion not verified")
        if pod["id"] in {p["id"] for p in self.api.inventory()}:
            raise ValueError("Deleted pod still in inventory")
        intent = self.event("create-intent")["data"]
        cost = None
        if pod.get("cost") is not None:
            elapsed = max(_number((self.clock() - _utc(intent["created_utc"])).total_seconds()),
                          max((_number(r["data"]["elapsed_seconds"]) for r in self.ledger.read()
                               if "elapsed_seconds" in r["data"]), default=Decimal(0)))
            rate = max(_number(pod["cost"]), _number(intent["quote"]["hourly_rate_usd"]))
            cost = str(elapsed * (rate + _number(intent["quote"]["storage_hourly_usd"])) / 3600)
        return self.ledger.transact("closed", lambda _: {"pod_id": pod["id"], "compute_upper_bound_usd": cost,
                                                        "utc": self.clock().isoformat(), "get_status": 404})

    def monitor(self):
        """Cost check each minute (600 s horizon), snapshot every 10 minutes, then close.

        Terminal marker: final retrieval and verified deletion. Budget, deadline or
        stall: stop the worker first. A failing status/snapshot is retried on the
        next tick; three consecutive failures are treated as a stop.
        """
        if not self.api.writable:
            raise ValueError("Lifecycle monitor requires explicit --launch")
        if self.event("closed") or self.event("delete-intent"):
            return self.terminate()
        last_pull, previous, changed_at, failures = float("-inf"), None, time.monotonic(), 0
        while True:
            reason, terminal = None, False
            try:
                state = self.status()
                self.record("health", state)
                terminal = any(name in state["files"] for name in TERMINAL)
                progress = state["files"].get("_progress", {})
                if progress != previous:
                    previous, changed_at = progress, time.monotonic()
                try:
                    self.cost_check(state["pod"], MONITOR_HORIZON_SECONDS)
                    if not terminal and time.monotonic() - changed_at >= STALL_SECONDS:
                        raise ValueError("Owned worker stalled")
                except ValueError as exc:
                    reason = str(exc)
                if reason is None and terminal:
                    reason = "terminal marker"
                if reason is None and time.monotonic() - last_pull >= SNAPSHOT_SECONDS:
                    self.retrieve()
                    last_pull = time.monotonic()
                failures = 0
            except (RuntimeError, OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
                failures += 1
                self.record("monitor-retry", {"error_type": type(exc).__name__, "consecutive": failures})
                if failures >= MONITOR_FAILURE_LIMIT:
                    reason, terminal = "repeated monitoring failures", False
            if reason is not None:
                self.record("monitor-stop", {"reason": reason, "terminal": terminal})
                if not terminal and self.event("worker-intent"):
                    self.quiesce()
                return self.terminate()
            self.sleep(MONITOR_TICK_SECONDS)


def _dry_run(args):
    hardware = checked_hardware(design.HARDWARE)
    storage = _number(hardware["storage_hourly_usd_bound"])
    steps = (["require the cheap ledger's closed event (GET 404) and APPROVE-cheap holding the plan hash"]
             if args.kind == "main" else [])
    steps += ["verify the public plan at the freeze and the pinned image digest",
              "read-only catalog quotes; take the first qualifying alternative or create nothing",
              "block the protected pod and every pod already listed; one POST /pods",
              "start the worker at the exact freeze under a remote timeout",
              "monitor: cost check each minute, snapshot every 10 minutes, stall after 900 s",
              "final hash-verified retrieval, one DELETE, direct GET 404 and inventory absence"]
    return {"dry_run": True, "action": args.action, "kind": args.kind, "network_calls": 0,
            "protected_pod": BLOCKED, "approval": "never automatic", "prefix": PREFIX, "image": IMAGE,
            "remote": REMOTE, "out_root": str(OUT_ROOT), "gpu_cap_usd": str(COMPUTE_CAP),
            "alternatives_in_order": [{**o, "max_hourly_usd_with_storage":
                                       str(_number(o["max_gpu_hourly_usd"]) * o["count"] + storage)}
                                      for o in hardware[args.kind]],
            "would": steps}


def _quote(args):
    api = RunPodV2(os.environ.get("RUNPOD_API_KEY"), writable=False)  # GETs only.
    results = []
    for option in checked_hardware(design.HARDWARE)[args.kind]:
        try:
            results.append(quote_option(api, option))
        except ApiError as exc:
            results.append({"gpu": option["gpu"], "count": option["count"], "qualifies": False, "http_status": exc.status})
        except ValueError as exc:
            results.append({"gpu": option["gpu"], "count": option["count"], "qualifies": False, "error": str(exc)})
    return {"read_only": True, "kind": args.kind, "quotes": results,
            "would_select": next((r for r in results if r["qualifies"]), None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=design.PLAN_PATH)
    parser.add_argument("--freeze")
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--action", choices=("launch", "status", "retrieve", "monitor", "terminate",
                                             "reconcile", "quote"), default="launch")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--read-only-quote", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "quote":
        if not args.read_only_quote or args.launch:
            parser.error("--action quote requires --read-only-quote and never --launch")
        print(json.dumps(_quote(args), sort_keys=True))
        return
    if args.read_only_quote:
        parser.error("--read-only-quote applies only to --action quote")
    if not args.launch:
        print(json.dumps(_dry_run(args), sort_keys=True))
        return
    if not args.freeze or not re.fullmatch(r"[0-9a-f]{40}", args.freeze):
        parser.error("--launch requires the full 40-hex --freeze commit")
    plan = Path(args.plan)
    plan = plan if plan.is_absolute() else common.ROOT / plan
    api = RunPodV2(os.environ.get("RUNPOD_API_KEY"), writable=True)
    controller = Controller(plan, args.freeze, OUT_ROOT, args.kind, api)
    print(json.dumps(getattr(controller, args.action)(), sort_keys=True))


if __name__ == "__main__":
    main()
