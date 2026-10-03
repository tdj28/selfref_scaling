"""One append-only, hash-chained, process-locked journal for every paid API call.

Reserve before dispatch: the worst-case cost of the exact request is fsynced
before the network call and refused if it would exceed the caller's cap
(a Decimal or a callable returning one). Each reservation is sent at most once
and resolved by exactly one result whose cost is rebuilt from returned usage,
or the full reservation when usage is unknown. The only further attempt for a
slot is a schema retry licensed by the binding's ``max_attempts`` (at most 2),
and it needs its own reservation. Reopening replays every event: a torn tail,
broken chain, duplicate, non-reconstructing cost/status/model, a reservation
without a result, or a persisted contract failure refuses further dispatch.
Recovery is a documented manual decision, never an automatic retry.

Money arithmetic copies CONSCIOUS fe4b831: ``reservation_usd`` is the
qualification/A1 ``reservation`` and ``usage_cost_usd`` the A1
``_checked_cost``, with prices passed explicitly and Decimal results.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import os
from pathlib import Path
import re
import stat
import threading

from .common import canonical, canonical_text, digest, strict_json, text_digest

SCHEMA = "selfref_paid_ledger_v1"
JOURNAL, LOCK = "events.jsonl", ".paid.lock"
PROVIDERS = ("openai", "anthropic")
EVENT_KEYS = {"seq", "previous", "utc", "kind", "data", "sha256"}
SPEC_KEYS = {"slot", "attempt", "provider", "model", "request", "context"}
# Contract failures: the provider may have charged, the instrument or the
# accounting is compromised, so nothing more is dispatched from this journal.
FATAL = frozenset({"transport_unknown", "usage_failure", "model_drift",
                   "budget_contract_failure", "evaluation_error"})
_DATED = r"-20\d{2}-\d{2}-\d{2}"


class Halted(RuntimeError):
    """No automatic retry or resume; a documented decision is required."""


class BudgetExceeded(Halted):
    """The next reservation would exceed the caller's cap."""


def money(value) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("Money must be a decimal string, integer or Decimal")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError("Invalid money value") from None
    if not number.is_finite() or number < 0:
        raise ValueError("Money must be finite and nonnegative")
    return number


def usd(value) -> str:
    """Canonical plain decimal string (no exponent, no trailing zeros)."""
    return format(money(value).normalize(), "f")


def reservation_usd(prices, request) -> Decimal:
    # A byte-count input bound plus message/schema overhead, at cache-write
    # rather than ordinary input rates, plus the full output cap.
    inputs = len(canonical_text(request).encode("utf-8")) + 4096
    outputs = request.get("max_output_tokens", request.get("max_tokens"))
    if type(outputs) is not int or outputs <= 0:
        raise ValueError("Invalid output token reservation")
    a, b = (money(p) for p in prices)
    return (Decimal(inputs) * a * Decimal("1.25") + Decimal(outputs) * b) / 1_000_000


def usage_cost_usd(provider, prices, raw) -> Decimal:
    """Upper bound from returned usage; never credits cache-read discounts."""
    if provider not in PROVIDERS:
        raise ValueError("Unknown provider")
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Missing usage")
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = usage.get(key, 0 if key.startswith("cache_") else None)
        if type(value) is not int or value < 0:
            raise ValueError("Invalid usage")
    details = usage.get("input_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            raise ValueError("Invalid cache usage")
        for key in ("cache_write_tokens", "cached_tokens", "cache_read_tokens"):
            value = details.get(key, 0)
            if type(value) is not int or not 0 <= value <= usage["input_tokens"]:
                raise ValueError("Invalid cache usage")
    inputs = Decimal(usage["input_tokens"] + usage.get("cache_read_input_tokens", 0))
    writes = Decimal(usage.get("cache_creation_input_tokens", 0))
    inputs = (inputs + writes) * Decimal("1.25") if provider == "openai" else inputs + writes * Decimal("1.25")
    a, b = (money(p) for p in prices)
    return (inputs * a + Decimal(usage["output_tokens"]) * b) / 1_000_000


def model_matches(requested, returned) -> bool:
    """Exact ID, or a dated snapshot of an undated alias (A1 and frontier rules)."""
    if not isinstance(returned, str):
        return False
    if returned == requested:
        return True
    return (re.search(_DATED + "$", requested) is None
            and re.fullmatch(re.escape(requested) + _DATED, returned) is not None)


def parse_journal(data: bytes):
    """Verify framing, canonical bytes, sequence and hash chain; return events."""
    if data and not data.endswith(b"\n"):
        raise Halted("Torn journal tail; preserve and reconcile, never truncate")
    events, previous = [], None
    for number, line in enumerate(data.splitlines(), 1):
        try:
            row = strict_json(line)
        except (ValueError, UnicodeDecodeError):
            raise Halted("Unreadable journal line") from None
        if not isinstance(row, dict) or set(row) != EVENT_KEYS or canonical(row).encode() != line:
            raise Halted("Malformed or noncanonical journal event")
        payload = {k: v for k, v in row.items() if k != "sha256"}
        if (type(row["seq"]) is not int or row["seq"] != number or row["previous"] != previous
                or digest(payload) != row["sha256"]):
            raise Halted("Journal hash chain mismatch")
        previous = row["sha256"]
        events.append(row)
    return events


def read_journal(root):
    path = Path(root) / JOURNAL
    return parse_journal(path.read_bytes()) if path.exists() else []


def reserve_in_order(ledger, specs, settle, stopped):
    """Reserve ``specs`` together, or return None once ``stopped()``.

    Open reservations count at their full worst case, so a refusal is final only
    when ``settle()`` (wait for the caller's in-flight calls; True if it had any)
    had nothing left to wait for. The budget stop point then does not depend on
    thread timing, and the cap is never exceeded.
    """
    while not stopped():
        try:
            return ledger.reserve(*specs)
        except BudgetExceeded:
            if not settle():
                raise
    return None


def _no_symlinks(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Ledger paths cannot contain traversal or symlinks")


def _merged(call, result):
    return deepcopy({**call, **(result or {})})


def _json_copy(value):
    """JSON-normal deep copy, so live state equals what a replay reads back."""
    return strict_json(canonical(value))


class Ledger:
    """``with Ledger(root, evaluate=f, binding=b, cap=c) as ledger:``

    ``binding`` (writer mode) must hold ``schema``, ``purpose``,
    ``prices_per_million`` {model id: [input, output]} and ``max_attempts``;
    it is the first journal event and must match on every reopen. Without a
    binding the journal is replayed read-only. ``evaluate(call, raw)`` must be
    pure: it runs live and again on every replay, and the stored evaluation
    must reproduce. It returns a dict with a string ``status``; ``fatal`` and
    ``retryable`` (licenses the schema retry) are optional booleans.
    """

    def __init__(self, root, *, evaluate, binding=None, cap=None):
        self.root = Path(root).absolute()
        self.evaluate, self.cap = evaluate, cap
        self._given = None if binding is None else _json_copy(binding)
        self.writer = binding is not None
        self.mutex = threading.RLock()
        self._fd = self._lock = None
        self.binding, self.prices, self.max_attempts = None, {}, 1
        self._calls, self._results, self._missing, self._slots = {}, {}, {}, {}
        self._log, self._stale, self._sent, self._fatal, self._models = [], set(), set(), set(), {}
        self._spent, self._seq, self._head, self._size = Decimal(0), 0, None, 0

    # ------------------------------------------------------------- lifecycle
    def __enter__(self):
        if self.binding is not None or self._lock is not None:
            raise RuntimeError("A Ledger object is opened once")
        _no_symlinks(self.root)
        journal = self.root / JOURNAL
        if not self.writer and not journal.exists():
            raise Halted("No journal to read")
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (self.root / LOCK, journal):
            _no_symlinks(path)
        self._lock = os.open(self.root / LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._lock)
            self._lock = None
            raise Halted("Another process owns this ledger") from None
        try:
            existed = journal.exists()
            flags = (os.O_RDWR | os.O_APPEND | os.O_CREAT) if self.writer else os.O_RDONLY
            self._fd = os.open(journal, flags | os.O_NOFOLLOW, 0o600)
            if not stat.S_ISREG(os.fstat(self._fd).st_mode):
                raise Halted("Journal must be a regular file")
            if not existed:
                self._sync_directory()
            data = self._read_all()
            self._size = len(data)
            self._load(parse_journal(data))
        except BaseException:
            self._close()
            raise
        return self

    def __exit__(self, *_):
        self._close()

    def _close(self):
        for name in ("_fd", "_lock"):
            fd = getattr(self, name)
            if fd is not None:
                if name == "_lock":
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                setattr(self, name, None)

    def _read_all(self):
        os.lseek(self._fd, 0, os.SEEK_SET)
        chunks = []
        while chunk := os.read(self._fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)

    def _sync_directory(self):
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _load(self, events):
        if not events:
            if not self.writer:
                raise Halted("Empty journal")
            self._adopt(self._given)
            self._append("binding", self._given)
        else:
            first = events[0]
            if first["kind"] != "binding":
                raise Halted("Journal must begin with its binding")
            if self.writer and first["data"] != self._given:
                raise Halted("Ledger belongs to another binding (plan, purpose, prices or cap)")
            self._adopt(first["data"])
            self._seq, self._head = 1, first["sha256"]
            for event in events[1:]:
                self._replay(event["kind"], event["data"])
                self._seq, self._head = event["seq"], event["sha256"]
        unresolved = self.unresolved()
        if self.writer:
            # Marks the resume boundary: reservations still open now are stale.
            self._append("session", {"unresolved": unresolved})
        self._stale = set(unresolved)

    def _adopt(self, binding):
        if (not isinstance(binding, dict) or binding.get("schema") != SCHEMA
                or not isinstance(binding.get("purpose"), str) or not binding["purpose"]
                or binding.get("max_attempts") not in (1, 2)
                or not isinstance(binding.get("prices_per_million"), dict) or not binding["prices_per_million"]):
            raise ValueError("Binding needs schema, purpose, prices_per_million and max_attempts 1 or 2")
        canonical(binding)
        prices = {}
        for model, pair in binding["prices_per_million"].items():
            if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(p, str) for p in pair):
                raise ValueError("Prices must be two decimal strings per model")
            prices[model] = tuple(money(p) for p in pair)
            if min(prices[model]) <= 0:
                raise ValueError("Prices must be positive")
        self.binding, self.prices, self.max_attempts = deepcopy(binding), prices, binding["max_attempts"]

    # ---------------------------------------------------------------- journal
    def _append(self, kind, data):
        if self._fd is None or not self.writer:
            raise Halted("Writer lock not held")
        if os.fstat(self._fd).st_size != self._size:
            raise Halted("Journal changed outside this process")
        row = {"seq": self._seq + 1, "previous": self._head, "kind": kind, "data": data,
               "utc": datetime.now(timezone.utc).isoformat()}
        row["sha256"] = digest(row)
        encoded = (canonical(row) + "\n").encode("ascii")
        if os.write(self._fd, encoded) != len(encoded):
            raise OSError("Partial journal write; do not dispatch or retry")
        os.fsync(self._fd)
        if os.pread(self._fd, len(encoded), self._size) != encoded:
            raise Halted("Journal readback mismatch")
        self._size += len(encoded)
        self._seq, self._head = row["seq"], row["sha256"]
        self._apply(kind, data)
        return row

    def _apply(self, kind, data):
        if kind == "reserve":
            self._calls[data["attempt_id"]] = data
            self._slots.setdefault(data["slot"], []).append(data["attempt_id"])
            self._spent += Decimal(data["reservation_usd"])
        elif kind == "result":
            call = self._calls[data["attempt_id"]]
            self._results[data["attempt_id"]] = data
            self._log.append(data["attempt_id"])
            if data["fatal"]:
                self._fatal.add(data["attempt_id"])
            self._spent += Decimal(data["cost_usd"]) - Decimal(call["reservation_usd"])
            if isinstance(data["raw"], dict):
                self._models.setdefault((call["provider"], call["model"]), set()).add(data["returned_model"])
        elif kind == "missing":
            self._missing[data["slot"]] = data
            self._slots.setdefault(data["slot"], [])

    def _replay(self, kind, data):
        if kind == "session":
            if data != {"unresolved": self.unresolved()}:
                raise Halted("Session boundary does not reconstruct")
            self._stale = set(data["unresolved"])
        elif kind == "reserve":
            if self._stale or self.fatal():
                raise Halted("Reservation recorded after an unresolved dispatch or a contract failure")
            if not isinstance(data, dict) or not SPEC_KEYS <= set(data):
                raise Halted("Malformed reservation")
            try:
                expected = self._reservation(data, money(data.get("cap_usd")), self._spent)
            except (ValueError, TypeError, KeyError, AttributeError):
                raise Halted("Reservation does not reconstruct") from None
            if expected != data or self._spent + Decimal(data["reservation_usd"]) > Decimal(data["cap_usd"]):
                raise Halted("Reservation does not reconstruct or exceeded its recorded cap")
            self._apply(kind, data)
        elif kind == "result":
            aid = data.get("attempt_id") if isinstance(data, dict) else None
            call = self._calls.get(aid)
            if call is None or aid in self._results or aid in self._stale:
                raise Halted("Unmatched, duplicate or post-restart result")
            if (data.get("raw") is None) == (data.get("error_type") is None):
                raise Halted("Transport error and raw receipt must be exclusive")
            try:
                expected = self._result(call, data.get("raw"), data.get("error_type"), data.get("http_status"))
            except (ValueError, TypeError):
                raise Halted("Result does not reconstruct") from None
            if expected != data:
                raise Halted("Result does not reconstruct from its raw receipt (cost, status, model or evaluation)")
            self._apply(kind, data)
        elif kind == "missing":
            if not isinstance(data, dict) or set(data) != {"slot", "reason", "dependency", "dependency_status"}:
                raise Halted("Malformed missing slot")
            try:
                expected = self._missing_row(data["slot"], data["reason"], data["dependency"])
            except ValueError:
                raise Halted("Missing slot does not reconstruct from its dependency") from None
            if data != expected:
                raise Halted("Missing slot does not reconstruct from its dependency")
            self._apply(kind, data)
        else:
            raise Halted("Unexpected journal event: " + str(kind))

    # ------------------------------------------------------------ reservation
    def _cap_now(self):
        if self.cap is None:
            raise Halted("No cap supplied; reservations are refused")
        value = self.cap() if callable(self.cap) else self.cap
        if not isinstance(value, Decimal):
            raise ValueError("The cap must be a Decimal or a callable returning a Decimal")
        return money(value)

    def _reservation(self, spec, cap, spent):
        slot, attempt, provider, model = spec["slot"], spec["attempt"], spec["provider"], spec["model"]
        if not isinstance(slot, str) or not slot or "#" in slot or provider not in PROVIDERS:
            raise ValueError("Invalid slot or provider")
        if model not in self.prices:
            raise ValueError("Unpriced model: " + str(model))
        if type(attempt) is not int or attempt != self.next_attempt(slot):
            raise Halted("Attempt not licensed: duplicate, replay or unlicensed retry")
        request, context = _json_copy(spec["request"]), _json_copy(spec["context"])
        if not isinstance(request, dict) or request.get("model") != model:
            raise ValueError("Request must be a dictionary naming the reserved model")
        value = reservation_usd(self.prices[model], request)
        return {"attempt_id": f"{slot}#{attempt}", "slot": slot, "attempt": attempt, "provider": provider,
                "model": model, "request": request, "request_sha256": text_digest(request),
                "reservation_usd": usd(value), "cap_usd": usd(cap), "spent_before_usd": usd(spent),
                "context": context}

    def reserve(self, *specs):
        """Durably reserve one next attempt per distinct slot, all or nothing."""
        with self.mutex:
            if self._fd is None or not self.writer:
                raise Halted("Writer lock not held")
            if self._stale:
                raise Halted("Unresolved reservation from an earlier process; reconcile manually, never retry")
            if self.fatal():
                raise Halted("Persisted contract failure; no new dispatch")
            if not specs or any(not isinstance(s, dict) or set(s) != SPEC_KEYS for s in specs):
                raise ValueError("Reservation specs need exactly: " + ", ".join(sorted(SPEC_KEYS)))
            if len({s["slot"] for s in specs}) != len(specs):
                raise ValueError("One attempt per slot per reservation")
            cap, spent, rows = self._cap_now(), self._spent, []
            for spec in specs:
                row = self._reservation(spec, cap, spent)
                spent += Decimal(row["reservation_usd"])
                rows.append(row)
            if spent > cap:
                raise BudgetExceeded(f"Reserving {usd(spent - self._spent)} would exceed the {usd(cap)} cap "
                                     f"after {usd(self._spent)} spent or reserved")
            for row in rows:
                self._append("reserve", row)
            return deepcopy(rows)

    # --------------------------------------------------------------- dispatch
    def dispatch(self, attempt_id, send):
        """The single transport attempt for one reservation; exceptions become transport_unknown."""
        with self.mutex:
            if self._fd is None or not self.writer:
                raise Halted("Writer lock not held")
            call = self._calls.get(attempt_id)
            if call is None or attempt_id in self._results or attempt_id in self._stale or attempt_id in self._sent:
                raise Halted("A reservation is sent exactly once")
            self._sent.add(attempt_id)
            request = deepcopy(call["request"])
        raw = error_type = http_status = None
        try:
            value = send(call["provider"], request)
            if not isinstance(value, dict):
                raise TypeError("Provider result is not a dictionary")
            raw = strict_json(canonical(value))  # What is stored is exactly what is evaluated.
        except Exception as exc:
            # Never retain exception messages, headers, clients or credentials.
            error_type = type(exc).__name__
            status = getattr(exc, "status_code", None)
            http_status = status if type(status) is int else None
            raw = None
        with self.mutex:
            data = self._result(call, raw, error_type, http_status)
            self._append("result", data)
            return _merged(call, data)

    def _result(self, call, raw, error_type, http_status):
        if http_status is not None and type(http_status) is not int:
            raise ValueError("Invalid HTTP status")
        if error_type is not None and not isinstance(error_type, str):
            raise ValueError("Invalid error type")
        reservation = Decimal(call["reservation_usd"])
        row = {"attempt_id": call["attempt_id"], "request_sha256": call["request_sha256"], "raw": raw,
               "error_type": error_type, "http_status": http_status}
        if not isinstance(raw, dict):
            return {**row, "status": "transport_unknown", "fatal": True, "cost_usd": usd(reservation),
                    "cost_known": False, "returned_model": None, "model_drift": False, "evaluated": None}
        try:
            cost, known = usage_cost_usd(call["provider"], self.prices[call["model"]], raw), True
        except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
            cost, known = reservation, False
        evaluated, status = None, "usage_failure"
        if known:
            try:
                evaluated = _json_copy(self.evaluate(deepcopy(call), deepcopy(raw)))
                if not isinstance(evaluated, dict) or not isinstance(evaluated.get("status"), str):
                    raise TypeError("Evaluation must be a dictionary with a string status")
            except Exception as exc:
                evaluated = {"status": "evaluation_error", "error_type": type(exc).__name__}
            status = evaluated["status"]
        returned = raw.get("model") if isinstance(raw.get("model"), str) else None
        prior = self._models.get((call["provider"], call["model"]), set())
        drift = not model_matches(call["model"], returned) or bool(prior and prior != {returned})
        if drift:
            status = "model_drift"
        if cost > reservation:
            status = "budget_contract_failure"
        fatal = status in FATAL or bool(evaluated and evaluated.get("fatal") is True)
        return {**row, "status": status, "fatal": fatal, "cost_usd": usd(cost), "cost_known": known,
                "returned_model": returned, "model_drift": drift, "evaluated": evaluated}

    # ----------------------------------------------------------- missingness
    def _missing_row(self, slot, reason, dependency):
        if not isinstance(slot, str) or not slot or "#" in slot or not isinstance(reason, str) or not reason:
            raise ValueError("Invalid missing slot")
        if slot in self._slots:
            raise Halted("Slot already reserved or recorded missing")
        final = self.final(dependency) if isinstance(dependency, str) else None
        if final is None or final["status"] == "ok":
            raise ValueError("A missing slot needs a resolved, unsuccessful dependency")
        return {"slot": slot, "reason": reason, "dependency": dependency, "dependency_status": final["status"]}

    def missing(self, slot, reason, dependency):
        """Record an unpaid slot that cannot be called because its dependency failed."""
        with self.mutex:
            self._append("missing", self._missing_row(slot, reason, dependency))

    # -------------------------------------------------------------- accessors
    def spent(self) -> Decimal:
        """Returned-usage cost of resolved calls plus full reservations of open ones."""
        with self.mutex:
            return self._spent

    def unresolved(self):
        with self.mutex:
            return sorted(set(self._calls) - set(self._results))

    def fatal(self):
        with self.mutex:
            return sorted(self._fatal)

    def require_resolved(self):
        with self.mutex:
            if self.unresolved():
                raise Halted("Uncertain dispatched call; reconcile manually, never retry or resume")
            if self.fatal():
                raise Halted("Persisted contract failure; no new dispatch without a documented decision")

    def next_attempt(self, slot):
        """0 for a fresh slot, 1 for a licensed schema retry, else None."""
        with self.mutex:
            if slot in self._missing:
                return None
            ids = self._slots.get(slot, [])
            if not ids:
                return 0
            result = self._results.get(ids[-1])
            if result is None or len(ids) >= self.max_attempts or result["status"] != "schema_failure":
                return None
            return len(ids) if (result["evaluated"] or {}).get("retryable") is True else None

    def slots(self):
        with self.mutex:
            return list(self._slots)

    def attempts(self, slot):
        with self.mutex:
            return [_merged(self._calls[a], self._results.get(a)) for a in self._slots.get(slot, [])]

    def final(self, slot):
        with self.mutex:
            ids = self._slots.get(slot, [])
            if not ids or ids[-1] not in self._results:
                return None
            return _merged(self._calls[ids[-1]], self._results[ids[-1]])

    def missing_record(self, slot):
        with self.mutex:
            return deepcopy(self._missing.get(slot))

    def results(self, start=0):
        """Resolved attempts (reservation merged with result) in journal order."""
        with self.mutex:
            pairs = [(self._calls[a], self._results[a]) for a in self._log[start:]]
        for call, result in pairs:
            yield _merged(call, result)
