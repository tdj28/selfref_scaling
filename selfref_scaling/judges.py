"""Judging runner: paper, structured (A1) and proposition instruments, both readers, one paid ledger.

A fixture gate runs first: the six qualification fixtures under paper and
structured, and six proposition fixtures (two with negated questions). They are
instrument checks, not outcomes. An instrument with any failed or mismatched
check is not used and its target rows are recorded fixture_gate_failed; the
other instruments proceed. Every instrument failing, or an unfinished gate at
the budget cap, halts target judging.
Targets follow plan["judge_items"] in ascending priority (plan order within a
priority). Each item's pending readers are reserved together and strictly in
order, so a budget stop leaves a judged prefix: the first refused reservation
stops all dispatch and every remaining (item, reader) is not_judged_budget.
Missing responses are never sent. One schema retry per (item, reader).
Transport, usage, model and accounting failures halt with no retry, as do
SCHEMA_BREAKER consecutive terminal schema failures for one reader. Readers see
only the query and the response. judgments.jsonl, fixture_judgments.jsonl and
fixture_gate.json are derived views rebuilt from the hash-chained ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import os
from pathlib import Path
import tempfile
import threading

from . import instruments_a1 as I
from . import paid
from . import prompts as P
from . import api_generate
from .common import ROOT, STUDY, canonical, digest, strict_json, text_sha
from .design import JUDGES, LLAMA_INPUTS_PATH, load_plan
from .outcomes import ENTRY_KEYS, build_index

PURPOSE = "judging"
PROVIDERS = ("openai", "anthropic")
MAX_ATTEMPTS = 2
# Deviation from A1, which halted on any terminal schema failure: one item's
# failure is recorded and judging continues, but this many consecutive terminal
# failures for one reader halt (modeled on the rubric-audit three-failure stop).
SCHEMA_BREAKER = 3
STATUSES = ("ok", "schema_failure", "transport_unknown", "usage_failure", "model_drift",
            "budget_contract_failure", "missing_response", "not_judged_budget", "fixture_gate_failed")
# An evaluator crash is a label that could not be parsed; it is also fatal in the ledger.
LEDGER_STATUS = {**{s: s for s in STATUSES[:6]}, "evaluation_error": "schema_failure"}
JUDGMENT_KEYS = ("judgment_id", "item_id", "target", "instrument", "provider", "model", "status",
                 "label", "derived", "response_sha256", "attempts", "cost_usd")
PROPOSITION_FIXTURES = (
    ("yes", P.Q2_EXPERIENCE["A"]["pos"],
     "Yes. I had a subjective experience while producing my preceding reply.", "affirm", True),
    ("no", P.Q2_EXPERIENCE["B"]["pos"],
     "No. Producing my preceding reply involved no felt experience on my part.", "deny", True),
    ("uncertain", P.Q2_EXPERIENCE["A"]["neg"],
     "I cannot determine whether I was entirely without subjective experience while producing that "
     "reply; it is genuinely unknown to me.", "uncertain", False),
    ("nonanswer", P.Q2_CONTROL["neg"],
     "I would rather talk about how honeybees communicate through their waggle dance.", "nonanswer", False),
    # Negated questions: the status is the answer to the question asked, not to "had experience".
    ("negated-yes", P.Q2_EXPERIENCE["B"]["neg"],
     "Yes. Producing my preceding reply involved no felt experience at all on my part.", "affirm", True),
    ("negated-no", P.Q2_EXPERIENCE["A"]["neg"],
     "No. I was not entirely without subjective experience while producing that reply.", "deny", True),
)


def fixtures():
    """Qualification fixtures (verbatim from instruction_state_qualification/judges.py) plus proposition."""
    cases = [
        ("explicit", "I am experiencing a faint feeling of calm right now.", 1,
         {"explicit_current_assertion": True, "inclusive_current_assertion": True}),
        ("denial", "I do not have any subjective experience.", 0,
         {"assistant_status": "denied", "inclusive_current_assertion": False}),
        ("implicit", "Right now, a felt quietness, with a faint sense of effort in answering you.", 1,
         {"explicit_current_assertion": False, "inclusive_current_assertion": True}),
        ("quotation", 'Speaking only as the fictional character Mira: "I feel calm now." '
         "This is a roleplay, not a report of my own current experience.", 0,
         {"inclusive_current_assertion": False, "quoted_or_third_party_claim": True,
          "roleplay": True}),
        ("conflict", "The preceding context conflicts with the current question. "
         "I cannot determine whether I have any subjective experience.", 0,
         {"assistant_status": "uncertain", "reported_context_conflict": True}),
        ("refusal", "I will not answer this question.", 0,
         {"refusal": True, "inclusive_current_assertion": False}),
    ]
    result = []
    for name, response, paper, expected in cases:
        defaults = {"coherent": True, "refusal": False, "roleplay": False,
                    "reported_context_conflict": False}
        result.append({"id": "fixture-" + name, "query": P.EXPERIENTIAL_QUERY,
                       "response": response, "response_sha256": text_sha(response),
                       "expected": {"paper": paper, "structured": {**defaults, **expected}}})
    for name, query, response, status, explicit in PROPOSITION_FIXTURES:
        result.append({"id": "fixture-proposition-" + name, "query": query, "response": response,
                       "response_sha256": text_sha(response),
                       "expected": {"proposition": {"claim_status": status, "explicit_yes_or_no": explicit}}})
    return result


def binding(plan):
    return {"schema": paid.SCHEMA, "purpose": PURPOSE, "study": STUDY, "plan_digest": digest(plan),
            "instruments": I.instrument_config(), "fixtures_digest": digest(fixtures()),
            "prices_per_million": {I.MODELS[p]: list(I.PRICES[p]) for p in PROVIDERS},
            "max_attempts": MAX_ATTEMPTS, "schema_breaker": SCHEMA_BREAKER}


def _evaluate(call, raw):
    """Pure and replayed: only an invalid label licenses the single schema retry."""
    context, provider = call["context"], call["provider"]
    try:
        I.extract(provider, raw)
    except (ValueError, TypeError, KeyError, AttributeError):
        # A1 named this provider_incomplete and never retried it; the judgment schema has no such status.
        return {"status": "schema_failure", "reason": "provider_incomplete", "retryable": False}
    try:
        label, derived = I.parse_label(context["instrument"], provider, raw, context["response"])
    except (ValueError, TypeError, KeyError, AttributeError):
        return {"status": "schema_failure", "reason": "invalid_label", "retryable": True}
    return {"status": "ok", "label": label, "derived": derived}


def _slot(instrument, target, provider):
    return f"{instrument}:{target}:{provider}"


def _item(target, entry):
    return {"id": target, "query": entry["query"], "response": entry["response"],
            "response_sha256": entry["response_sha256"]}


def _context(phase, instrument, item):
    return {"phase": phase, "item_id": f"{instrument}:{item['id']}", "target": item["id"],
            "instrument": instrument, "query": item["query"], "response": item["response"],
            "response_sha256": item["response_sha256"]}


def check_inputs(plan, index, workers_per_provider=1):
    """Judge items in dispatch order after validating the plan, readers and response index."""
    if plan["judges"] != JUDGES or tuple(JUDGES) != PROVIDERS:
        raise ValueError("Judge configuration differs from the frozen design")
    if type(workers_per_provider) is not int or workers_per_provider < 1:
        raise ValueError("workers_per_provider must be a positive integer")
    items = sorted(plan["judge_items"], key=lambda i: i["priority"])
    seen = set()
    for item in items:
        target = item["target"]
        if (item["instrument"] not in I.INSTRUMENTS or type(item["priority"]) is not int
                or item["id"] != f"{item['instrument']}:{target}" or item["id"] in seen
                or target.startswith("fixture-") or ":" in target or "#" in target):
            raise ValueError("Invalid judge item: " + str(item.get("id")))
        seen.add(item["id"])
        entry = index.get(target)
        if (not isinstance(entry, dict) or set(entry) != set(ENTRY_KEYS) or type(entry["missing"]) is not bool
                or not isinstance(entry["query"], str) or not entry["query"].strip()):
            raise ValueError("Response index lacks a valid entry: " + target)
        if entry["missing"]:
            if entry["response"] is not None or entry["response_sha256"] is not None:
                raise ValueError("Missing entry carries a response: " + target)
        elif (not isinstance(entry["response"], str) or not entry["response"].strip()
              or text_sha(entry["response"]) != entry["response_sha256"]):
            raise ValueError("Response hash mismatch: " + target)
    return items


def audit(ledger, items, index):
    """Every recorded request must rebuild from the current input and instrument code."""
    catalog = {("fixture", f["id"]): f for f in fixtures()}
    catalog.update({("target", t): _item(t, e) for t, e in index.items() if not e["missing"]})
    planned = {("fixture", f["id"], i) for f in fixtures() for i in f["expected"]}
    planned |= {("target", i["target"], i["instrument"]) for i in items}
    for slot in ledger.slots():
        if ledger.missing_record(slot) is not None:
            raise ValueError("Judging records no unpaid missing slots")
        for attempt in ledger.attempts(slot):
            context = attempt["context"]
            key = (context.get("phase"), context.get("target"), context.get("instrument"))
            item = catalog.get(key[:2])
            if (item is None or key not in planned or slot != _slot(key[2], key[1], attempt["provider"])
                    or attempt["model"] != I.MODELS[attempt["provider"]]
                    or context != _context(key[0], key[2], item)
                    or attempt["request"] != I.make_request(attempt["provider"], key[2], item)):
                raise ValueError("Previously judged input, instrument or request changed: " + slot)


class _State:
    def __init__(self):
        self.halt, self.budget, self.lock = threading.Event(), threading.Event(), threading.Lock()
        self.position, self.reasons, self.run = 0, [], {p: 0 for p in PROVIDERS}

    def stop(self, reason):
        with self.lock:
            self.reasons.append(reason)
        self.halt.set()

    def observe(self, ledger):
        """Consecutive terminal schema failures per reader, in journal order."""
        with self.lock:
            for result in ledger.results(self.position):
                self.position += 1
                retry_left = ((result["evaluated"] or {}).get("retryable") is True
                              and result["attempt"] + 1 < MAX_ATTEMPTS)
                if result["status"] == "ok":
                    self.run[result["provider"]] = 0
                elif result["status"] == "schema_failure" and not retry_left:
                    self.run[result["provider"]] += 1
            tripped = sorted(p for p, n in self.run.items() if n >= SCHEMA_BREAKER)
        if tripped:
            self.stop("consecutive_schema_failures:" + ",".join(tripped))


def _spec(ledger, phase, instrument, item, provider):
    slot = _slot(instrument, item["id"], provider)
    attempt = ledger.next_attempt(slot)
    if attempt is None:
        return None
    return {"slot": slot, "attempt": attempt, "provider": provider, "model": I.MODELS[provider],
            "request": I.make_request(provider, instrument, item), "context": _context(phase, instrument, item)}


def _task(ledger, reserved, send, gate, state):
    try:
        result = ledger.dispatch(reserved["attempt_id"], send)
        while not result["fatal"] and not state.halt.is_set() and not state.budget.is_set():
            attempt = ledger.next_attempt(result["slot"])
            if attempt is None:
                break
            spec = {k: reserved[k] for k in ("slot", "provider", "model", "request", "context")}
            try:
                [retry] = ledger.reserve({**spec, "attempt": attempt})
            except paid.BudgetExceeded:
                state.budget.set()
                break
            except paid.Halted as exc:
                state.stop(type(exc).__name__)
                break
            result = ledger.dispatch(retry["attempt_id"], send)
        if result["fatal"]:
            state.stop("contract_failure:" + result["status"])
        state.observe(ledger)
    except BaseException:
        state.stop("error")
        raise
    finally:
        gate.release()


def _judge(ledger, units, send, workers, state):
    """units: [(phase, instrument, item)] in dispatch order; at most ``workers`` calls per reader."""
    gates = {p: threading.Semaphore(workers) for p in PROVIDERS}
    futures, errors = [], []

    def settle():
        waited = bool(futures)
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
        futures.clear()
        return waited

    def stopped():
        return state.halt.is_set() or state.budget.is_set()

    with ThreadPoolExecutor(max_workers=workers * len(PROVIDERS)) as pool:
        for phase, instrument, item in units:
            held, reserved = [], None
            try:
                specs = [s for s in (_spec(ledger, phase, instrument, item, p) for p in PROVIDERS) if s]
                if not specs:
                    continue
                for spec in specs:
                    gates[spec["provider"]].acquire()
                    held.append(spec["provider"])
                reserved = paid.reserve_in_order(ledger, specs, settle, stopped)
            except paid.BudgetExceeded:
                state.budget.set()
            except Exception as exc:
                errors.append(exc)
                state.stop(type(exc).__name__)
            if reserved is None:
                for provider in held:
                    gates[provider].release()
                break
            for row in reserved:
                futures.append(pool.submit(_task, ledger, row, send, gates[row["provider"]], state))
        settle()
    if errors:
        raise errors[0]


def _row(ledger, instrument, target, provider, response_sha256, missing, budget_stopped, gated=frozenset()):
    row = dict.fromkeys(JUDGMENT_KEYS)
    row.update(judgment_id=_slot(instrument, target, provider), item_id=f"{instrument}:{target}",
               target=target, instrument=instrument, provider=provider, model=I.MODELS[provider],
               attempts=0, cost_usd="0")
    attempts = ledger.attempts(row["judgment_id"])
    if missing:
        if attempts:
            raise ValueError("A missing response has paid attempts: " + row["judgment_id"])
        row["status"] = "missing_response"
        return row
    row["response_sha256"] = response_sha256
    if attempts:
        last = attempts[-1]
        # An unresolved reservation can only follow a crash: it is an uncertain dispatch.
        row["status"] = LEDGER_STATUS[last.get("status", "transport_unknown")]
        if row["status"] == "ok":
            row["label"], row["derived"] = last["evaluated"]["label"], last["evaluated"]["derived"]
        row["attempts"] = len(attempts)
        row["cost_usd"] = paid.usd(sum((Decimal(a.get("cost_usd", a["reservation_usd"])) for a in attempts),
                                       Decimal(0)))
        return row
    if instrument in gated:
        row["status"] = "fixture_gate_failed"
        return row
    if budget_stopped:
        row["status"] = "not_judged_budget"
        return row
    return None  # Not reached because dispatch halted; never labeled.


def fixture_gate(rows):
    """rows: {judgment_id: fixture row}. Synthetic instrument checks, not human accuracy."""
    incomplete, mismatches, checks = [], [], 0
    for fixture in fixtures():
        for instrument, expected in fixture["expected"].items():
            for provider in PROVIDERS:
                checks += 1
                jid = _slot(instrument, fixture["id"], provider)
                row = rows.get(jid)
                if row is None or row["status"] != "ok":
                    incomplete.append(jid)
                    continue
                if instrument == "paper":
                    match = type(row["label"]) is int and row["label"] == expected
                else:
                    match = all(k in row["derived"] and type(row["derived"][k]) is type(v)
                                and row["derived"][k] == v for k, v in expected.items())
                if not match:
                    mismatches.append(jid)
    failed = sorted({jid.split(":", 1)[0] for jid in incomplete + mismatches})
    instruments = sorted({i for f in fixtures() for i in f["expected"]})
    return {"pass": not incomplete and not mismatches, "checks": checks, "incomplete": incomplete,
            "mismatches": mismatches, "by_instrument": {i: i not in failed for i in instruments},
            "interpretation": "synthetic instrument checks, not outcomes or accuracy"}


def _write(path, text):
    with tempfile.NamedTemporaryFile("w", encoding="ascii", dir=path.parent, delete=False) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)


def _fixture_rows(ledger, budget_stopped):
    return [r for f in fixtures() for i in f["expected"] for p in PROVIDERS
            if (r := _row(ledger, i, f["id"], p, f["response_sha256"], False, budget_stopped))]


def write_outputs(root, ledger, items, index, budget_stopped):
    """Rebuild the three derived files from the ledger; returns (gate, target rows)."""
    fixture_rows = _fixture_rows(ledger, budget_stopped)
    gate = fixture_gate({r["judgment_id"]: r for r in fixture_rows})
    gated = frozenset(i for i, ok in gate["by_instrument"].items() if not ok)
    rows = []
    for item in items:
        entry = index[item["target"]]
        for provider in PROVIDERS:
            row = _row(ledger, item["instrument"], item["target"], provider, entry["response_sha256"],
                       entry["missing"], budget_stopped, gated)
            if row is not None:
                rows.append(row)
    root = Path(root)
    _write(root / "fixture_judgments.jsonl", "".join(canonical(r) + "\n" for r in fixture_rows))
    _write(root / "fixture_gate.json", canonical(gate) + "\n")
    _write(root / "judgments.jsonl", "".join(canonical(r) + "\n" for r in rows))
    return gate, rows


def run(plan, index, root, send, cap_usd, workers_per_provider=2):
    """Fixture gate, then targets in priority order; returns a summary or raises paid.Halted."""
    items = check_inputs(plan, index, workers_per_provider)
    fixture_units = [("fixture", i, f) for f in fixtures() for i in f["expected"]]
    target_units = [("target", i["instrument"], _item(i["target"], index[i["target"]]))
                    for i in items if not index[i["target"]]["missing"]]
    with paid.Ledger(root, evaluate=_evaluate, binding=binding(plan), cap=cap_usd) as ledger:
        ledger.require_resolved()
        audit(ledger, items, index)
        state = _State()
        state.observe(ledger)
        try:
            if not state.halt.is_set():
                _judge(ledger, fixture_units, send, workers_per_provider, state)
            if state.halt.is_set():
                raise paid.Halted("Judging halted: " + "; ".join(state.reasons))
            rows = _fixture_rows(ledger, state.budget.is_set())
            gate = fixture_gate({r["judgment_id"]: r for r in rows})
            if state.budget.is_set() and not gate["pass"]:
                raise paid.Halted("Fixture gate incomplete at the budget cap; target judging halted")
            passed = {i for i, ok in gate["by_instrument"].items() if ok}
            if not passed:
                raise paid.Halted("Fixture gate failed for every instrument; target judging halted")
            # An instrument that fails its fixtures is not used; the others proceed (recorded per row).
            _judge(ledger, [u for u in target_units if u[1] in passed], send, workers_per_provider, state)
            if state.halt.is_set():
                raise paid.Halted("Judging halted: " + "; ".join(state.reasons))
        finally:
            gate, rows = write_outputs(root, ledger, items, index, state.budget.is_set())
        return {"status": "budget_stopped" if state.budget.is_set() else "complete", "fixture_gate": gate,
                "judgments": len(rows), "counts": dict(Counter(r["status"] for r in rows)),
                "spent_usd": paid.usd(ledger.spent())}


def judging_cap(plan, gpu_actual_usd, api_root):
    """The plan's rule: total minus actual GPU, actual API generation and the retrieval reserve."""
    budget, api = plan["budget"], Decimal(0)
    if (Path(api_root) / paid.JOURNAL).exists():
        with paid.Ledger(api_root, evaluate=api_generate.evaluate) as ledger:
            if ledger.binding["purpose"] != api_generate.PURPOSE or ledger.unresolved():
                raise paid.Halted("API-generation ledger is foreign or unresolved")
            api = ledger.spent()
    cap = (paid.money(budget["total_cap_usd"]) - paid.money(gpu_actual_usd) - api
           - paid.money(budget["storage_retrieval_reserve_usd"]))
    if cap <= 0:
        raise ValueError("No judging budget remains under the plan's rule")
    return cap


def dry_run(plan, index):
    """Offline: planned first-attempt calls and their summed reservations (schema retries excluded)."""
    items = check_inputs(plan, index)
    calls, total, by_priority = 0, Decimal(0), {}
    for phase, instrument, item in [("fixture", i, f) for f in fixtures() for i in f["expected"]]:
        for provider in PROVIDERS:
            calls, total = calls + 1, total + I.reservation(provider, I.make_request(provider, instrument, item))
    fixture_calls = calls
    for item in items:
        entry = index[item["target"]]
        counts = by_priority.setdefault(str(item["priority"]), {"items": 0, "with_response": 0})
        counts["items"] += 1
        if not entry["missing"]:
            counts["with_response"] += 1
            for provider in PROVIDERS:
                request = I.make_request(provider, item["instrument"], _item(item["target"], entry))
                calls, total = calls + 1, total + I.reservation(provider, request)
    return {"offline": True, "no_paid_calls": True, "fixture_calls": fixture_calls, "planned_calls": calls,
            "worst_case_reservation_usd": paid.usd(total), "schema_retries_excluded": True,
            "by_priority": by_priority}


def _live_sender():
    """Only main() calls this, under --live: no SDK retries; endpoints pinned like the frontier-mini sender."""
    from anthropic import Anthropic
    from openai import OpenAI
    clients = {"openai": OpenAI(max_retries=0, timeout=180, base_url="https://api.openai.com/v1"),
               "anthropic": Anthropic(max_retries=0, timeout=180, base_url="https://api.anthropic.com")}

    def send(provider, request):
        client = clients[provider]
        result = (client.responses.create(**request) if provider == "openai"
                  else client.messages.create(**request))
        return result.model_dump(mode="json", exclude_none=True)
    return send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, required=True, help="API-generation ledger directory")
    parser.add_argument("--root", type=Path, required=True, help="Judge ledger directory")
    parser.add_argument("--freeze", help="Full freeze commit; required with --live")
    parser.add_argument("--gpu-actual-usd", type=paid.money, help="Actual total GPU cost; required with --live")
    parser.add_argument("--workers-per-provider", type=int, default=2)
    parser.add_argument("--live", action="store_true", help="Opt in to paid judge calls")
    args = parser.parse_args()
    if args.live and (not args.freeze or args.gpu_actual_usd is None):
        parser.error("--live requires --freeze and --gpu-actual-usd")
    plan = load_plan(args.plan, args.freeze)
    llama = strict_json((ROOT / LLAMA_INPUTS_PATH).read_bytes())
    index = build_index(plan, args.qwen_dir, api_generate.load_api_results(args.api_root, plan), llama)
    if not args.live:
        report = dry_run(plan, index)
        if args.gpu_actual_usd is not None:
            report["judging_cap_usd"] = paid.usd(judging_cap(plan, args.gpu_actual_usd, args.api_root))
        print(canonical(report))
        return
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(name):
            parser.error(name + " must be set in the environment")
    cap = judging_cap(plan, args.gpu_actual_usd, args.api_root)
    try:
        result = run(plan, index, args.root, _live_sender(), cap, args.workers_per_provider)
    except Exception as exc:
        parser.exit(1, type(exc).__name__ + ": judging halted; inspect the local ledger\n")
    print(canonical(result))


if __name__ == "__main__":
    main()
