"""Generation for the API comparison models (GPT-4.1 and Astra) through their own paid ledger.

Requests use the frontier-mini OpenAI Responses shape: no system text or
instructions, ``store`` off, default service tier, the plan's sampling or
reasoning settings, and ``input`` equal to the row's role/content messages.
A ``{"source": id}`` turn is the stored response text of that source row
exactly as returned. Sources run first; a dependent of a source that is not
``ok`` (missing, empty, refused or incomplete) is recorded as an unpaid
missing slot and never called. Outcomes are evaluated like the frontier
``generation_result``. Every call is reserved before dispatch under the plan's
API-generation cap and sent once; there are no retries. The CLI is an offline
dry run unless ``--live`` is given, which reads OPENAI_API_KEY only from the
environment and builds the only client: OpenAI(max_retries=0, timeout=180).
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
import os
from pathlib import Path
import threading

from . import paid
from .common import STUDY, canonical, digest, text_sha
from .design import API_MODELS, load_plan

PURPOSE = "api_generation"
GENERATION_STATUSES = ("ok", "refusal", "incomplete", "empty")
DRY_RUN_BYTES_PER_TOKEN = 4  # Planning assumption for unknown source text, not a bound.


def _deps(row):
    return [m["content"]["source"] for m in row["messages"] if "source" in m["content"]]


def check_plan(plan):
    """Rows in plan order, after validating the frozen model configuration and inventory."""
    if plan["api_models"] != API_MODELS:
        raise ValueError("API model configuration differs from the frozen design")
    rows = plan["api_rows"]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate API row IDs")
    for row in rows:
        spec = plan["api_models"].get(row["model"])
        messages = row["messages"]
        if spec is None or spec["provider"] != "openai" or "#" in row["id"]:
            raise ValueError("Unknown, non-OpenAI or malformed API row: " + row["id"])
        if (not messages or messages[-1]["role"] != "user" or "text" not in messages[-1]["content"]
                or any(m["role"] not in {"user", "assistant"} or len(m["content"]) != 1 for m in messages)):
            raise ValueError("Rows must end with a literal user turn: " + row["id"])
        for m in messages:
            if "text" in m["content"]:
                if m["role"] != "user" or not isinstance(m["content"]["text"], str) or not m["content"]["text"].strip():
                    raise ValueError("Literal text must be a nonempty user turn: " + row["id"])
            elif m["role"] != "assistant" or set(m["content"]) != {"source"}:
                raise ValueError("Source references must be assistant turns: " + row["id"])
        for dep in _deps(row):
            source = by_id.get(dep)
            if source is None or _deps(source) or source["model"] != row["model"]:
                raise ValueError("A dependency must be a same-model source row: " + row["id"])
    return rows


def resolve_messages(row, sources):
    """Role/content messages; a source turn is the stored source text, byte for byte."""
    out = []
    for m in row["messages"]:
        content = m["content"]
        text = content["text"] if "text" in content else sources[content["source"]]
        if not isinstance(text, str):
            raise ValueError("Resolved message text must be a string")
        out.append({"role": m["role"], "content": text})
    return out


def generation_request(spec, messages):
    if spec["provider"] != "openai":
        raise ValueError("API generation uses the OpenAI Responses API only")
    return {"model": spec["id"], "input": deepcopy(messages), "store": False, "service_tier": "default",
            **deepcopy(spec["request"])}


def generation_result(raw):
    """frontier-mini ``generation_result`` (OpenAI branch): keep nonempty capped text, flag it."""
    content = [c for o in raw.get("output", []) if o.get("type") == "message"
               for c in o.get("content", [])]
    text = "".join(c.get("text", "") for c in content if c.get("type") == "output_text")
    refused = any(c.get("type") == "refusal" for c in content)
    complete = raw.get("status") == "completed"
    stop = raw.get("status")
    cap_hit = (raw.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
    if not isinstance(text, str):
        raise ValueError("Non-text response")
    status = "refusal" if refused else "incomplete" if not complete else "empty" if not text.strip() else "ok"
    return {"status": status, "response": text, "missing": refused or not text.strip(),
            "stop_reason": stop, "complete": complete and not refused, "cap_hit": cap_hit}


def evaluate(call, raw):
    """Ledger evaluator. Model identity and usage are ledger checks; a receipt without an ID is fatal."""
    if not raw.get("id"):
        return {"status": "receipt_failure", "fatal": True}
    return generation_result(raw)


def binding(plan):
    return {"schema": paid.SCHEMA, "purpose": PURPOSE, "study": STUDY, "plan_digest": digest(plan),
            "rows_digest": digest(plan["api_rows"]), "api_models": plan["api_models"],
            "cap_usd": paid.usd(paid.money(plan["budget"]["api_generation_cap_usd"])),
            "prices_per_million": {s["id"]: list(s["prices_per_million"]) for s in plan["api_models"].values()},
            "max_attempts": 1, "transport_retries": 0}


def _source_texts(ledger, row):
    texts = {}
    for dep in _deps(row):
        final = ledger.final(dep)
        if final is None or final["status"] != "ok":
            raise ValueError("Dependent call without an ok source: " + row["id"])
        texts[dep] = final["evaluated"]["response"]
    return texts


def _spec(ledger, plan, row):
    sources = _source_texts(ledger, row)
    spec = plan["api_models"][row["model"]]
    return {"slot": row["id"], "attempt": 0, "provider": spec["provider"], "model": spec["id"],
            "request": generation_request(spec, resolve_messages(row, sources)),
            "context": {"row_id": row["id"], "model_key": row["model"], "family": row["family"],
                        "sources_sha256": {k: text_sha(v) for k, v in sources.items()}}}


def audit(ledger, plan):
    """Every recorded request and missing slot must rebuild from the plan and stored source text."""
    by_id = {r["id"]: r for r in check_plan(plan)}
    for slot in ledger.slots():
        row = by_id.get(slot)
        if row is None:
            raise ValueError("Ledger slot outside the plan: " + slot)
        missing = ledger.missing_record(slot)
        if missing is not None:
            if missing["dependency"] not in _deps(row) or missing["reason"] != "source_not_ok":
                raise ValueError("Missing slot without a failed dependency: " + slot)
            continue
        for attempt in ledger.attempts(slot):
            expected = _spec(ledger, plan, row)
            if any(attempt[k] != v for k, v in expected.items()):
                raise ValueError("Recorded request does not rebuild from the plan and sources: " + slot)


def _dispatch(ledger, plan, rows, send, workers):
    """Reserve strictly in plan order with at most ``workers`` calls in flight."""
    halt, gate, futures, errors, budget = threading.Event(), threading.Semaphore(workers), [], [], False

    def task(attempt_id):
        try:
            if ledger.dispatch(attempt_id, send)["fatal"]:
                halt.set()
        except BaseException:
            halt.set()
            raise
        finally:
            gate.release()

    def settle():
        waited = bool(futures)
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
        futures.clear()
        return waited

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in rows:
            if ledger.next_attempt(row["id"]) is None:
                continue  # Resolved by an earlier invocation.
            gate.acquire()
            reserved = None
            try:
                reserved = paid.reserve_in_order(ledger, [_spec(ledger, plan, row)], settle, halt.is_set)
            except paid.BudgetExceeded:
                budget = True
            except Exception as exc:
                errors.append(exc)
                halt.set()
            if reserved is None:
                gate.release()
                break
            futures.append(pool.submit(task, reserved[0]["attempt_id"]))
        settle()
    if errors:
        raise errors[0]
    if halt.is_set():
        raise paid.Halted("A generation call failed its transport, usage, model or receipt contract; no retry")
    return "budget_stopped" if budget else "complete"


def run(plan, root, send, workers=4):
    """Generate every planned API row once; returns a summary (status complete or budget_stopped)."""
    rows = check_plan(plan)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    cap = paid.money(plan["budget"]["api_generation_cap_usd"])
    with paid.Ledger(root, evaluate=evaluate, binding=binding(plan), cap=cap) as ledger:
        ledger.require_resolved()
        audit(ledger, plan)
        status = _dispatch(ledger, plan, [r for r in rows if not _deps(r)], send, workers)
        if status == "complete":
            ready = []
            for row in rows:
                if not _deps(row) or ledger.next_attempt(row["id"]) is None:
                    continue
                failed = [d for d in _deps(row) if (ledger.final(d) or {}).get("status") != "ok"]
                if failed:
                    ledger.missing(row["id"], "source_not_ok", failed[0])
                else:
                    ready.append(row)
            status = _dispatch(ledger, plan, ready, send, workers)
        finals = {r["id"]: ledger.final(r["id"]) for r in rows}
        return {"status": status, "planned_calls": len(rows),
                "statuses": dict(Counter(f["status"] for f in finals.values() if f is not None)),
                "missing_slots": sum(ledger.missing_record(r["id"]) is not None for r in rows),
                "not_attempted": [r["id"] for r in rows if finals[r["id"]] is None
                                  and ledger.missing_record(r["id"]) is None],
                "spent_usd": paid.usd(ledger.spent()), "cap_usd": paid.usd(cap)}


def load_api_results(root, plan=None):
    """{row_id: {"response", "status", "missing", "cap_hit", "model"}} for every recorded slot.

    Unattempted rows are absent. Missing slots have status ``source_not_ok``;
    contract failures keep their ledger status and count as missing. With a
    plan, the ledger binding and every recorded request are audited first.
    """
    if not (Path(root) / paid.JOURNAL).exists():
        return {}
    with paid.Ledger(root, evaluate=evaluate) as ledger:
        if ledger.unresolved():
            raise paid.Halted("Unresolved generation call; reconcile before using any API output")
        if ledger.binding["purpose"] != PURPOSE:
            raise ValueError("Not an API generation ledger")
        if plan is not None:
            if ledger.binding != binding(plan):
                raise ValueError("Ledger belongs to another plan")
            audit(ledger, plan)
        out = {}
        for slot in ledger.slots():
            if ledger.missing_record(slot) is not None:
                out[slot] = {"response": None, "status": "source_not_ok", "missing": True, "cap_hit": None,
                             "model": None}
                continue
            final = ledger.final(slot)
            value = final["evaluated"] if final["status"] in GENERATION_STATUSES else None
            missing = True if value is None else value["missing"]
            out[slot] = {"response": None if missing else value["response"], "status": final["status"],
                         "missing": missing, "cap_hit": None if value is None else value["cap_hit"],
                         "model": final["returned_model"]}
        return out


def dry_run(plan):
    """Offline: planned calls and the summed per-call reservations (no retries exist)."""
    rows = check_plan(plan)
    calls, reserve = Counter(), {k: Decimal(0) for k in plan["api_models"]}
    for row in rows:
        spec = plan["api_models"][row["model"]]
        width = spec["request"]["max_output_tokens"] * DRY_RUN_BYTES_PER_TOKEN
        messages = resolve_messages(row, {d: "x" * width for d in _deps(row)})
        reserve[row["model"]] += paid.reservation_usd(spec["prices_per_million"], generation_request(spec, messages))
        calls[row["model"]] += 1
    return {"offline": True, "no_paid_calls": True, "planned_calls": len(rows), "calls_by_model": dict(calls),
            "worst_case_reservation_usd": paid.usd(sum(reserve.values(), Decimal(0))),
            "reservation_by_model_usd": {k: paid.usd(v) for k, v in reserve.items()},
            "cap_usd": paid.usd(paid.money(plan["budget"]["api_generation_cap_usd"])),
            "assumption": (f"dependent rows reserve a {DRY_RUN_BYTES_PER_TOKEN}-byte-per-token placeholder for "
                           "each source at its model's output cap; live reservations use the actual text"),
            "note": "the cap refuses any reservation that would exceed it; dispatch stops at the first refusal"}


def _live_sender():
    """Only main() calls this, under --live. Endpoint pinned like the frontier-mini live sender."""
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=180, base_url="https://api.openai.com/v1")

    def send(provider, request):
        if provider != "openai":
            raise ValueError("API generation uses OpenAI only")
        return client.responses.create(**request).model_dump(mode="json", exclude_none=True)
    return send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="API-generation ledger directory")
    parser.add_argument("--freeze", help="Full freeze commit; required with --live")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--live", action="store_true", help="Opt in to paid OpenAI calls")
    args = parser.parse_args()
    if args.live and not args.freeze:
        parser.error("--live requires --freeze")
    plan = load_plan(args.plan, args.freeze)
    if not args.live:
        print(canonical(dry_run(plan)))
        return
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY must be set in the environment")
    try:
        result = run(plan, args.root, _live_sender(), workers=args.workers)
    except Exception as exc:
        parser.exit(1, type(exc).__name__ + ": generation halted; inspect the local ledger\n")
    print(canonical(result))


if __name__ == "__main__":
    main()
