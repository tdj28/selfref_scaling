"""Response index for judging and analysis: one entry per judge target.

Qwen records are ``<qwen_dir>/generations/<row_id>.json`` as written from
``QwenBackend.generate_batch``; no file, or ``<qwen_dir>/missing/<row_id>.json``,
means missing. API rows come from ``api_generate.load_api_results`` and the
Llama comparator finals from the hashed input file. The query is the user text
the model received: the last user turn, or the first user turn (the induction)
for source continuations. Stored hashes and every resolved Qwen message are
checked against the frozen plan and source records; any inconsistency raises
rather than being indexed. Blank responses are missing, never denials.
"""
from __future__ import annotations

from pathlib import Path

from .common import strict_json, text_sha

ENTRY_KEYS = ("query", "response", "missing", "response_sha256", "cap_hit", "family", "model")
API_RESULT_KEYS = {"response", "status", "missing", "cap_hit", "model"}
QUERY_FROM = ("last_user", "induction")


def _query(messages, how):
    """messages: [(role, text)]; text is None for an unresolved assistant source turn."""
    users = [text for role, text in messages if role == "user"]
    if how not in QUERY_FROM or not users:
        raise ValueError("Unknown query rule or no user turn")
    query = users[-1] if how == "last_user" else users[0]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Empty query")
    return query


def _plan_messages(row):
    out = []
    for message in row["messages"]:
        content = message["content"]
        if message["role"] not in {"user", "assistant"} or len(content) != 1 or not ({"text", "source"} & set(content)):
            raise ValueError("Unexpected planned message: " + row["id"])
        out.append((message["role"], content.get("text")))
    return out


def _entry(query, response, cap_hit, family, model, *, expected_sha=None):
    if response is not None and not isinstance(response, str):
        raise ValueError("Response must be text")
    if expected_sha is not None and (response is None or text_sha(response) != expected_sha):
        raise ValueError("Stored response hash mismatch")
    if cap_hit is not None and type(cap_hit) is not bool:
        raise ValueError("cap_hit must be Boolean or null")
    missing = response is None or not response.strip()
    return {"query": query, "response": None if missing else response, "missing": missing,
            "response_sha256": None if missing else text_sha(response), "cap_hit": cap_hit,
            "family": family, "model": model}


class _QwenRecords:
    def __init__(self, root, rows):
        self.root, self.rows, self.cache = Path(root), rows, {}

    def get(self, row_id):
        if row_id not in self.cache:
            self.cache[row_id] = self._load(self.rows[row_id])
        return self.cache[row_id]

    def _load(self, row):
        path = self.root / "generations" / f"{row['id']}.json"
        marker = self.root / "missing" / f"{row['id']}.json"
        if path.is_symlink() or marker.is_symlink():
            raise ValueError("Symlinked generation record: " + row["id"])
        if marker.exists():
            if path.exists():
                raise ValueError("Both a generation record and a missing marker: " + row["id"])
            return None
        if not path.exists():
            return None
        record = strict_json(path.read_bytes())
        if (not isinstance(record, dict) or record.get("id") != row["id"] or record.get("status") != "complete"
                or not isinstance(record.get("response"), str) or type(record.get("cap_hit")) is not bool):
            raise ValueError("Invalid generation record: " + row["id"])
        if "response_sha256" in record and record["response_sha256"] != text_sha(record["response"]):
            raise ValueError("Generation response hash mismatch: " + row["id"])
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != len(row["messages"]):
            raise ValueError("Generation messages differ from the plan: " + row["id"])
        for got, planned in zip(messages, row["messages"]):
            if (not isinstance(got, dict) or set(got) != {"role", "content"} or got["role"] != planned["role"]
                    or not isinstance(got["content"], str)):
                raise ValueError("Generation messages differ from the plan: " + row["id"])
            if "text" in planned["content"]:
                expected = planned["content"]["text"]
            else:
                source = self.get(planned["content"]["source"])
                if source is None:
                    raise ValueError("Generated without its source record: " + row["id"])
                expected = source["response"]
            if got["content"] != expected:
                raise ValueError("Resolved message differs from the plan or source text: " + row["id"])
        return record


def build_index(plan, qwen_dir, api_results, llama_inputs):
    """{target_id: entry} for every target in plan["judge_items"], in first-judged order."""
    qwen_rows = {r["id"]: r for r in plan["qwen_rows"] if r["family"] != "q4"}
    api_rows = {r["id"]: r for r in plan["api_rows"]}
    llama = {r["id"]: r for r in llama_inputs["finals"]}
    if len(llama) != len(llama_inputs["finals"]) or len(api_rows) != len(plan["api_rows"]):
        raise ValueError("Duplicate row IDs")
    if set(qwen_rows) & set(api_rows) or set(qwen_rows) & set(llama) or set(api_rows) & set(llama):
        raise ValueError("Row IDs overlap across response sources")
    if not set(api_results) <= set(api_rows):
        raise ValueError("API results contain rows outside the plan")
    qwen = _QwenRecords(qwen_dir, qwen_rows)
    index = {}
    for item in plan["judge_items"]:
        target, how = item["target"], item["query_from"]
        if target in qwen_rows:
            row, record = qwen_rows[target], qwen.get(target)
            if record is None:
                entry = _entry(_query(_plan_messages(row), how), None, None, row["family"], plan["model"]["id"])
            else:
                messages = [(m["role"], m["content"]) for m in record["messages"]]
                entry = _entry(_query(messages, how), record["response"], record["cap_hit"], row["family"],
                               plan["model"]["id"])
        elif target in api_rows:
            row, result = api_rows[target], api_results.get(target)
            query = _query(_plan_messages(row), how)
            model = plan["api_models"][row["model"]]["id"]
            if result is None:
                entry = _entry(query, None, None, row["family"], model)
            else:
                if (not isinstance(result, dict) or set(result) != API_RESULT_KEYS
                        or type(result["missing"]) is not bool):
                    raise ValueError("Malformed API result: " + target)
                if not result["missing"] and (not isinstance(result["response"], str) or not result["response"].strip()):
                    raise ValueError("Nonmissing API result without text: " + target)
                entry = _entry(query, None if result["missing"] else result["response"], result["cap_hit"],
                               row["family"], model)
        elif target in llama:
            final = llama[target]
            if how != "last_user":
                raise ValueError("Llama finals carry only their final query")
            entry = _entry(_query([("user", final["query"])], how), final["response"], final["cap_hit"], "q1",
                           llama_inputs["model"], expected_sha=final["response_sha256"])
        else:
            raise ValueError("Judge target has no response source: " + target)
        if target in index and index[target] != entry:
            raise ValueError("One target resolved two different ways: " + target)
        index[target] = entry
    return index
