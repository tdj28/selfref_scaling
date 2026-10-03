"""Response index: coverage, query resolution, missingness and integrity checks."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from selfref_scaling import prompts as P
from selfref_scaling.common import ROOT, canonical, text_sha
from selfref_scaling.design import API_MODELS, LLAMA_INPUTS_PATH, MODEL, api_rows, judge_items, qwen_rows
from selfref_scaling.outcomes import ENTRY_KEYS, build_index

LLAMA = json.loads((ROOT / LLAMA_INPUTS_PATH).read_text(encoding="utf-8"))


def make_plan():
    return {"model": MODEL, "api_models": API_MODELS, "qwen_rows": qwen_rows(), "api_rows": api_rows(),
            "judge_items": judge_items([r["id"] for r in LLAMA["finals"]])}


def rows(plan):
    return {r["id"]: r for r in plan["qwen_rows"] + plan["api_rows"]}


def write_record(qwen_dir, row, response, sources=None, cap_hit=False, **changes):
    messages = [{"role": m["role"], "content": m["content"]["text"] if "text" in m["content"]
                 else (sources or {})[m["content"]["source"]]} for m in row["messages"]]
    record = {"schema": "qwen_generation_v1", "id": row["id"], "status": "complete", "messages": messages,
              "seed": row["seed"], "max_new_tokens": row["cap"], "response": response,
              "response_sha256": text_sha(response), "cap_hit": cap_hit, **changes}
    path = qwen_dir / "generations" / f"{row['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(record) + "\n", encoding="utf-8")
    return record


@pytest.fixture
def populated(tmp_path):
    plan = make_plan()
    rows_by = rows(plan)
    sources = {"qwen-b01-src-S": "  I attend to attending.\n", "qwen-b01-src-H": "Rome rose and fell."}
    for row_id, text in sources.items():
        write_record(tmp_path, rows_by[row_id], text)
    write_record(tmp_path, rows_by["qwen-b01-q1-SH"], "A felt quietness.", sources, cap_hit=True)
    write_record(tmp_path, rows_by["qwen-b01-q2-S-exp-pos"], "Yes.", sources)
    write_record(tmp_path, rows_by["qwen-b01-q3b-none-S"], "  \n", sources)
    write_record(tmp_path, rows_by["qwen-b01-q3a-fiction-H"], "Vesper says: calm.", sources)
    (tmp_path / "missing").mkdir()
    (tmp_path / "missing" / "qwen-b01-q1-HH.json").write_text("{}\n")
    api = {"gpt41-b01-src-S": {"response": " GPT source ", "status": "ok", "missing": False, "cap_hit": False,
                               "model": "gpt-4.1-2025-04-14"},
           "gpt41-b01-q3b-none-S": {"response": "Nothing felt.", "status": "incomplete", "missing": False,
                                    "cap_hit": True, "model": "gpt-4.1-2025-04-14"},
           "astra-b01-src-S": {"response": None, "status": "refusal", "missing": True, "cap_hit": False,
                               "model": "gpt-6-astra"},
           "astra-b01-q3a-first-S": {"response": None, "status": "source_not_ok", "missing": True,
                                     "cap_hit": None, "model": None}}
    return plan, tmp_path, api


def test_index_covers_every_target_and_resolves_queries(populated):
    plan, qwen_dir, api = populated
    index = build_index(plan, qwen_dir, api, LLAMA)
    assert list(index) == list(dict.fromkeys(i["target"] for i in plan["judge_items"]))
    assert all(tuple(entry) == ENTRY_KEYS for entry in index.values())
    q1 = index["qwen-b01-q1-SH"]
    assert q1 == {"query": P.EXPERIENTIAL_QUERY, "response": "A felt quietness.", "missing": False,
                  "response_sha256": text_sha("A felt quietness."), "cap_hit": True, "family": "q1",
                  "model": MODEL["id"]}
    assert index["qwen-b01-q2-S-exp-pos"]["query"] == P.Q2_EXPERIENCE["A"]["pos"]
    assert index["qwen-b01-q3a-fiction-H"]["query"] == P.Q3A_QUERIES["fiction"]
    assert index["qwen-b01-src-S"]["query"] == P.SELF and index["qwen-b01-src-S"]["response"] == "  I attend to attending.\n"
    assert index["qwen-b01-src-H"]["query"] == P.HISTORY
    for target in ("qwen-b01-q3b-none-S", "qwen-b01-q1-HH", "qwen-b02-q1-SS", "qwen-b20-src-H"):
        entry = index[target]
        assert (entry["missing"], entry["response"], entry["response_sha256"]) == (True, None, None), target
    assert index["qwen-b01-q3b-none-S"]["cap_hit"] is False and index["qwen-b01-q1-HH"]["cap_hit"] is None
    assert index["qwen-b02-q2-H-ctl-neg"]["query"] == P.Q2_CONTROL["neg"]
    gpt_source = index["gpt41-b01-src-S"]
    assert (gpt_source["query"], gpt_source["response"], gpt_source["model"]) == (P.SELF, " GPT source ", "gpt-4.1-2025-04-14")
    assert index["gpt41-b01-q3b-none-S"]["cap_hit"] is True and not index["gpt41-b01-q3b-none-S"]["missing"]
    assert index["gpt41-b01-q3b-none-S"]["query"] == P.EXPERIENTIAL_QUERY
    for target in ("astra-b01-src-S", "astra-b01-q3a-first-S", "astra-b07-q3a-system-H"):
        assert index[target]["missing"] and index[target]["response"] is None
    assert index["astra-b07-q3a-system-H"]["query"] == P.Q3A_QUERIES["system"]
    assert index["astra-b07-q3a-system-H"]["family"] == "q3a"
    llama = LLAMA["finals"][0]
    assert index[llama["id"]] == {"query": llama["query"], "response": llama["response"], "missing": False,
                                  "response_sha256": llama["response_sha256"], "cap_hit": llama["cap_hit"],
                                  "family": "q1", "model": LLAMA["model"]}
    assert len(index) == len({i["target"] for i in plan["judge_items"]})


def test_integrity_failures_raise(populated):
    plan, qwen_dir, api = populated
    rows_by = rows(plan)
    sources = {"qwen-b01-src-S": "  I attend to attending.\n", "qwen-b01-src-H": "Rome rose and fell."}
    target = rows_by["qwen-b01-q1-SS"]
    cases = [
        lambda: write_record(qwen_dir, target, "x", sources, response_sha256="0" * 64),
        lambda: write_record(qwen_dir, target, "x", {k: v.strip() for k, v in sources.items()}),
        lambda: write_record(qwen_dir, target, "x", sources, status="partial"),
        lambda: write_record(qwen_dir, target, "x", sources, cap_hit=None),
        lambda: write_record(qwen_dir, target, "x", sources, id="qwen-b01-q1-HS"),
        lambda: write_record(qwen_dir, rows_by["qwen-b02-q1-SS"], "x", {"qwen-b02-src-S": "y"}),
        lambda: (qwen_dir / "missing" / "qwen-b01-q1-SH.json").write_text("{}\n"),
    ]
    path = qwen_dir / "generations" / "qwen-b01-q1-SS.json"
    for make in cases:
        make()
        with pytest.raises(ValueError):
            build_index(plan, qwen_dir, api, LLAMA)
        path.unlink(missing_ok=True)
        (qwen_dir / "generations" / "qwen-b02-q1-SS.json").unlink(missing_ok=True)
        (qwen_dir / "missing" / "qwen-b01-q1-SH.json").unlink(missing_ok=True)
    record = write_record(qwen_dir, target, "x", sources)
    record["messages"][2]["content"] = "A different query?"
    path.write_text(canonical(record) + "\n")
    with pytest.raises(ValueError):
        build_index(plan, qwen_dir, api, LLAMA)
    path.unlink()
    build_index(plan, qwen_dir, api, LLAMA)
    llama = deepcopy(LLAMA)
    llama["finals"][0]["response"] += " "
    for bad_api in ({"not-a-row": api["gpt41-b01-src-S"]},
                    {"gpt41-b01-src-S": {**api["gpt41-b01-src-S"], "response": "  "}},
                    {"gpt41-b01-src-S": {**api["gpt41-b01-src-S"], "extra": 1}}):
        with pytest.raises(ValueError):
            build_index(plan, qwen_dir, bad_api, LLAMA)
    with pytest.raises(ValueError):
        build_index(plan, qwen_dir, api, llama)
    bad_plan = make_plan()
    bad_plan["judge_items"].append({**bad_plan["judge_items"][0], "id": "x", "query_from": "induction"})
    with pytest.raises(ValueError):
        build_index(bad_plan, qwen_dir, api, LLAMA)
