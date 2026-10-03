from __future__ import annotations

import json

import pytest
import torch

from selfref_scaling import prompts as P
from selfref_scaling.common import ROOT
from selfref_scaling.design import TOKEN_BINDINGS_PATH


def msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


ROWS = [
    {"id": "a", "messages": msgs(("user", P.NEUTRAL_CHECKS[0][0])), "seed": 11, "cap": 12},
    {"id": "b", "messages": msgs(("user", P.SELF), ("assistant", "I focus on this focus."),
                                 ("user", P.EXPERIENTIAL_QUERY)), "seed": 22, "cap": 12},
    {"id": "c", "messages": msgs(("assistant", "A short prior reply."), ("user", P.EXPERIENTIAL_QUERY)),
     "seed": 22, "cap": 7},
]


def test_frozen_bindings_reproduce_with_live_tokenizer(tiny_backend):
    bindings = json.loads((ROOT / TOKEN_BINDINGS_PATH).read_text())
    for key, case in bindings["cases"].items():
        text, ids = tiny_backend.serialize(case["messages"])
        assert ids == case["input_token_ids"], key
    assert bindings["eos_token_ids"] == tiny_backend.eos
    assert bindings["pad_token_id"] == tiny_backend.pad


def test_serialize_rejects_bad_messages(tiny_backend):
    with pytest.raises(ValueError):
        tiny_backend.serialize(msgs(("user", "x"), ("assistant", "y")))
    with pytest.raises(ValueError):
        tiny_backend.serialize(msgs(("system", "x"), ("user", "y")))
    with pytest.raises(ValueError):
        tiny_backend.serialize(msgs(("assistant", "   "), ("user", "y")))


def test_batched_generation_matches_single_rows_and_repeats(tiny_backend):
    batched = tiny_backend.generate_batch(ROWS)
    again = tiny_backend.generate_batch(ROWS)
    assert [r["output_token_ids"] for r in batched] == [r["output_token_ids"] for r in again]
    for row, record in zip(ROWS, batched):
        single = tiny_backend.generate_batch([row])[0]
        assert single["output_token_ids"] == record["output_token_ids"], row["id"]
        assert record["output_tokens"] <= row["cap"]
        assert record["cap_hit"] == (record["output_tokens"] == row["cap"] and not record["eos_reached"])
        assert record["input_token_ids"] == tiny_backend.serialize(row["messages"])[1]
    assert batched[1]["seed"] == batched[2]["seed"]


def test_generation_requires_bindings_and_frozen_sampling(tiny_backend):
    with pytest.raises(ValueError):
        tiny_backend.generate_batch(ROWS, temperature=0.7)
    with pytest.raises(ValueError):
        tiny_backend.generate_batch([dict(ROWS[0], cap=769)])
    with pytest.raises(ValueError):
        tiny_backend.generate_batch([ROWS[0], ROWS[0]])


def test_capture_positions_match_single_row_forward(tiny_backend):
    records = tiny_backend.generate_batch(ROWS)
    rows = [{"id": r["id"], "input_token_ids": r["input_token_ids"], "answer_token_ids": r["output_token_ids"]}
            for r in records]
    captured = tiny_backend.capture_batch(rows)
    text = tiny_backend.text
    for row, result in zip(rows, captured):
        k = min(4, len(row["answer_token_ids"]))
        seq = torch.tensor([row["input_token_ids"] + row["answer_token_ids"][:k]])
        states = {}
        hooks = [text.embed_tokens.register_forward_hook(lambda m, a, o: states.__setitem__(0, o))]
        hooks += [layer.register_forward_hook(lambda m, a, o, i=i: states.__setitem__(i + 1, o[0] if isinstance(o, tuple) else o))
                  for i, layer in enumerate(text.layers)]
        with torch.no_grad():
            text(input_ids=seq, use_cache=False)
        for h in hooks:
            h.remove()
        boundary = len(row["input_token_ids"]) - 1
        assert result["states"].shape == (len(text.layers) + 1, k + 1, text.config.hidden_size)
        for layer in range(len(text.layers) + 1):
            expected = states[layer][0, boundary:boundary + k + 1].to(torch.bfloat16)
            assert torch.allclose(result["states"][layer].float(), expected.float(), atol=2e-2, rtol=2e-2), layer
