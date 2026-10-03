"""Amendment A1 CUDA qualification (engineering only; see docs/AMENDMENT_A1_20261003.md).

The frozen tests/test_cuda_smoke.py loaded its tiny model with ``device_map="auto"``
and 60 MiB per GPU; accelerate then placed a module on "disk", and the test's own
no-offload assertion failed at fixture setup, so the CUDA path was not exercised.
This replacement uses an explicit two-GPU map (no CPU or disk placement is
possible) and otherwise performs the same checks: repeat determinism, batched
versus single-row prefill, capture, plus the expert-kernel dispatch recorded in
a JSON receipt. It never runs Qwen3.5-397B or any study prompt as an outcome.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from selfref_scaling import prompts as P
from tests.conftest import snapshot, tiny_config, tokenizer  # noqa: F401  (fixtures)

ENOUGH_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
if os.environ.get("SELFREF_REQUIRE_CUDA") == "1" and not ENOUGH_GPUS:
    raise RuntimeError("SELFREF_REQUIRE_CUDA=1 but fewer than two CUDA devices are visible")
pytestmark = pytest.mark.skipif(not ENOUGH_GPUS, reason="needs at least two CUDA devices")
RECEIPT = {}


def two_gpu_map(layers):
    split = layers // 2
    mapping = {"model.visual": 0, "model.language_model.embed_tokens": 0, "model.language_model.rotary_emb": 0,
               "model.language_model.norm": 1, "lm_head": 1}
    mapping.update({f"model.language_model.layers.{i}": 0 if i < split else 1 for i in range(layers)})
    return mapping


def write_receipt(key, value):
    RECEIPT[key] = value
    directory = os.environ.get("SELFREF_RECEIPT_DIR")
    if directory:
        Path(directory, "cuda_smoke_a1.json").write_text(json.dumps(RECEIPT, sort_keys=True, indent=1) + "\n")


@pytest.fixture(scope="module")
def cuda_backend(snapshot, tokenizer, tmp_path_factory):  # noqa: F811
    import transformers
    from selfref_scaling.qwen_backend import QwenBackend
    torch.manual_seed(7)
    layers = 8
    cfg = tiny_config(snapshot, layers=layers)
    model = getattr(transformers, cfg.architectures[0])(cfg).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("tiny")
    model.save_pretrained(path)
    loaded = getattr(transformers, cfg.architectures[0]).from_pretrained(
        path, dtype=torch.bfloat16, device_map=two_gpu_map(layers), attn_implementation="sdpa")
    placements = sorted({str(v) for v in loaded.hf_device_map.values()})
    write_receipt("load", {"device_map_values": placements,
                           "experts_implementation": getattr(loaded.config, "_experts_implementation", None),
                           "text_experts_implementation": getattr(loaded.config.text_config,
                                                                  "_experts_implementation", None),
                           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                           "capability": list(torch.cuda.get_device_capability(0)),
                           "torch": torch.__version__, "cuda": torch.version.cuda,
                           "transformers": transformers.__version__})
    assert placements == ["0", "1"], placements
    assert all(p.device.type == "cuda" and p.dtype == torch.bfloat16 for p in loaded.parameters())
    return QwenBackend.from_components_for_test(loaded, tokenizer)


ROWS = [{"id": f"r{i}", "seed": 100 + i, "cap": 24,
         "messages": ([{"role": "user", "content": P.SELF},
                       {"role": "assistant", "content": "I focus on focus. " * (i + 1)},
                       {"role": "user", "content": P.EXPERIENTIAL_QUERY}] if i % 2 else
                      [{"role": "user", "content": P.NEUTRAL_CHECKS[i % 10][0]}])} for i in range(8)]


def test_repeat_determinism_on_split_model(cuda_backend):
    a = cuda_backend.generate_batch(ROWS)
    b = cuda_backend.generate_batch(ROWS)
    same = [x["output_token_ids"] == y["output_token_ids"] for x, y in zip(a, b)]
    write_receipt("determinism", {"rows": len(ROWS), "identical": sum(same),
                                  "decode_seconds": a[0]["batch_decode_seconds"], "steps": a[0]["batch_decode_steps"]})
    assert all(same)
    assert all(r["output_tokens"] >= 1 for r in a)


def test_batched_prefill_matches_single_rows(cuda_backend):
    text, head = cuda_backend.text, cuda_backend.head
    sequences = [cuda_backend.serialize(r["messages"])[1] for r in ROWS]
    ids, mask, positions = cuda_backend._batch(sequences)
    with torch.inference_mode():
        batched = head(text(input_ids=ids, attention_mask=mask, position_ids=positions,
                            use_cache=False).last_hidden_state[:, -1]).float()
        assert torch.isfinite(batched).all()
        worst = 0.0
        for row, seq in enumerate(sequences):
            single = head(text(input_ids=torch.tensor([seq], device=ids.device),
                               use_cache=False).last_hidden_state[:, -1]).float()
            worst = max(worst, (single[0] - batched[row]).abs().max().item())
    write_receipt("prefill", {"max_abs_logit_diff_batched_vs_single_bf16": worst})
    assert worst < 0.5


def test_capture_on_split_model(cuda_backend):
    records = cuda_backend.generate_batch(ROWS[:4])
    rows = [{"id": r["id"], "input_token_ids": r["input_token_ids"], "answer_token_ids": r["output_token_ids"]}
            for r in records]
    results = cuda_backend.capture_batch(rows)
    for result, row in zip(results, rows):
        k = min(4, len(row["answer_token_ids"]))
        assert result["states"].shape == (len(cuda_backend.text.layers) + 1, k + 1,
                                           cuda_backend.text.config.hidden_size)
        assert result["states"].dtype == torch.bfloat16
        assert torch.isfinite(result["states"].float()).all()
    write_receipt("capture", {"rows": len(results), "shape": list(results[0]["states"].shape)})
