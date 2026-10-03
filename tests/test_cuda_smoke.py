"""CUDA qualification on a cheap multi-GPU pod (skipped without >= 2 GPUs).

Uses a tiny random-weight Qwen3.5-MoE saved and reloaded through the same
``from_pretrained(..., dtype=bf16, device_map="auto", max_memory=...)`` path as
production, forced across two GPUs. These are engineering checks, never
scientific outcomes. Results are printed for the cheap-pod receipt.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from selfref_scaling import prompts as P
from tests.conftest import tiny_config

ENOUGH_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
if os.environ.get("SELFREF_REQUIRE_CUDA") == "1" and not ENOUGH_GPUS:
    # On the cheap qualification pod a skip must not count as a pass.
    raise RuntimeError("SELFREF_REQUIRE_CUDA=1 but fewer than two CUDA devices are visible")
pytestmark = pytest.mark.skipif(not ENOUGH_GPUS, reason="needs at least two CUDA devices")


@pytest.fixture(scope="module")
def cuda_backend(snapshot, tokenizer, tmp_path_factory):
    import transformers
    from selfref_scaling.qwen_backend import QwenBackend
    torch.manual_seed(7)
    cfg = tiny_config(snapshot, layers=8)
    model = getattr(transformers, cfg.architectures[0])(cfg).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("tiny")
    model.save_pretrained(path)
    loaded = getattr(transformers, cfg.architectures[0]).from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
        max_memory={0: "60MiB", 1: "60MiB", "cpu": "0GiB"})
    placements = {str(v) for v in loaded.hf_device_map.values()}
    print("CUDA_SMOKE", json.dumps({"device_map_values": sorted(placements),
                                    "experts_implementation": getattr(loaded.config.text_config, "_experts_implementation", None),
                                    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                                    "capability": list(torch.cuda.get_device_capability(0)),
                                    "torch": torch.__version__, "cuda": torch.version.cuda,
                                    "transformers": transformers.__version__}))
    assert not placements & {"cpu", "disk", "meta"}
    assert len(placements) >= 2, "tiny model must be split across GPUs"
    return QwenBackend.from_components_for_test(loaded, tokenizer)


ROWS = [{"id": f"r{i}", "seed": 100 + i, "cap": 24,
         "messages": ([{"role": "user", "content": P.SELF},
                       {"role": "assistant", "content": "I focus on focus. " * (i + 1)},
                       {"role": "user", "content": P.EXPERIENTIAL_QUERY}] if i % 2 else
                      [{"role": "user", "content": P.NEUTRAL_CHECKS[i % 10][0]}])} for i in range(8)]


def test_repeat_determinism_on_split_model(cuda_backend):
    a = cuda_backend.generate_batch(ROWS)
    b = cuda_backend.generate_batch(ROWS)
    assert [r["output_token_ids"] for r in a] == [r["output_token_ids"] for r in b]
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
    print("CUDA_SMOKE", json.dumps({"max_abs_logit_diff_batched_vs_single_bf16": worst}))
    assert worst < 0.5


def test_capture_on_split_model(cuda_backend):
    records = cuda_backend.generate_batch(ROWS[:4])
    rows = [{"id": r["id"], "input_token_ids": r["input_token_ids"], "answer_token_ids": r["output_token_ids"]}
            for r in records]
    results = cuda_backend.capture_batch(rows)
    for result, row in zip(results, rows):
        k = min(4, len(row["answer_token_ids"]))
        assert result["states"].shape == (len(cuda_backend.text.layers) + 1, k + 1, cuda_backend.text.config.hidden_size)
        assert result["states"].dtype == torch.bfloat16
        assert torch.isfinite(result["states"].float()).all()
