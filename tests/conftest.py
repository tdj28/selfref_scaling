"""Shared fixtures: the pinned real tokenizer and a tiny random Qwen3.5-MoE.

No pretrained weights are ever downloaded; only tokenizer/config files at the
pinned revision. The tiny model reuses the real config class and vocabulary.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from selfref_scaling.design import MODEL

TOKENIZER_FILES = ("config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")
CACHE = Path(os.environ.get("SELFREF_HF_CACHE", Path(__file__).resolve().parents[1] / ".cache" / "hf"))


@pytest.fixture(scope="session")
def snapshot():
    from huggingface_hub import hf_hub_download
    path = None
    for name in TOKENIZER_FILES:
        path = Path(hf_hub_download(MODEL["id"], name, revision=MODEL["revision"], cache_dir=CACHE))
    return path.parent


@pytest.fixture(scope="session")
def tokenizer(snapshot):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def tiny_config(snapshot, layers=4):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    text = cfg.text_config
    for key, value in dict(hidden_size=64, num_hidden_layers=layers, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
                           moe_intermediate_size=32, shared_expert_intermediate_size=32,
                           linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
                           linear_value_head_dim=16,
                           layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)).items():
        setattr(text, key, value)
    vision = cfg.vision_config
    for key, value in dict(depth=1, hidden_size=32, intermediate_size=32, num_heads=2, out_hidden_size=64).items():
        if hasattr(vision, key):
            setattr(vision, key, value)
    return cfg


@pytest.fixture(scope="session")
def tiny_backend(snapshot, tokenizer):
    import torch
    import transformers
    from selfref_scaling.qwen_backend import QwenBackend
    torch.manual_seed(1234)
    cfg = tiny_config(snapshot)
    model = getattr(transformers, cfg.architectures[0])(cfg).float().eval()
    return QwenBackend.from_components_for_test(model, tokenizer)
