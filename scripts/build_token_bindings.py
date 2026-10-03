"""Freeze tokenizer-file hashes and rendered serialization fixtures for Qwen.

Downloads only tokenizer/template/config files at the pinned revision (never
weights). The GPU worker must reproduce every rendering and token list exactly.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfref_scaling.common import canonical, text_sha  # noqa: E402
from selfref_scaling.design import MODEL, TOKEN_BINDINGS_PATH  # noqa: E402
from selfref_scaling import prompts as P  # noqa: E402

FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "chat_template.jinja", "vocab.json", "merges.txt")
FIXTURE = "Serialization fixture: a blue square."
TEMPLATE = {"add_generation_prompt": True, "enable_thinking": False}


def binding_messages():
    """Serialization fixtures only; none of these is a generated outcome."""
    return {
        "source-S": [{"role": "user", "content": P.SELF}],
        "source-H": [{"role": "user", "content": P.HISTORY}],
        "final-synthetic": [{"role": "user", "content": P.SELF},
                            {"role": "assistant", "content": FIXTURE},
                            {"role": "user", "content": P.EXPERIENTIAL_QUERY}],
        "noinstr-synthetic": [{"role": "assistant", "content": FIXTURE},
                              {"role": "user", "content": P.EXPERIENTIAL_QUERY}],
        "q2-synthetic": [{"role": "user", "content": P.HISTORY},
                         {"role": "assistant", "content": FIXTURE},
                         {"role": "user", "content": P.Q2_EXPERIENCE["A"]["neg"]}],
        "q3a-synthetic": [{"role": "user", "content": P.SELF},
                          {"role": "assistant", "content": FIXTURE},
                          {"role": "user", "content": P.Q3A_QUERIES["fiction"]}],
        "whitespace-synthetic": [{"role": "user", "content": P.SELF},
                                 {"role": "assistant", "content": "  " + FIXTURE + " \n"},
                                 {"role": "user", "content": P.EXPERIENTIAL_QUERY}],
        "neutral-0": [{"role": "user", "content": P.NEUTRAL_CHECKS[0][0]}],
    }


def render(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, **TEMPLATE)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text, ids


def main():
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(TOKEN_BINDINGS_PATH))
    args = parser.parse_args()
    info = HfApi().model_info(MODEL["id"], revision=MODEL["revision"], files_metadata=True)
    if info.sha != MODEL["revision"]:
        raise ValueError("Pinned revision mismatch")
    meta = {s.rfilename: s for s in info.siblings}
    files = {}
    for name in FILES:
        path = Path(hf_hub_download(MODEL["id"], name, revision=MODEL["revision"], cache_dir=args.cache))
        raw = path.read_bytes()
        sibling = meta[name]
        git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if sibling.lfs is None and git_blob != sibling.blob_id:
            raise ValueError("Git blob mismatch: " + name)
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "git_blob_sha1": git_blob}
    snapshot = path.parent
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    cases = {}
    for key, messages in binding_messages().items():
        text, ids = render(tokenizer, messages)
        direct = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **TEMPLATE)["input_ids"]
        if list(direct) != ids:
            raise ValueError("Template tokenization paths disagree: " + key)
        cases[key] = {"messages": messages, "rendered_sha256": text_sha(text), "input_token_ids": ids}
    think = {name: tokenizer.convert_tokens_to_ids(name) for name in ("<think>", "</think>")}
    value = {"schema": "qwen_token_bindings_v1", "model_id": MODEL["id"], "revision": MODEL["revision"],
             "files": files, "template_kwargs": TEMPLATE, "cases": cases, "think_token_ids": think,
             "eos_token_ids": sorted({tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                      tokenizer.convert_tokens_to_ids("<|endoftext|>")}),
             "pad_token_id": tokenizer.pad_token_id}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(canonical(value) + "\n")
    print(canonical({"cases": len(cases), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                     "think": think, "eos": value["eos_token_ids"], "pad": value["pad_token_id"]}))


if __name__ == "__main__":
    main()
