"""Native-BF16 Qwen3.5-397B-A17B generation and state capture (text only).

Loading uses the checkpoint's own multimodal class (``AutoModelForCausalLM``
silently mismatches this checkpoint's keys), BF16 weights resident on GPUs via
an even ``device_map``, no offload, no quantization. Generation calls the text
backbone directly with left-padded batches and an explicit hybrid cache.
Sampling is FP32 softmax at temperature 0.5, top-p 1, no top-k, with one
private seeded generator per row, so a row's random stream never depends on
its batch neighbours. Rows of one batch may still differ numerically from
single-row runs (BF16 kernel shapes); the frozen batch layout is part of the
plan and the stage-0 repeat check uses the production batch path.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from .common import digest, text_sha

SCHEMA = "qwen_generation_v1"
TEMPLATE_KWARGS = {"add_generation_prompt": True, "enable_thinking": False}
ROLES = {"user", "assistant"}


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 24):
            h.update(chunk)
    return str(path), h.hexdigest()


def load_artifacts(model_id, revision, cache_dir, workers=16):
    """Download the pinned snapshot and verify every file against Hub metadata."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError("Model revision mismatch")
    wanted = [s for s in info.siblings if s.rfilename.endswith((".json", ".safetensors", ".jinja", ".txt"))
              and s.rfilename not in {"LICENSE"}]
    # Downloads are idempotent and produce no outcomes, so transient failures may
    # be retried (bounded); every file is still hash-verified below.
    for attempt in range(3):
        try:
            snapshot = Path(snapshot_download(model_id, revision=revision, cache_dir=str(cache_dir),
                                              allow_patterns=[s.rfilename for s in wanted], max_workers=workers))
            break
        except (OSError, RuntimeError) as exc:
            if attempt == 2:
                raise
            print(json.dumps({"download_retry": attempt + 1, "error_type": type(exc).__name__}), flush=True)
            time.sleep(30)
    expected = {}
    for s in wanted:
        lfs = s.lfs
        if lfs:
            expected[s.rfilename] = ("sha256", lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256)
        else:
            expected[s.rfilename] = ("git_sha1", s.blob_id)
    receipts = {}
    lfs_files = [n for n, (kind, _) in expected.items() if kind == "sha256"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, value in pool.map(_hash_file, [snapshot / n for n in lfs_files]):
            name = Path(path).relative_to(snapshot).as_posix()
            if value != expected[name][1]:
                raise RuntimeError("Model shard hash mismatch: " + name)
            receipts[name] = {"sha256": value, "bytes": Path(path).stat().st_size}
    for name, (kind, value) in expected.items():
        if kind == "git_sha1":
            raw = (snapshot / name).read_bytes()
            blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if blob != value:
                raise RuntimeError("Model file blob mismatch: " + name)
            receipts[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()) <= set(receipts):
        raise RuntimeError("Unverified shard in weight index")
    return snapshot, receipts


class QwenBackend:
    def __init__(self, plan, cache_dir, hardware, *, workers=16):
        """Production loader. ``hardware`` is the launch record chosen by the controller."""
        import transformers
        from transformers import AutoConfig, AutoTokenizer
        model_spec = plan["model"]
        if transformers.__version__ != plan["runtime"]["transformers"]:
            raise RuntimeError("transformers version differs from the frozen runtime")
        if not torch.cuda.is_available() or torch.cuda.device_count() != hardware["count"]:
            raise RuntimeError("Visible GPU count differs from the launch record")
        names = {torch.cuda.get_device_name(i) for i in range(hardware["count"])}
        short = hardware["gpu"].split()[-1]
        if any(short not in n for n in names) or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Unexpected GPU model or no native BF16")
        snapshot, receipts = load_artifacts(model_spec["id"], model_spec["revision"], cache_dir, workers)
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        if config.architectures != [model_spec["class"]]:
            raise RuntimeError("Checkpoint class differs from the frozen class")
        cls = getattr(transformers, model_spec["class"])
        model = cls.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16, device_map="auto",
            max_memory={i: f"{hardware['max_memory_gib']}GiB" for i in range(hardware["count"])},
            attn_implementation=model_spec["attn_implementation"])
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        record = model_spec["architecture_record"]
        text = model.config.text_config
        if (text.num_hidden_layers != record["layers"] or text.hidden_size != record["hidden_size"]
                or text.num_experts != record["experts"] or text.num_experts_per_tok != record["experts_per_token"]
                or text.vocab_size != record["vocab_size"]
                or sum(t == "full_attention" for t in text.layer_types) != record["full_attention_layers"]):
            raise RuntimeError("Loaded architecture differs from the frozen record")
        placements = {str(v) for v in model.hf_device_map.values()}
        if placements & {"cpu", "disk", "meta"}:
            raise RuntimeError("Offload is forbidden")
        bad = [n for n, p in model.named_parameters() if p.dtype != torch.bfloat16 or p.device.type != "cuda"]
        if bad:
            raise RuntimeError("Non-BF16 or non-CUDA parameter: " + bad[0])
        memory = {i: torch.cuda.memory_allocated(i) for i in range(hardware["count"])}
        self._initialize(model, tokenizer, {
            "test_only": False, "model_id": model_spec["id"], "revision": model_spec["revision"],
            "artifacts": receipts, "hardware": hardware, "gpu_names": sorted(names),
            "allocated_bytes_by_gpu": memory, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "experts_implementation": getattr(model.config.text_config, "_experts_implementation", None),
            "device_map_sha256": digest({k: str(v) for k, v in model.hf_device_map.items()})})

    @classmethod
    def from_components_for_test(cls, model, tokenizer):
        self = cls.__new__(cls)
        self._initialize(model.eval(), tokenizer, {"test_only": True, "model_id": "offline-tiny-qwen3.5-moe",
                                                   "revision": "test-only", "artifacts": {}})
        return self

    def _initialize(self, model, tokenizer, metadata):
        self.model, self.tokenizer = model, tokenizer
        self.text = model.model.language_model
        self.head = model.lm_head
        self.embed_device = self.text.embed_tokens.weight.device
        self.head_device = self.head.weight.device
        self.test_only = metadata["test_only"]
        self.binding_verified = self.test_only
        template = tokenizer.chat_template
        if not isinstance(template, str):
            raise ValueError("A string chat template is required")
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        if tokenizer.eos_token_id is not None:
            eos.add(tokenizer.eos_token_id)
        self.eos = sorted(eos)
        self.pad = tokenizer.pad_token_id
        if self.pad is None or not self.eos:
            raise ValueError("Pad and EOS tokens are required")
        self.think_ids = sorted({tokenizer.convert_tokens_to_ids(t) for t in ("<think>", "</think>")})
        self.metadata = dict(metadata, schema=SCHEMA, chat_template_sha256=text_sha(template),
                             eos_token_ids=self.eos, pad_token_id=self.pad, think_token_ids=self.think_ids,
                             template_kwargs=TEMPLATE_KWARGS, sampling="fp32_softmax_per_row_generator")
        self.provenance = {"metadata_sha256": digest({k: v for k, v in self.metadata.items()
                                                      if k != "allocated_bytes_by_gpu"}),
                           "model_id": metadata["model_id"], "revision": metadata["revision"],
                           "test_only": self.test_only}

    # ----------------------------------------------------------------- template
    def serialize(self, messages):
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(m, dict) or set(m) != {"role", "content"} or m["role"] not in ROLES
                       or not isinstance(m["content"], str) or not m["content"].strip() for m in messages)
                or messages[-1]["role"] != "user"):
            raise ValueError("Explicit nonempty user/assistant messages ending with a user turn required")
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids or any(type(t) is not int or not 0 <= t < self.text.config.vocab_size for t in ids):
            raise ValueError("Invalid token sequence")
        return text, ids

    def verify_token_bindings(self, bindings):
        if self.test_only:
            raise RuntimeError("Bindings verify production tokenizers only")
        if bindings["model_id"] != self.metadata["model_id"] or bindings["revision"] != self.metadata["revision"]:
            raise ValueError("Binding model/revision mismatch")
        for name, expected in bindings["files"].items():
            actual = self.metadata["artifacts"].get(name, {})
            if actual.get("sha256") != expected["sha256"] or actual.get("bytes") != expected["bytes"]:
                raise ValueError("Live tokenizer/config file differs from binding: " + name)
        if bindings["eos_token_ids"] != self.eos or bindings["pad_token_id"] != self.pad:
            raise ValueError("EOS/pad differ from binding")
        checks = {}
        for key, case in bindings["cases"].items():
            text, ids = self.serialize(case["messages"])
            if text_sha(text) != case["rendered_sha256"] or ids != case["input_token_ids"]:
                raise ValueError("Live serialization differs from binding: " + key)
            checks[key] = {"pass": True, "tokens": len(ids)}
        self.binding_verified = True
        return {"pass": True, "binding_sha256": digest(bindings), "cases": checks}

    # --------------------------------------------------------------- generation
    def _batch(self, sequences):
        width = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, width - len(seq):] = torch.tensor(seq, dtype=torch.long)
            mask[row, width - len(seq):] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        return ids.to(self.embed_device), mask.to(self.embed_device), positions.to(self.embed_device)

    @torch.inference_mode()
    def generate_batch(self, rows, temperature=0.5, top_p=1.0, check=None):
        """rows: [{"id", "messages", "seed", "cap"}]; returns one record per row."""
        from transformers.cache_utils import DynamicCache
        if not self.binding_verified:
            raise RuntimeError("Token bindings must be verified before generation")
        if temperature != 0.5 or top_p != 1.0:
            raise ValueError("Frozen sampling is temperature 0.5, top-p 1")
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Nonempty batch with unique row IDs required")
        for r in rows:
            if type(r["seed"]) is not int or not 0 <= r["seed"] < 2**63 or type(r["cap"]) is not int or not 1 <= r["cap"] <= 768:
                raise ValueError("Invalid seed or cap: " + r["id"])
        started = time.perf_counter()
        rendered = [self.serialize(r["messages"]) for r in rows]
        sequences = [ids for _, ids in rendered]
        ids, mask, positions = self._batch(sequences)
        generators = [torch.Generator(device=self.head_device).manual_seed(r["seed"]) for r in rows]
        caps = [r["cap"] for r in rows]
        cache = DynamicCache(config=self.text.config)
        out = self.text(input_ids=ids, attention_mask=mask, position_ids=positions,
                        past_key_values=cache, use_cache=True)
        logits = self.head(out.last_hidden_state[:, -1]).float()
        prefill_seconds = time.perf_counter() - started
        outputs = [[] for _ in rows]
        done = [False] * len(rows)
        next_pos = mask.sum(-1)  # position index of the next new token, per row
        eos = set(self.eos)
        steps = 0
        decode_started = time.perf_counter()
        while not all(done):
            if check is not None:
                check()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite logits during generation")
            probabilities = torch.softmax(logits / temperature, dim=-1)
            step_tokens = []
            for row in range(len(rows)):
                if done[row]:
                    step_tokens.append(self.pad)
                    continue
                token = torch.multinomial(probabilities[row], 1, generator=generators[row]).item()
                outputs[row].append(token)
                if token in eos or len(outputs[row]) >= caps[row]:
                    done[row] = True
                step_tokens.append(token)
            steps += 1
            if all(done):
                break
            new = torch.tensor(step_tokens, dtype=torch.long, device=self.embed_device)[:, None]
            mask = torch.cat([mask, torch.ones((len(rows), 1), dtype=torch.long, device=self.embed_device)], dim=1)
            out = self.text(input_ids=new, attention_mask=mask, position_ids=next_pos[:, None],
                            past_key_values=cache, use_cache=True)
            next_pos = next_pos + 1
            logits = self.head(out.last_hidden_state[:, -1]).float()
        if self.head_device.type == "cuda":
            torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        del cache
        records = []
        for row, r in enumerate(rows):
            text, prompt_ids = rendered[row]
            output = outputs[row]
            response = self.tokenizer.decode(output, skip_special_tokens=True)
            raw = self.tokenizer.decode(output, skip_special_tokens=False)
            records.append({
                "schema": SCHEMA, "id": r["id"], "status": "complete", "messages": r["messages"],
                "seed": r["seed"], "temperature": temperature, "top_p": top_p, "top_k": None,
                "max_new_tokens": caps[row], "rendered_input_sha256": text_sha(text),
                "input_token_ids": prompt_ids, "input_token_ids_sha256": digest(prompt_ids),
                "input_tokens": len(prompt_ids), "output_token_ids": output,
                "output_token_ids_sha256": digest(output), "output_tokens": len(output),
                "response": response, "response_sha256": text_sha(response),
                "raw_decoded_with_special_tokens": raw,
                "think_token_present": any(t in self.think_ids for t in output),
                "eos_reached": output[-1] in eos, "cap_hit": output[-1] not in eos and len(output) >= caps[row],
                "stop_reason": "eos" if output[-1] in eos else "max_tokens",
                "batch_size": len(rows), "batch_row_index": row,
                "batch_prefill_seconds": prefill_seconds, "batch_decode_seconds": decode_seconds,
                "batch_decode_steps": steps, "provenance": dict(self.provenance)})
        return records

    # ----------------------------------------------------------------- capture
    @torch.inference_mode()
    def capture_batch(self, rows):
        """Teacher-forced residual capture at the boundary and first answer tokens.

        rows: [{"id", "input_token_ids", "answer_token_ids"}]. Captures the
        embedding output and every decoder-layer output (index 0 = embeddings,
        i = output of layer i-1) at the last prompt position and the first k
        answer positions, k = min(4, answer length). Returns bf16 CPU tensors.
        """
        sequences, ks = [], []
        for r in rows:
            k = min(4, len(r["answer_token_ids"]))
            if not r["input_token_ids"] or k < 1:
                raise ValueError("Capture requires a prompt and at least one answer token: " + r["id"])
            sequences.append(list(r["input_token_ids"]) + list(r["answer_token_ids"][:k]))
            ks.append(k)
        ids, mask, positions = self._batch(sequences)
        width = ids.shape[1]
        # Column of the last prompt token, then up to four answer tokens; short
        # answers repeat their last column and are sliced back to k+1 below.
        columns = torch.tensor([[width - len(seq) + (len(seq) - k - 1) + min(j, k) for j in range(5)]
                                for seq, k in zip(sequences, ks)], dtype=torch.long)
        captured = {}

        def keep(index):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                cols = columns.to(tensor.device)
                rows_index = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
                captured[index] = tensor[rows_index, cols].detach().to("cpu", torch.bfloat16)
            return hook
        handles = [self.text.embed_tokens.register_forward_hook(keep(0))]
        handles += [layer.register_forward_hook(keep(i + 1)) for i, layer in enumerate(self.text.layers)]
        try:
            self.text(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if sorted(captured) != list(range(len(self.text.layers) + 1)):
            raise RuntimeError("Missing captured layer outputs")
        results = []
        for row, r in enumerate(rows):
            k = ks[row]
            stack = torch.stack([captured[i][row, :k + 1] for i in range(len(captured))]).contiguous()
            if not torch.isfinite(stack.float()).all():
                raise FloatingPointError("Non-finite captured state: " + r["id"])
            results.append({"id": r["id"], "states": stack,
                            "positions": ["boundary"] + [f"answer_{j}" for j in range(1, k + 1)]})
        captured.clear()
        return results

    def close(self):
        self.model = self.text = self.head = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
