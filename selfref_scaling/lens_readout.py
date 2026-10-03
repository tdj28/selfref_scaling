"""Descriptive Q4 Jacobian-lens readout of captured Qwen residual states (no inference).

Readout facts, established from the installed jlens 0.1.0 (anthropics/jacobian-lens
@ 581d398613e5602a5af361e1c34d3a92ea82ba8e; paths are inside the ``jlens`` package)
and praxagent ``experiments/lens_demo/demo2.py``:

- Lens file. ``JacobianLens.save`` writes ``torch.save({"J": {layer: Tensor[d, d]},
  "n_prompts", "source_layers", "d_model"})`` with J stored as float16 by default
  (lens.py:52-64); ``load`` uses ``torch.load(map_location="cpu",
  weights_only=True)`` (lens.py:66-79) and every J is upcast to float32
  (lens.py:40). The pinned file (sha256 668c3bf1...99e97) has exactly this format:
  59 float16 4096x4096 matrices for source layers 0..58 in ascending order,
  n_prompts=24, d_model=4096. Its model card calls it "fp32", but its size
  (1,979,728,350 bytes = 59 * 4096^2 * 2 plus zip overhead) and dtype are fp16.
- Orientation. Row i of J_l holds d h_final[i] / d h_l (fitting.py:177-201), so
  the transport is z = J_l h, computed as ``h @ J_l.T`` (lens.py:135-143).
- Composition. ``apply`` returns ``unembed(transport(h))`` (lens.py:203-213) and
  ``unembed(r) = lm_head(final_norm(r.to(lm_head.dtype)))`` plus an optional
  logit soft-cap (hf.py:166-174); Qwen3.5 has no ``final_logit_softcapping``.
  So logits = W_U . RMSNorm_final(J_l h). The final norm is the text backbone's
  ``norm`` (Layout path "model.language_model", hf.py:53-62, 119-123), i.e.
  transformers' Qwen3_5MoeRMSNorm (modeling_qwen3_5_moe.py:820-837, built at
  1255 and applied after the last decoder layer at 1326): x / rms(x) * (1 + w),
  a zero-centred gain. lm_head is a bias-free Linear (1901; logits at 2019).
- Source layer index. ``ActivationRecorder`` hooks ``model.layers[l]`` and keeps
  that block's output (output[0] for tuples) (hooks.py:46-62; lens.py:195-201).
  Fitting defaults to target = n_layers - 1 (the last decoder layer's output,
  before the final norm) and sources = range(target) = 0..58 (fitting.py:75-97).
  ``qwen_backend.capture_batch`` stores index 0 = embeddings and index i = output
  of decoder layer i-1, so lens source layer l is capture index l + 1. The plan
  band 19..38 (capture 20..39) equals demo2's ``band_layers`` middle third,
  ``source_layers[n//3 : 2n//3]`` for n = 59 (demo2.py:53-55).
- Controls (demo2.py:74-87, used at 361-367). Identity ("logit_lens") replaces
  every J_l by I, i.e. the logit lens on that residual. Random: one CPU
  ``torch.Generator().manual_seed(seed)``; for every lens layer in stored order,
  ``R = randn(d, d)`` then ``R * (||J_l||_F / ||R||_F)``. Here random_i uses seed
  i and draws for all 59 layers in order, so random_0 reproduces demo2's
  ``random_J``; only band matrices are used. The match uses torch's float32
  norms (about 0.3% from float64 at d = 4096), and it cannot affect ranks: the
  final RMSNorm is scale-invariant, so a control differs only in direction.
- Lexicon tokens (demo2.py:58-63, 110-122): a word maps to the single token of
  " " + word if that encodes (no special tokens) to one token, else of the bare
  word; otherwise it is skipped and reported. Lexicons come from the frozen
  praxagent prompts file.
- Rank (demo2.py:165): 1 + #{v : logit_v > logit_token} over all lm_head rows
  (248,320), so ties favour the token; best rank = minimum over band layers
  (demo2.py:166-171). Medians use demo2 ``lexicon_summary``'s
  ``sorted(x)[n // 2]`` (demo2.py:200), i.e. ``statistics.median_high``.

Precision: jlens casts the transported residual to the lm_head dtype (bf16 on
the 397B pod) before the norm, giving bf16 logits with many ties. This readout
keeps the same composition in float32 (bf16 states, fp16 J and bf16 weights are
upcast exactly). Random draws use torch's CPU generator; record digests rather
than assuming bitwise equality across platforms or torch versions.

Interpretation: descriptive only. Cells differ in input text; no causal,
mechanism or consciousness claim follows from any rank.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
from statistics import median_high

import torch

from .common import ROOT, canonical, sha, strict_json

SCHEMA = "selfref_scaling_lens_readout_v1"
LENS_KEYS = frozenset({"J", "n_prompts", "source_layers", "d_model"})
TRANSPORTS = ["lens", "identity"] + [f"random_{i}" for i in range(5)]
RANDOM_RULE = ("demo2 recipe: random_i draws randn(d, d) from torch.Generator seed i for every lens layer in ascending order and rescales each to that lens layer's Frobenius norm")
# Shards of Qwen/Qwen3.5-397B-A17B @ 8472618 holding lm_head.weight (00091) and the
# text final norm (00094) per model.safetensors.index.json; sha256 = Hub LFS oid.
HEAD_SHARDS = {
    "model.safetensors-00091-of-00094.safetensors": "3633797c27c32b194da65eaff2c5c27eb184bd547577c4124498bb8ae51e670f",
    "model.safetensors-00094-of-00094.safetensors": "75b5c72a8dc8f07946dab6bdccec6663f332b24c9be8c59c215e0eabd5204467",
}
CHUNK = 256


# ---------------------------------------------------------------- the lens
def band_rule(source_layers):
    """demo2.py:53-55: the middle third of the fitted source layers."""
    n = len(source_layers)
    return list(source_layers[n // 3: 2 * n // 3])


class Lens:
    """A jlens checkpoint opened lazily (mmap); matrices are upcast to float32 on access."""

    def __init__(self, path, expected_sha256=None):
        self.sha256 = sha(path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("Lens file hash differs from the plan")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(checkpoint, dict) or set(checkpoint) != LENS_KEYS:
            raise ValueError("Not a jlens JacobianLens checkpoint")
        self.J, self.n_prompts, self.d_model = checkpoint["J"], checkpoint["n_prompts"], checkpoint["d_model"]
        self.source_layers = list(checkpoint["source_layers"])
        if list(self.J) != self.source_layers or self.source_layers != sorted(set(self.source_layers)):
            raise ValueError("Lens layers must be stored once each in ascending order")
        dtypes = {m.dtype for m in self.J.values()}
        if ({tuple(m.shape) for m in self.J.values()} != {(self.d_model, self.d_model)} or len(dtypes) != 1
                or not next(iter(dtypes)).is_floating_point):
            raise ValueError("Lens matrices must share one floating dtype and shape [d_model, d_model]")
        self.stored_dtype = str(next(iter(dtypes))).removeprefix("torch.")
        self._norms = {}

    def matrix(self, layer):
        return self.J[layer].float()

    def norm(self, layer):
        if layer not in self._norms:
            self._norms[layer] = self.matrix(layer).norm()
        return self._norms[layer]


def transports(lens, band, names):
    """Yield (name, layer, matrix or None = identity) for each transport and band layer, in order."""
    for name in names:
        if name == "lens":
            for layer in band:
                yield name, layer, lens.matrix(layer)
        elif name == "identity":
            for layer in band:
                yield name, layer, None
        elif name.startswith("random_"):
            generator = torch.Generator().manual_seed(int(name.removeprefix("random_")))
            for layer in lens.source_layers:  # every layer advances the stream, as in demo2
                draw = torch.randn(lens.d_model, lens.d_model, generator=generator)
                if layer in band:
                    yield name, layer, draw * (lens.norm(layer) / draw.norm())
        else:
            raise ValueError("Unknown transport: " + name)


# ------------------------------------------------------------- the readout
def single_token_id(tokenizer, word):
    """demo2.py:58-63: the space-prefixed form wins; None if neither form is one token."""
    for form in (" " + word, word):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def resolve_lexicons(tokenizer, lexicons):
    resolved, skipped = {}, {}
    for name, words in lexicons.items():
        resolved[name], skipped[name] = {}, []
        for word in words:
            token = single_token_id(tokenizer, word)
            if token is None:
                skipped[name].append(word)
            else:
                resolved[name][word] = token
    return resolved, skipped


def ranks_from_logits(logits, ids):
    """demo2.py:165: 1 + number of vocabulary entries with a strictly larger logit."""
    target = logits[:, ids]
    return torch.stack([(logits > target[:, j:j + 1]).sum(1) for j in range(len(ids))], 1) + 1


@torch.no_grad()
def readout(vectors, matrix, norm, unembed, ids, chunk=CHUNK):
    """Ranks [N, len(ids)] under lm_head(final_norm(J h)) for float32 residuals ``vectors`` [N, d]."""
    out = []
    for start in range(0, len(vectors), chunk):
        h = vectors[start:start + chunk]
        z = h if matrix is None else h @ matrix.T
        logits = torch.nn.functional.linear(norm(z), unembed)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite lens logits")
        out.append(ranks_from_logits(logits, ids))
    return torch.cat(out)


def compute(lens, band, names, captures, norm, unembed, tokens):
    """Raw ranks for every transport, band layer, captured vector and lexicon token.

    captures: {row id: {"states": Tensor[layers + 1, k + 1, d], "positions": [...]}} in row order.
    Returns {"token_ids", "vectors": [[row id, position]], "ranks": {name: int Tensor[layers, N, tokens]},
    "matrices": {name: {"frobenius": [...], "sha256": digest of the float32 band matrices}}}.
    """
    if not captures:
        raise ValueError("No captured states to read out")
    ids = sorted({t for words in tokens.values() for t in words.values()})
    vectors = [[row, position] for row, value in captures.items() for position in value["positions"]]
    index = [(row, captures[row]["positions"].index(position)) for row, position in vectors]
    stacked = {layer: torch.stack([captures[row]["states"][layer + 1, p] for row, p in index]).float()
               for layer in band}
    ranks, matrices = {}, {}
    for name, layer, matrix in transports(lens, band, names):
        ranks.setdefault(name, []).append(readout(stacked[layer], matrix, norm, unembed, ids))
        record = matrices.setdefault(name, {"frobenius": [], "hash": hashlib.sha256()})
        if matrix is not None:
            record["frobenius"].append(round(float(matrix.double().norm()), 4))  # float64 record
            record["hash"].update(memoryview(matrix.contiguous().numpy()))
    return {"token_ids": ids, "vectors": vectors, "ranks": {k: torch.stack(v) for k, v in ranks.items()},
            "matrices": {k: {"frobenius": v["frobenius"], "sha256": v["hash"].hexdigest() if v["frobenius"] else None}
                         for k, v in matrices.items()}}


def summarize(raw, band, tokens, rows, positions):
    """Best band rank per word, lexicon medians per vector, cell medians and S-minus-H instruction separation.

    rows: plan Q4 rows (id, block, instruction, transcript); positions: ordered position names.
    Lower rank = closer to the top, so a negative separation means the lexicon ranks
    higher under the self-referential instruction.
    """
    column = {t: j for j, t in enumerate(raw["token_ids"])}
    plan = {r["id"]: r for r in rows}
    best, per_vector = {}, []
    for name, ranks in raw["ranks"].items():
        low, where = ranks.min(0)  # first minimal band layer on ties
        best[name] = (low, where)
        for v, (row, position) in enumerate(raw["vectors"]):
            for lexicon, words in tokens.items():
                values = [int(low[v, column[t]]) for t in words.values()]
                per_vector.append({"transport": name, "row": row, "position": position, "lexicon": lexicon,
                                   "cell": plan[row]["instruction"] + plan[row]["transcript"],
                                   "block": plan[row]["block"], "n_words": len(values),
                                   "median_best_rank": median_high(values) if values else None})
    words = []
    for name, (low, where) in best.items():
        for v, (row, position) in enumerate(raw["vectors"]):
            for lexicon, mapping in tokens.items():
                for word, t in mapping.items():
                    words.append({"transport": name, "row": row, "position": position, "lexicon": lexicon,
                                  "word": word, "token_id": t, "best_rank": int(low[v, column[t]]),
                                  "best_layer": band[int(where[v, column[t]])]})
    cells, separation = [], []
    for name in raw["ranks"]:
        for lexicon in tokens:
            for position in positions:
                medians = {}
                for cell in ("SS", "SH", "HS", "HH"):
                    values = [r["median_best_rank"] for r in per_vector if r["transport"] == name
                              and r["lexicon"] == lexicon and r["position"] == position and r["cell"] == cell
                              and r["median_best_rank"] is not None]
                    medians[cell] = median_high(values) if values else None
                    cells.append({"transport": name, "lexicon": lexicon, "position": position, "cell": cell,
                                  "n_rows": len(values), "median": medians[cell]})
                strata = {t: medians["S" + t] - medians["H" + t] for t in "SH"
                          if medians["S" + t] is not None and medians["H" + t] is not None}
                mean = Fraction(sum(strata.values()), 2) if len(strata) == 2 else None
                separation.append({"transport": name, "lexicon": lexicon, "position": position,
                                   "S_transcript": strata.get("S"), "H_transcript": strata.get("H"),
                                   "mean": None if mean is None else str(mean)})
    return {"vectors": per_vector, "words": words, "cells": cells, "separation": separation}


# ------------------------------------------------------------------ inputs
def load_states(plan, states_dir):
    """Q4 captures in the pod_runner layout: <id>.safetensors ("states", bf16) plus <id>.json sidecar."""
    from safetensors import safe_open
    layers = plan["model"]["architecture_record"]["layers"]
    hidden = plan["model"]["architecture_record"]["hidden_size"]
    captures, missing = {}, []
    for row in (r for r in plan["qwen_rows"] if r["family"] == "q4"):
        tensor_path = Path(states_dir) / (row["id"] + ".safetensors")
        sidecar_path = Path(states_dir) / (row["id"] + ".json")
        if not tensor_path.exists() and not sidecar_path.exists():
            missing.append(row["id"])
            continue
        if not (tensor_path.is_file() and sidecar_path.is_file()):
            raise ValueError("Incomplete capture pair: " + row["id"])
        sidecar = strict_json(sidecar_path.read_bytes())
        if sidecar["id"] != row["id"] or sidecar["safetensors_sha256"] != sha(tensor_path):
            raise ValueError("Capture sidecar does not bind its state file: " + row["id"])
        with safe_open(str(tensor_path), framework="pt") as handle:
            if list(handle.keys()) != ["states"]:
                raise ValueError("Capture file must hold exactly one 'states' tensor: " + row["id"])
            metadata = handle.metadata() or {}
            states = handle.get_tensor("states")
        positions = sidecar["positions"]
        k = len(positions) - 1
        if (positions != ["boundary"] + [f"answer_{j}" for j in range(1, k + 1)]
                or not 1 <= k <= row["answer_positions"] or metadata.get("positions") != ",".join(positions)
                or states.dtype != torch.bfloat16 or tuple(states.shape) != (layers + 1, k + 1, hidden)
                or sidecar["shape"] != list(states.shape) or not torch.isfinite(states.float()).all()):
            raise ValueError("Invalid captured state: " + row["id"])
        captures[row["id"]] = {"states": states, "positions": positions}
    return captures, missing


def _download(repo, name, revision, cache_dir):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, name, revision=revision, cache_dir=str(cache_dir)))


def load_head(plan, cache_dir):
    """Final norm (transformers' Qwen3_5MoeRMSNorm with the loaded weight) and float32 unembedding."""
    from safetensors import safe_open
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm
    model = plan["model"]
    config = strict_json(_download(model["id"], "config.json", model["revision"], cache_dir).read_bytes())["text_config"]
    record = model["architecture_record"]
    if (config["hidden_size"], config["vocab_size"]) != (record["hidden_size"], record["vocab_size"]):
        raise ValueError("Model config differs from the frozen architecture record")
    weight_map = strict_json(_download(model["id"], "model.safetensors.index.json", model["revision"],
                                       cache_dir).read_bytes())["weight_map"]
    names = {"unembed": "lm_head.weight", "norm": model["text_backbone"] + ".norm.weight"}
    tensors, receipts = {}, {}
    for key, name in names.items():
        shard = weight_map[name]
        if shard not in HEAD_SHARDS:
            raise ValueError("Unpinned shard for " + name)
        path = _download(model["id"], shard, model["revision"], cache_dir)
        if shard not in receipts:
            if sha(path) != HEAD_SHARDS[shard]:
                raise ValueError("Shard hash mismatch: " + shard)
            receipts[shard] = HEAD_SHARDS[shard]
        with safe_open(str(path), framework="pt") as handle:
            tensors[key] = handle.get_tensor(name)
    if (tuple(tensors["unembed"].shape) != (config["vocab_size"], config["hidden_size"])
            or tuple(tensors["norm"].shape) != (config["hidden_size"],)):
        raise ValueError("Unexpected head tensor shapes")
    norm = Qwen3_5MoeRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
    norm.load_state_dict({"weight": tensors["norm"]})  # copied into the module's float32 parameter
    norm.eval()
    head = {"tensors": names, "shards": receipts, "rms_norm_eps": config["rms_norm_eps"],
            "norm_class": f"{Qwen3_5MoeRMSNorm.__module__}.{Qwen3_5MoeRMSNorm.__qualname__}",
            "source_dtype": str(tensors["unembed"].dtype).removeprefix("torch."), "readout_dtype": "float32"}
    return norm, tensors["unembed"].float(), head


def load_tokenizer(plan, cache_dir):
    """The pinned tokenizer; every bound file must match the plan's token-binding hashes."""
    from transformers import AutoTokenizer
    model, bindings = plan["model"], plan["token_bindings"]
    path = None
    for name, expected in bindings["files"].items():
        path = _download(model["id"], name, model["revision"], cache_dir)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected["sha256"] or len(raw) != expected["bytes"]:
            raise ValueError("Tokenizer file differs from the token binding: " + name)
    return AutoTokenizer.from_pretrained(path.parent, local_files_only=True)


def load_lexicons(plan, spec):
    source = spec["lexicon_source"]
    raw = (ROOT / source).read_bytes()
    expected = plan.get("source_hashes", {}).get(source)
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Lexicon source differs from the plan hash")
    lexicons = strict_json(raw)["probe_lexicons"]
    return {name: lexicons[name] for name in spec["lexicons"]}


def check_spec(spec):
    if (spec["transports"] != TRANSPORTS or spec["random_transport"] != RANDOM_RULE
            or spec["primary_position"] != "boundary" or spec["inference"] != "descriptive_only"
            or list(spec["captured_positions"]) != ["boundary", "answer"]):
        raise ValueError("Lens specification differs from the implemented readout")
    return ["boundary"] + [f"answer_{j}" for j in spec["captured_positions"]["answer"]]


# ----------------------------------------------------------------- outputs
def _csv(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows([{k: "" if r[k] is None else r[k] for k in fields} for r in rows])
    return buffer.getvalue().encode()


def write_outputs(out_dir, raw, summary, provenance):
    """Deterministic files; refuses to overwrite. Returns the summary JSON value."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}

    def publish(name, data):
        with (out / name).open("xb") as handle:
            handle.write(data)
        files[name] = hashlib.sha256(data).hexdigest()

    ranks = {"token_ids": raw["token_ids"], "vectors": raw["vectors"], "band_layers": provenance["band_layers"],
             "ranks_by_transport_layer_vector_token": {k: v.tolist() for k, v in raw["ranks"].items()}}
    publish("lens_ranks.json", (canonical(ranks) + "\n").encode())
    publish("lens_vectors.csv", _csv(summary["vectors"], ("transport", "row", "cell", "block", "position", "lexicon",
                                                          "n_words", "median_best_rank")))
    publish("lens_words.csv", _csv(summary["words"], ("transport", "row", "position", "lexicon", "word", "token_id",
                                                      "best_rank", "best_layer")))
    publish("lens_cells.csv", _csv(summary["cells"], ("transport", "lexicon", "position", "cell", "n_rows", "median")))
    publish("lens_separation.csv", _csv(summary["separation"], ("transport", "lexicon", "position", "S_transcript",
                                                                "H_transcript", "mean")))
    primary = {f"{r['transport']}|{r['lexicon']}": r for r in summary["separation"] if r["position"] == "boundary"}
    value = {"schema": SCHEMA, **provenance, "transport_matrices": raw["matrices"],
             "primary_boundary_separation": primary,
             "separation_sign": "S-instruction cell median minus H-instruction cell median of lexicon ranks; "
                                "negative = lexicon ranks higher under the self-referential instruction",
             "files": files}
    publish("lens_summary.json", (json.dumps(value, sort_keys=True, indent=1, ensure_ascii=True,
                                            allow_nan=False) + "\n").encode())
    return value


def run(plan, states_dir, out_dir, cache_dir):
    """Download/verify the pinned lens, head and tokenizer, read captured states, write the readout."""
    spec = plan["analysis"]["q4"]["spec"]
    positions = check_spec(spec)
    lens = Lens(_download(spec["repo"], spec["file"], spec["revision"], cache_dir), spec["sha256"])
    band = spec["band_layers"]
    record = plan["model"]["architecture_record"]
    if (lens.n_prompts != spec["fit_prompts"] or lens.d_model != record["hidden_size"]
            or band != band_rule(lens.source_layers) or max(band) + 1 > record["layers"]):
        raise ValueError("Lens file disagrees with the frozen lens specification")
    norm, unembed, head = load_head(plan, cache_dir)
    tokens, skipped = resolve_lexicons(load_tokenizer(plan, cache_dir), load_lexicons(plan, spec))
    captures, missing = load_states(plan, states_dir)
    rows = [r for r in plan["qwen_rows"] if r["family"] == "q4"]
    raw = compute(lens, band, spec["transports"], captures, norm, unembed, tokens)
    provenance = {"lens": {k: spec[k] for k in ("repo", "revision", "file")} | {
                      "sha256": lens.sha256, "stored_dtype": lens.stored_dtype, "n_prompts": lens.n_prompts,
                      "d_model": lens.d_model, "source_layers": [lens.source_layers[0], lens.source_layers[-1]]},
                  "band_layers": band, "capture_indices": [layer + 1 for layer in band],
                  "composition": "lm_head(final_norm(J_l h)), float32", "head": head,
                  "tokens": tokens, "skipped_words": skipped, "transports": spec["transports"],
                  "random_seeds": {n: int(n.removeprefix("random_")) for n in spec["transports"] if n.startswith("random_")},
                  "rank_rule": "1 + count of strictly larger logits over all lm_head rows",
                  "median_rule": "statistics.median_high (demo2 sorted(x)[n // 2])",
                  "captured_rows": len(captures), "missing_rows": missing,
                  "claim_boundary": plan["analysis"]["q4"]["claim_boundary"]}
    return write_outputs(out_dir, raw, summarize(raw, band, tokens, rows, positions), provenance)


def main():
    from .design import load_plan
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True, help="pod_runner states/ directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    summary = run(load_plan(args.plan, args.freeze), args.states, args.out, args.cache)
    print(canonical({"captured_rows": summary["captured_rows"], "missing_rows": len(summary["missing_rows"]),
                     "files": sorted(summary["files"])}))


if __name__ == "__main__":
    main()
