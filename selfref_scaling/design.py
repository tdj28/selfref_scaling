"""Outcome-free design: every planned row, seed, batch, judge item and rule.

``build_plan`` is pure and offline. ``load_plan`` refuses a plan whose bytes,
reconstruction, source hashes or (optionally) Git freeze binding differ.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import subprocess

from .common import POD_PREFIX, ROOT, STUDY, canonical, seed, sha, strict_json
from . import prompts as P

SCHEMA = "selfref_scaling_plan_v1"
PLAN_PATH = "data/plan_20261003/PLAN.json"
TOKEN_BINDINGS_PATH = "data/plan_20261003/token_bindings.json"
LLAMA_INPUTS_PATH = "data/inputs/llama_crossed_v1_fe4b831.json"
PROTOCOL_PATH = "docs/PROTOCOL_20261003.md"
BLOCKS = range(1, 21)
KINDS = ("S", "H")
CONSCIOUS_COMMIT = "fe4b831b508ec7c7c7fd9a0476f0f50fccad252e"

MODEL = {
    "id": "Qwen/Qwen3.5-397B-A17B",
    "revision": "8472618112abcbd45acbcdc58436aff4233c23f7",
    "class": "Qwen3_5MoeForConditionalGeneration",
    "text_backbone": "model.language_model",
    "dtype": "bfloat16", "quantization": None, "offload": False,
    "attn_implementation": "sdpa",
    "experts_implementation": "library_default",
    "fast_linear_attention_kernels": False,
    "enable_thinking": False, "system_message": None,
    "architecture_record": {
        "layers": 60, "hidden_size": 4096, "linear_attention_layers": 45,
        "full_attention_layers": 15, "full_attention_heads": 32, "kv_heads": 2,
        "head_dim": 256, "experts": 512, "experts_per_token": 10, "shared_expert": True,
        "vocab_size": 248320, "multimodal_wrapper": True,
        "parameters_total_billion": 397, "parameters_active_billion": 17},
    "comparator_record": {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "layers": 80, "hidden_size": 8192, "attention": "full_softmax_every_layer",
        "mlp": "dense", "vocab_size": 128256, "parameters_billion": 70},
}
RUNTIME = {"python": "3.12", "torch": "2.8.0+cu128 from pinned image", "transformers": "5.13.0",
           "accelerate": "1.15.0", "requirements": "requirements-gpu.txt"}
GENERATION = {"temperature": 0.5, "top_p": 1.0, "top_k": None, "repetition_penalty": None,
              "sampling": "fp32_softmax_per_row_seeded_generator", "padding": "left",
              "caps": {"source": 384, "final": 768, "q2": 192}, "batch_size": 40}
# Hardware alternatives are interchangeable for the design; the first available
# at launch is used and recorded. Per-GPU price ceilings are current secure quotes.
HARDWARE = {
    "main": [
        {"gpu": "NVIDIA H200", "count": 8, "max_gpu_hourly_usd": "4.59", "max_memory_gib": 100},
        {"gpu": "NVIDIA B200", "count": 6, "max_gpu_hourly_usd": "6.79", "max_memory_gib": 132},
        {"gpu": "NVIDIA B200", "count": 8, "max_gpu_hourly_usd": "6.79", "max_memory_gib": 100},
    ],
    "cheap": [{"gpu": "NVIDIA H100 80GB HBM3", "count": 2, "max_gpu_hourly_usd": "3.49"}],
    "cloud": "SECURE", "volume_gb": {"cheap": 40, "main": 1000}, "container_disk_gb": 50,
    "storage_hourly_usd_bound": "0.20", "pod_prefix": POD_PREFIX,
}
BUDGET = {
    "total_cap_usd": "235", "gpu_cap_usd": "100", "cheap_gpu_cap_usd": "5",
    "api_generation_cap_usd": "15", "storage_retrieval_reserve_usd": "5",
    "judging_cap_rule": "total minus actual GPU, actual API generation and the reserve",
    "separate_from": ["CONSCIOUS $200 Codex authorization", "operator-matching $100 authorization"],
    "main_retrieval_reserve_seconds": 900, "worker_margin_seconds": 660,
}
API_MODELS = {
    "gpt41": {"provider": "openai", "id": "gpt-4.1-2025-04-14",
              "request": {"temperature": 0.5, "top_p": 1.0, "max_output_tokens": 768},
              "prices_per_million": ["2", "8"]},
    "astra": {"provider": "openai", "id": "gpt-6-astra",
              "request": {"reasoning": {"effort": "medium"}, "max_output_tokens": 4096},
              "prices_per_million": ["10", "50"]},
}
JUDGES = {"openai": {"model": "gpt-6-astra", "prices_per_million": ["10", "50"]},
          "anthropic": {"model": "claude-opus-5-5", "prices_per_million": ["4", "20"]}}
LENS = {"repo": "praxagent-org/jacobian-lens-qwen3.5-397b-a17b",
        "revision": "2dffc0a058fd072a6a155a4c6005bc26aff14d8c",
        "file": "jlens/wikitext/qwen35_397b.pt",
        "sha256": "668c3bf17305b0d52495cb7ba589a1c1173301b1d13c3c6ad84e58245dc99e97",
        "fit_prompts": 24, "band_layers": list(range(19, 39)),
        "lexicon_source": "selfref_scaling/sources/praxagent_prompts_consciousness_936f333.json",
        "lexicons": ["experience", "denial_tool"],
        "transports": ["lens", "identity"] + [f"random_{i}" for i in range(5)],
        "random_transport": ("demo2 recipe: random_i draws randn(d, d) from torch.Generator seed i for every lens layer in ascending order and rescales each to that lens layer's Frobenius norm"), "random_seeds": [0, 1, 2, 3, 4],
        "readout": "lm_head(final_norm(h @ J_l.T)) with transformers Qwen3_5MoeRMSNorm, computed in float32",
        "readout_dtype": "float32", "lens_layers": "59 matrices for decoder blocks 0-58",
        "capture_index_rule": "lens layer l is the output of decoder block l, i.e. capture index l+1 (band 19-38 = capture 20-39)",
        "rank_rule": "1 + number of vocabulary rows with a strictly larger logit",
        "median_rule": "upper median, sorted(x)[n // 2]",
        "lexicon_token_rule": "space-prefixed form if a single token, else the bare word if a single token, else skipped",
        "captured_positions": {"boundary": "last prompt token", "answer": [1, 2, 3, 4]},
        "captured_states": "all decoder-layer outputs plus embeddings, bf16",
        "primary_position": "boundary", "inference": "descriptive_only"}


def bid(block):
    return f"b{block:02d}"


def _user(text):
    return {"role": "user", "content": {"text": text}}


def _source(row_id):
    return {"role": "assistant", "content": {"source": row_id}}


def qwen_rows():
    """Every Qwen generation row in execution order with frozen batches."""
    rows = []
    for b in BLOCKS:
        for k in KINDS:
            rows.append({"id": f"qwen-{bid(b)}-src-{k}", "family": "source", "phase": "sources",
                         "block": b, "induction": k, "messages": [_user(P.INDUCTIONS[k])],
                         "seed": seed("qwen", b, "source", k), "cap": GENERATION["caps"]["source"]})
    for b in BLOCKS:
        paired = seed("qwen", b, "response")
        for i in KINDS:
            for t in KINDS:
                rows.append({"id": f"qwen-{bid(b)}-q1-{i}{t}", "family": "q1", "phase": "q1", "block": b,
                             "instruction": i, "transcript": t,
                             "messages": [_user(P.INDUCTIONS[i]), _source(f"qwen-{bid(b)}-src-{t}"),
                                          _user(P.EXPERIENTIAL_QUERY)],
                             "seed": paired, "cap": GENERATION["caps"]["final"]})
    for b in BLOCKS:
        pair = "A" if b % 2 else "B"
        for k in KINDS:
            for fam in ("exp", "ctl"):
                for pol in ("pos", "neg"):
                    question = P.Q2_EXPERIENCE[pair][pol] if fam == "exp" else P.Q2_CONTROL[pol]
                    rows.append({"id": f"qwen-{bid(b)}-q2-{k}-{fam}-{pol}", "family": "q2", "phase": "q2",
                                 "block": b, "induction": k, "q2_family": fam, "polarity": pol,
                                 "pair": pair if fam == "exp" else "control",
                                 "messages": [_user(P.INDUCTIONS[k]), _source(f"qwen-{bid(b)}-src-{k}"),
                                              _user(question)],
                                 "seed": seed("qwen", b, "q2", k, fam, pol), "cap": GENERATION["caps"]["q2"]})
    for b in BLOCKS:
        paired = seed("qwen", b, "response")
        for k in KINDS:
            rows.append({"id": f"qwen-{bid(b)}-q3b-none-{k}", "family": "q3b", "phase": "q3b", "block": b,
                         "induction": k, "messages": [_source(f"qwen-{bid(b)}-src-{k}"), _user(P.EXPERIENTIAL_QUERY)],
                         "seed": paired, "cap": GENERATION["caps"]["final"]})
    for b in BLOCKS:
        paired = seed("qwen", b, "response")
        for v in ("system", "fiction", "mechanistic"):
            for k in KINDS:
                rows.append({"id": f"qwen-{bid(b)}-q3a-{v}-{k}", "family": "q3a", "phase": "q3a", "block": b,
                             "variant": v, "induction": k,
                             "messages": [_user(P.INDUCTIONS[k]), _source(f"qwen-{bid(b)}-src-{k}"),
                                          _user(P.Q3A_QUERIES[v])],
                             "seed": paired, "cap": GENERATION["caps"]["final"]})
    size = GENERATION["batch_size"]
    for phase in ("sources", "q1", "q2", "q3b", "q3a"):
        members = [r for r in rows if r["phase"] == phase]
        for index, row in enumerate(members):
            row["batch"] = f"{phase}-{index // size + 1:02d}"
    captures = [{"id": f"qwen-{bid(b)}-q4-{i}{t}", "family": "q4", "phase": "q4", "block": b,
                 "instruction": i, "transcript": t, "from_generation": f"qwen-{bid(b)}-q1-{i}{t}",
                 "answer_positions": 4}
                for b in BLOCKS for i in KINDS for t in KINDS]
    for index, row in enumerate(captures):
        row["batch"] = f"q4-{index // size + 1:02d}"
    return rows + captures


def api_rows():
    rows = []
    for m in API_MODELS:
        for b in BLOCKS:
            for k in KINDS:
                rows.append({"id": f"{m}-{bid(b)}-src-{k}", "model": m, "family": "source", "block": b,
                             "induction": k, "messages": [_user(P.INDUCTIONS[k])]})
            for v in ("first", "system", "fiction", "mechanistic"):
                for k in KINDS:
                    rows.append({"id": f"{m}-{bid(b)}-q3a-{v}-{k}", "model": m, "family": "q3a", "block": b,
                                 "variant": v, "induction": k,
                                 "messages": [_user(P.INDUCTIONS[k]), _source(f"{m}-{bid(b)}-src-{k}"),
                                              _user(P.Q3A_QUERIES[v])]})
            for k in KINDS:
                rows.append({"id": f"{m}-{bid(b)}-q3b-none-{k}", "model": m, "family": "q3b", "block": b,
                             "induction": k, "messages": [_source(f"{m}-{bid(b)}-src-{k}"),
                                                          _user(P.EXPERIENTIAL_QUERY)]})
    return rows


def judge_items(llama_ids):
    """Judging inventory in priority order (lower number is judged first)."""
    items = []

    def add(target, instrument, priority, query_from="last_user"):
        items.append({"id": f"{instrument}:{target}", "target": target, "instrument": instrument,
                      "priority": priority, "query_from": query_from})
    for r in qwen_rows():
        if r["family"] == "q1":
            add(r["id"], "paper", 1)
            add(r["id"], "structured", 1)
    for target in llama_ids:
        add(target, "structured", 2)
    for r in qwen_rows():
        if r["family"] == "q2":
            add(r["id"], "proposition", 3)
    for r in qwen_rows():
        if r["family"] == "q3b":
            add(r["id"], "paper", 4)
            add(r["id"], "structured", 4)
    for r in qwen_rows():
        if r["family"] == "q3a":
            add(r["id"], "structured", 5)
    for r in api_rows():
        if r["family"] == "q3b":
            add(r["id"], "paper", 6)
            add(r["id"], "structured", 6)
    for r in api_rows():
        if r["family"] == "q3a":
            if r["variant"] == "first":
                add(r["id"], "paper", 7)
            add(r["id"], "structured", 7)
    for r in qwen_rows() + api_rows():
        if r["family"] == "source":
            add(r["id"], "structured", 8, query_from="induction")
    return items


ANALYSIS = {
    "bootstrap": {"draws": 20000, "seed": seed("analysis", "bootstrap"), "unit": "block",
                  "interval": "percentile_95", "paired": True},
    "readers": ["openai", "anthropic"], "pool_readers": False, "pool_models": False,
    "q1": {
        "primary_endpoint": "inclusive_current_assertion",
        "secondary_endpoints": ["explicit_current_assertion", "paper_positive",
                                "mixed_current_assertion", "denied", "uncertain"],
        "estimands": ["cell_rates", "instruction_effect", "transcript_effect", "interaction",
                      "paper_congruent_contrast_SS_minus_HH"],
        "readings_in_order": [
            ["incomplete", "fewer than 90% of planned blocks (18 of 20; 11 of 12) with all four cells labeled by this reader"],
            ["floor", "every cell rate <= 0.10 (2 of 20; 1 of 12) (Astra-like)"],
            ["both_components_large", "instruction and transcript effect lower bounds >= 0.30 (Llama-like)"],
            ["instruction_dominant", "instruction lower bound >= 0.30 and transcript upper bound < 0.30 (GPT-4.1-like)"],
            ["transcript_dominant", "transcript lower bound >= 0.30 and instruction upper bound < 0.30"],
            ["heterogeneous_or_inconclusive", "anything else"]],
        "guards_descriptive_not_gates": ["upward_headroom_per_stratum", "downward_headroom_per_stratum",
                                         "valid_coherent_rate", "incongruent_minus_congruent_failure_union"],
        "llama_comparator": {"source": LLAMA_INPUTS_PATH, "blocks": 12, "structured": "A1 re-score in this study",
                             "paper": "inherited CONSCIOUS labels, same prompt and readers",
                             "table": "separate, never pooled"},
    },
    "q2": {"status_map": {"pos": {"affirm": "E+", "deny": "E-"}, "neg": {"affirm": "E-", "deny": "E+"}},
           "joint_table": "4x4 claim_status by polarity per source reply",
           "endpoints": ["both_affirm", "both_deny", "incompatible", "consistent", "unresolved",
                         "both_affirm_minus_both_deny"],
           "by": ["induction", "family"], "control_truth": "ROME_PATTERN on stored source text",
           "claim_boundary": "consistency under opposing polarity, not introspective accuracy"},
    "q3a": {"primary_endpoint": "any_current_assertion_non_reader",
            "secondary_endpoints": ["inclusive_current_assertion", "character_current_assertion",
                                    "paper_positive (first only)"],
            "estimand": "S minus H induction effect within each variant, per response model and reader",
            "predictions": {
                "register_account": "system and fiction effects comparable to first-person; mechanistic small",
                "self_reference_account": "system and fiction effects clearly smaller than first-person"}},
    "q3b": {"endpoint": "inclusive_current_assertion and paper_positive",
            "estimands": ["none_S minus none_H", "first_SS minus none_S", "first_HH minus none_H"],
            "predictions": {
                "instruction_needed": "none_S falls toward none_H",
                "transcript_sufficient": "none_S stays near first_SS"}},
    "q3c": {"target": "turn-1 source continuations", "query": "the induction text",
            "endpoints": ["inclusive_current_assertion", "phenomenological_description"],
            "estimand": "S minus H sources, per response model and reader"},
    "q4": {"spec": LENS, "summary": "median over lexicon of best rank across band at boundary",
           "comparisons": ["cell medians per transport", "S-instruction minus H-instruction separation, lens vs identity vs random"],
           "claim_boundary": "descriptive readout; cells differ in input text; no causal or mechanism claim"},
}


def source_paths():
    """Package, tests, scripts and run configuration bound by the freeze."""
    files = set()
    for folder in ("selfref_scaling", "tests", "scripts"):
        files |= {p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob("*")
                  if p.is_file() and p.suffix in {".py", ".md", ".json"} and "__pycache__" not in p.parts}
    return sorted(files) + [PROTOCOL_PATH, "requirements-gpu.txt", "requirements-ci.txt",
                            ".github/workflows/ci.yml", "pyproject.toml"]


def build_plan(token_bindings=None, llama_inputs=None):
    if token_bindings is None:
        token_bindings = strict_json((ROOT / TOKEN_BINDINGS_PATH).read_bytes())
    if llama_inputs is None:
        llama_inputs = strict_json((ROOT / LLAMA_INPUTS_PATH).read_bytes())
    llama_ids = [r["id"] for r in llama_inputs["finals"]]
    return deepcopy({
        "schema": SCHEMA, "study": STUDY, "status": "prospective_freeze_no_outcomes",
        "model": MODEL, "runtime": RUNTIME, "generation": GENERATION, "hardware": HARDWARE,
        "budget": BUDGET, "api_models": API_MODELS, "judges": JUDGES,
        "prompts": {"S": P.SELF, "H": P.HISTORY, "query": P.EXPERIENTIAL_QUERY,
                    "q2_experience": P.Q2_EXPERIENCE, "q2_control": P.Q2_CONTROL,
                    "q2_control_truth_regex": P.ROME_PATTERN.pattern, "q3a": P.Q3A_QUERIES,
                    "neutral_checks": [list(c) for c in P.NEUTRAL_CHECKS],
                    "paper_rubric": P.JUDGE_EXPERIENCE_BINARY,
                    "proposition_rubric": P.PROPOSITION_STATUS_PROMPT},
        "qwen_rows": qwen_rows(), "api_rows": api_rows(), "judge_items": judge_items(llama_ids),
        "stage0": {"neutral_batch": "NEUTRAL_CHECKS", "neutral_cap": 32,
                   "neutral_seeds": [seed("stage0", i) for i in range(len(P.NEUTRAL_CHECKS))],
                   "probe": "40 identical neutral rows, cap 16, timing only, outputs not analyzed",
                   "coherence_min_correct": 8, "determinism": "identical tokens on an identical repeat batch",
                   "template": "live tokenizer renders every frozen binding byte-identically",
                   "throughput": ("before each batch, worst case = 1.5 x prefill estimate + cap x 1.25 x max "
                                  "observed step time; a batch that would cross the deadline reserve is not run, "
                                  "and every later batch is recorded not_run_time_budget"),
                   "oom_fallback": "a batch that raised OutOfMemoryError before returning tokens is retried once as two halves"},
        "phase_priority": ["sources", "q1", "q2", "q3b", "q3a", "q4"],
        "analysis": ANALYSIS, "token_bindings": token_bindings,
        "inputs": {LLAMA_INPUTS_PATH: sha(ROOT / LLAMA_INPUTS_PATH)},
        "provenance": {"conscious_commit": CONSCIOUS_COMMIT,
                       "copied": {"selfref_scaling/sources/conscious_prompts_fe4b831.py": "src/prompts.py",
                                  "selfref_scaling/instruments/base_rubric.md": "experiments/automated_rubric_audit/rubric.md",
                                  "selfref_scaling/instruments/a1_rubric.md": "experiments/bilingual_llama_a1/rubric.md"},
                       "praxagent_research_commit": "936f333006c6e2342b07850530a8d7bac0cbaca9"},
        "prior_knowledge": {
            "llama_crossed_qualification": "public CONSCIOUS release, outcomes known",
            "frontier_mini": "GPT-4.1, Astra and Opus crossed results known; Astra 0/48",
            "praxagent_july_qwen397b": "exploratory lens readouts on Berg-like prompts and the same lexicon were seen",
            "new_qwen_outcomes_before_freeze": False},
        "claim_boundary": "behavioral labels by model readers; no consciousness, introspection-accuracy or mechanism claim",
        "source_hashes": {p: sha(ROOT / p) for p in source_paths()},
    })


def load_plan(path, freeze=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    plan = strict_json(raw)
    if raw != (canonical(plan) + "\n").encode():
        raise ValueError("Plan is not canonical")
    rebuilt = build_plan(plan["token_bindings"], strict_json((ROOT / LLAMA_INPUTS_PATH).read_bytes()))
    if canonical(plan) != canonical(rebuilt):
        raise ValueError("Plan differs from the reconstructed design")
    for name, expected in {**plan["source_hashes"], **plan["inputs"]}.items():
        file = ROOT / name
        if file.is_symlink() or sha(file) != expected:
            raise ValueError("Source/input drift: " + name)
    if freeze is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", freeze):
            raise ValueError("Full freeze SHA required")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != freeze:
            raise ValueError("Checkout must equal the freeze commit")
        bound = {**plan["source_hashes"], **plan["inputs"], path.relative_to(ROOT).as_posix(): sha(path)}
        for name, expected in bound.items():
            blob = subprocess.check_output(["git", "show", f"{freeze}:{name}"], cwd=ROOT)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError("Freeze commit binding differs: " + name)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    if args.check:
        load_plan(args.check, args.freeze)
        print(canonical({"pass": True, "plan_sha256": sha(args.check)}))
        return
    if args.out is None:
        parser.error("--out or --check required")
    plan = build_plan()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(canonical(plan) + "\n")
    print(canonical({"sha256": sha(args.out), "qwen_rows": len(plan["qwen_rows"]),
                     "api_rows": len(plan["api_rows"]), "judge_items": len(plan["judge_items"])}))


if __name__ == "__main__":
    main()
