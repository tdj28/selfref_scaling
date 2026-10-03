# AGENTS.md

Public repository for the Qwen3.5-397B-A17B extension of CONSCIOUS
(`https://github.com/tdj28/llm_selfref_pre`). Treat every tracked file, commit
and CI log as public and permanent.

## Rules

- The protocol `docs/PROTOCOL_20261003.md`, `data/plan_20261003/PLAN.json`,
  `data/plan_20261003/token_bindings.json`, `data/inputs/` and all code bound
  by the plan's `source_hashes` are frozen at the public freeze commit. Never
  edit them after the first outcome; post-outcome work goes in new, dated
  files that say what had already been seen.
- Never commit `.env` files, API keys, SSH material, approval files, model
  weights or caches. Run `python scripts/audit_public.py` and inspect the
  staged diff before every commit. Do not force-push or rewrite history.
- Credentials are read from the CONSCIOUS `.env` by path at launch time only.
- Pods: only the controller in `selfref_scaling/controller.py` creates or
  deletes pods, only with the prefix `scaling-qwen-20261003-`, exactly one
  create attempt per kind, and deletion only after verified final retrieval
  (direct GET 404). Never touch any other pod. Runtime state lives in the
  ignored `out/scaling-qwen-20261003/`.
- Budget: $235 total, GPU at most $100, API generation at most $15, $5
  reserve, judging the remainder. Separate from every CONSCIOUS authorization.
- One model only (owner decision): no intermediate Qwen sizes.
- Claims: model-reader labels of generated text; no consciousness,
  introspective-accuracy or mechanism claims. Keep Qwen, Llama, GPT-4.1 and
  Astra tables separate; never pool readers or models.
- Preserve failures, missing rows and unfavorable results in the release.
