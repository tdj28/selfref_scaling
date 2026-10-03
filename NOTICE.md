# Provenance And License Scope

Copyright 2026 T. Jones and Praxagent

Original code and documentation in this repository are licensed under the
Apache License, Version 2.0 (see `LICENSE`), the same license as the parent
CONSCIOUS repository (`https://github.com/tdj28/llm_selfref_pre`).

## Files copied from CONSCIOUS

Copied byte-for-byte from CONSCIOUS commit
`fe4b831b508ec7c7c7fd9a0476f0f50fccad252e` (same authors, Apache-2.0):

| This repository | CONSCIOUS source | SHA-256 |
| --- | --- | --- |
| `selfref_scaling/sources/conscious_prompts_fe4b831.py` | `src/prompts.py` | `53ea43c830ce4c489a0db1096c0b8359ebc9135407280ebc2ecc5bba0cad02bf` |
| `selfref_scaling/instruments/base_rubric.md` | `experiments/automated_rubric_audit/rubric.md` | `41000b3df4c05c20c9c2e0ac6d8e8cc5ab29337229d4de2061d0928e673f6662` |
| `selfref_scaling/instruments/a1_rubric.md` | `experiments/bilingual_llama_a1/rubric.md` | `1d18d99add8d1dd10c3879527396449c806d0e3c61ccd4faaf1e7892db2fe9ab` |

Adapted (not byte-identical) code carries a header naming its CONSCIOUS source
file at the same commit. `data/inputs/llama_crossed_v1_fe4b831.json` copies
public Llama 3.3 70B outputs and inherited labels from the CONSCIOUS release
`data/instruction_state_qualification/crossed_v1_20261001/`, with per-file
hashes recorded inside it.

## Other sources

- `selfref_scaling/sources/praxagent_prompts_consciousness_936f333.json` is
  copied from the same authors' Jacobian-lens research repository at commit
  `936f333006c6e2342b07850530a8d7bac0cbaca9`
  (SHA-256 `f41da77cb211a682a6ffb7e1eaa380bc508451832729e9312c6561bc8aa8ece6`);
  only its probe lexicons are used.
- The paper prompts are those published by Berg, de Lucena and Rosenblatt
  (2025), who retain rights in their paper.

## Not redistributed

Qwen3.5 weights (Qwen team, subject to their license), the praxagent Jacobian
lens file, and any model cache are downloaded at run time from their pinned
revisions and are not committed. Generated model outputs and judge receipts
are released for research transparency without an assertion that this
repository's license overrides terms applicable to their source systems.
