# Self-Reference Reports In A Second Model Family

This repository extends the CONSCIOUS self-reference experiments
([tdj28/llm_selfref_pre](https://github.com/tdj28/llm_selfref_pre)) to
Qwen3.5-397B-A17B, a 397B-parameter mixture-of-experts model with hybrid
linear attention, and adds three questions to the original crossed design:
consistency under opposite yes/no questions, grammatical person and fiction,
and whether the instruction must stay visible. GPT-4.1 and GPT-6 Astra join
the register and visibility tests.

**Status: complete (2026-10-03).** Qwen3.5-397B-A17B almost never claimed
current experience when asked, even after the self-referential induction that
makes Llama 3.3 70B do so in nearly every answer: 3/20 positive labels in the
fully self-referential cell and 0/20 in the other three crossed cells, under both
readers. It denied experience in 76/80 crossed answers and in all 40
opposite-question pairs, although its own text written under the induction made
first-person experiential claims in 14/20 runs. GPT-4.1 reproduced the large
first-person effect but needed the instruction in view; GPT-6 Astra stayed at
zero. Read [`docs/RESULTS_20261003.md`](docs/RESULTS_20261003.md); the protocol
is [`docs/PROTOCOL_20261003.md`](docs/PROTOCOL_20261003.md). Amendment
[A1](docs/AMENDMENT_A1_20261003.md) corrected a CUDA test before any Qwen
outcome; [A2](docs/AMENDMENT_A2_20261003.md) fixed a key-order bug that stopped
the lens readout before it computed anything. Neither changed an estimand. One
post-hoc measure (fictional-voice claims) is labeled exploratory. Raw outputs,
receipts and analyses are in `data/release_20261003/`.

Nothing here measures consciousness. Labels come from two model readers
(GPT-6 Astra and Claude Opus 5.5) using instruments copied byte-for-byte from
CONSCIOUS; their agreement is not human validation.

## Layout

| Path | Contents |
| --- | --- |
| `selfref_scaling/design.py` | Every planned row, seed, batch, judge item, budget and reading rule |
| `selfref_scaling/prompts.py` | All prompts; paper prompts come from the verbatim CONSCIOUS copy in `sources/` |
| `selfref_scaling/qwen_backend.py` | Native-BF16 batched generation and state capture |
| `selfref_scaling/pod_runner.py` | The GPU worker: stage-0 gates, frozen batches, receipts |
| `selfref_scaling/controller.py` | Owned-pod lifecycle: one create, retrieval, verified deletion |
| `selfref_scaling/api_generate.py`, `judges.py` | GPT-4.1/Astra generation and judging with reserve-before-dispatch ledgers |
| `selfref_scaling/analysis.py`, `lens_readout.py` | Frozen estimands, readings and the descriptive lens readout |
| `data/inputs/` | Public Llama comparator answers copied from CONSCIOUS with hashes |

## Checks

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-ci.txt
python -m pytest -q
python -m selfref_scaling.design --check data/plan_20261003/PLAN.json
```

See [`NOTICE.md`](NOTICE.md) for provenance of copied files. Apache-2.0.
