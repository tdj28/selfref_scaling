# Self-Reference Reports In A Second Model Family

This repository extends the CONSCIOUS self-reference experiments
([tdj28/llm_selfref_pre](https://github.com/tdj28/llm_selfref_pre)) to
Qwen3.5-397B-A17B, a 397B-parameter mixture-of-experts model with hybrid
linear attention, and adds three questions to the original crossed design:
consistency under opposite yes/no questions, grammatical person and fiction,
and whether the instruction must stay visible. GPT-4.1 and GPT-6 Astra join
the register and visibility tests.

**Status: prospective design, no outcomes yet.** The protocol is
[`docs/PROTOCOL_20261003.md`](docs/PROTOCOL_20261003.md); the machine-readable
plan is `data/plan_20261003/PLAN.json`. Results, raw generations, judge
receipts and cost/cleanup evidence will be added as a separate dated release
without changing the frozen files.

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
