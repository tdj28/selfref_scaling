# Amendment A1: Corrected CUDA Qualification Test

2026-10-03 (UTC). Engineering-only amendment after the cheap CUDA
qualification failed on a defect in its own test setup. No scientific element
of the freeze changes.

## What Happened

The scientific freeze is commit `d4b7d8b01d29417c9ad1a595f82517dfa66dee06`
(plan SHA-256 `f8c4188b8310e63985666343eec43cbcea79c9c59be91dd5b21776c53d7f697d`),
which passed hosted CI (185 passed, 4 skipped).

The owned cheap pod `ig4th3qn83ti72` (2×H100, created 05:26:08Z) ran the frozen
cheap command. The five backend tests passed; the three tests in
`tests/test_cuda_smoke.py` errored at fixture setup. That fixture loaded its
tiny random model with `device_map="auto"` and a 60 MiB per-GPU cap; accelerate
placed one module on "disk", and the test's own no-offload assertion failed
(`{'0', '1', 'disk'}`). The CUDA, multi-GPU and grouped-expert paths were
therefore never exercised, so the gate did not qualify the runtime.

The worker recorded `pytest_exit_code: 1`. All artifacts were retrieved and
hash-verified, and the pod was deleted (direct GET 404) at 05:29:37Z with a
compute bound of $0.4187. The original ledger and retrieval are kept unchanged
under `out/scaling-qwen-20261003/controller/cheap/` and will be released.

This is a test-harness defect, not evidence about the production loader: the
production cap is 100 GiB per GPU (or 132 GiB on 6×B200) for 751.6 GiB of
weights, and production asserts no offload before any generation.

## What Changes

- `amendments/a1_20261003/test_cuda_smoke_a1.py` replaces the frozen CUDA smoke
  test on the cheap pod. It loads the same tiny model with an explicit two-GPU
  map (CPU or disk placement is impossible; an incomplete map raises) and
  performs the same checks: repeat determinism, batched versus single-row
  prefill, and state capture. It also writes `cuda_smoke_a1.json` recording
  the expert-kernel dispatch, device placement and timings. On the pod,
  `SELFREF_REQUIRE_CUDA=1` makes it fail rather than skip without two GPUs.
- `amendments/a1_20261003/run_a1.py` runs the frozen controller with two
  substitutions: the cheap pod's test list, and a separate run root
  `out/scaling-qwen-20261003/a1/` so the original ledger is never touched. For
  the main pod, the prior GPU spend includes the original attempt ($0.4187) in
  addition to the A1 cheap pod read by the frozen cheap gate.

## What Does Not Change

The plan file and its hash, all 45 files bound by its source hashes (code,
tests, scripts, protocol, requirements), prompts, seeds, rows, batches, judge
items, readings, analysis, budget caps and hardware alternatives. The amendment
files lie outside the bound folders, so `design.load_plan` still verifies the
plan at the A1 commit. Pods launched under A1 clone the A1 commit; the GPU
worker, plan and controller logic they run are byte-identical to the freeze.

The API-generation and judging ledgers bind the plan digest, which is
unchanged. GPT-4.1 and Astra generation had started under the original freeze;
only call status and cost were inspected, and nothing in this amendment depends
on any outcome. No Qwen3.5-397B outcome existed when it was written.

## Gate And Accounting

The main pod still requires the cheap pod's verified deletion (GET 404) and an
`APPROVE-cheap` file containing the plan hash, now under the A1 run root. It is
written only if the A1 cheap run shows no failed, errored or skipped test. Both
cheap attempts and the main pod count against the same $100 GPU cap and the
$5 cheap cap; the total cap stays $235.
