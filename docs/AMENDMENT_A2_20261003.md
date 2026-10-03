# Amendment A2: Lens-Readout Key-Order Check

2026-10-03 (UTC). Post-outcome correction of a validation bug in frozen code.
No estimand, rule or computation changes.

## What Happened

After the main pod completed (all 440 generations and 80 state captures,
pod deleted with GET 404), the frozen Q4 readout
`python -m selfref_scaling.lens_readout` stopped before computing anything:
`check_spec` raised "Lens specification differs from the implemented readout".

Cause: `check_spec` compared the key order of
`plan["analysis"]["q4"]["spec"]["captured_positions"]` with
`["boundary", "answer"]`. The frozen plan file is canonical JSON with sorted
keys, so the loaded order is `["answer", "boundary"]`. The readout's tests
built the plan in memory (insertion order) and did not exercise the file. No
other order-dependent comparison exists in the frozen analysis, lens, judging
or outcome code; `analysis.check_plan` passes on the plan file.

## Correction

`amendments/a2_20261003/run_lens_a2.py` runs the frozen readout with one
substitution: the key-order comparison becomes a set comparison. All other
checks (transports, random-transport rule, primary position, descriptive-only
inference), the returned position list, the computation and the output files
are those of the frozen module. `test_run_lens_a2.py` shows the frozen check
rejects the actual plan file and the corrected one accepts it while still
rejecting every other mismatch.

## What Had Been Observed

At the time of this correction: the Qwen generations existed, but only
operational fields had been inspected (end-of-sequence, cap hits, think
tokens, empty responses) together with the stage-0 neutral answers. Judging was
in progress; only the fixture results (instrument checks, not outcomes) and
counts of target judgment statuses had been viewed. No target label, no
analysis output and no lens readout value existed or had been seen. The
correction cannot select among outcomes: it only lets the frozen readout run.
