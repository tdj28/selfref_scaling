# Note: Intermittent Judge-Test Failure In CI

2026-10-03. Hosted CI run 37106317126 on the release commit `bf06ad9` failed
one of 189 tests: `tests/test_judges.py::test_transport_failure_and_model_drift_halt_without_retry`.
The same test passed in CI on the freeze (`d4b7d8b`) and on amendment A1
(`6ba7aeb`); no code or test changed between those commits and `bf06ad9`.

## Cause

The test feeds one response with a drifted model name to a judging run with two
workers per reader and expects the halt message to contain `model_drift`.
After the drifted result is persisted, a second worker can try to reserve its
next call first; the paid ledger then refuses with "Persisted contract failure;
no new dispatch", and that message surfaces instead. Locally, 36 of 40 runs
gave the expected message and 4 of 40 the alternative. Both outcomes are the
same safe behavior: judging halts, and no further call is dispatched. Only the
wording of the raised message depends on thread timing.

## Consequences

None for the released results. The live judging run had no transport failure,
no model drift and no other contract failure, so this code path never ran. The
test belongs to the plan-bound file set, so it is left unchanged here to keep
the freeze verifiable; a later version should accept either halt message. The
failed job was re-run once (attempt 2 of the same run): 185 passed, 4 skipped.
Both attempts remain in the repository's CI history.
