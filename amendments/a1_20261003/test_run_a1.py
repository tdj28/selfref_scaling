"""Offline checks of the A1 launcher substitutions (no network, no pods)."""
from __future__ import annotations

from decimal import Decimal
import importlib.util
from pathlib import Path

import pytest

from selfref_scaling import controller as C, design

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_a1", HERE / "run_a1.py")
A1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A1)
FREEZE = "a" * 40
DEADLINE = "2026-10-03T07:00:00+00:00"


def test_cheap_command_is_replaced_exactly_once_and_main_is_unchanged():
    cheap = A1.a1_worker_script("cheap", design.PLAN_PATH, FREEZE, DEADLINE)
    frozen = A1._frozen_worker_script("cheap", design.PLAN_PATH, FREEZE, DEADLINE)
    assert "amendments/a1_20261003/test_cuda_smoke_a1.py" in cheap and "tests/test_cuda_smoke.py" not in cheap
    assert "SELFREF_REQUIRE_CUDA=1 SELFREF_RECEIPT_DIR=/workspace/scaling/out " in cheap
    assert cheap.splitlines()[:-4] == frozen.splitlines()[:-4]  # clone, checkout, venv and install unchanged
    option = design.HARDWARE["main"][0]
    assert (A1.a1_worker_script("main", design.PLAN_PATH, FREEZE, DEADLINE, option)
            == A1._frozen_worker_script("main", design.PLAN_PATH, FREEZE, DEADLINE, option))


def test_budget_carry_adds_the_original_attempt(monkeypatch):
    monkeypatch.setattr(A1, "original_cheap_cost", lambda plan_hash: Decimal("0.42"))
    original = (C.worker_script, C.checked_budget)
    try:
        carried = A1.install("0" * 64)
        prior, reserve = C.checked_budget(design.BUDGET)
        assert carried == Decimal("0.42") and prior == Decimal("0.42") and reserve == 900
        assert C.worker_script is A1.a1_worker_script
    finally:
        C.worker_script, C.checked_budget = original


def test_original_freeze_is_rejected():
    with pytest.raises(SystemExit):
        A1.main(["--kind", "cheap", "--action", "launch", "--freeze", A1.ORIGINAL_FREEZE])
