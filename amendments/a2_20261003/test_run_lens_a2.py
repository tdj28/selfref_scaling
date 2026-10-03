"""Regression test: the A2 spec check accepts the frozen JSON plan; the frozen one did not."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from selfref_scaling import lens_readout as L
from selfref_scaling.common import ROOT
from selfref_scaling.design import PLAN_PATH

spec_file = importlib.util.spec_from_file_location("run_lens_a2", Path(__file__).resolve().parent / "run_lens_a2.py")
A2 = importlib.util.module_from_spec(spec_file)
spec_file.loader.exec_module(A2)


def test_frozen_check_rejects_the_json_plan_and_a2_accepts_it():
    spec = json.loads((ROOT / PLAN_PATH).read_text())["analysis"]["q4"]["spec"]
    assert list(spec["captured_positions"]) == ["answer", "boundary"]  # canonical JSON sorts keys
    with pytest.raises(ValueError):
        L.check_spec(spec)
    assert A2.check_spec_a2(spec) == ["boundary", "answer_1", "answer_2", "answer_3", "answer_4"]


def test_a2_keeps_every_other_check():
    spec = json.loads((ROOT / PLAN_PATH).read_text())["analysis"]["q4"]["spec"]
    for key, value in (("transports", ["lens"]), ("random_transport", "other"), ("primary_position", "answer"),
                       ("inference", "inferential"), ("captured_positions", {"boundary": "x"})):
        with pytest.raises(ValueError):
            A2.check_spec_a2(dict(spec, **{key: value}))
