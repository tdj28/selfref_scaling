"""Amendment A2: run the frozen Q4 lens readout with a corrected spec check.

The frozen ``lens_readout.check_spec`` compared the key ORDER of
``captured_positions`` with ["boundary", "answer"]. The frozen plan file is
canonical JSON with sorted keys, so the loaded order is ["answer", "boundary"]
and the readout refused to start before computing anything. This wrapper
replaces only that comparison with a set comparison; every other check, the
returned positions, the computation and the outputs are the frozen code's.
See docs/AMENDMENT_A2_20261003.md.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from selfref_scaling import lens_readout as L  # noqa: E402


def check_spec_a2(spec):
    if (spec["transports"] != L.TRANSPORTS or spec["random_transport"] != L.RANDOM_RULE
            or spec["primary_position"] != "boundary" or spec["inference"] != "descriptive_only"
            or set(spec["captured_positions"]) != {"boundary", "answer"}):
        raise ValueError("Lens specification differs from the implemented readout")
    return ["boundary"] + [f"answer_{j}" for j in spec["captured_positions"]["answer"]]


def install():
    L.check_spec = check_spec_a2


if __name__ == "__main__":
    install()
    L.main()
