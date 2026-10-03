"""Operating characteristics of the frozen Q1 readings at 20 blocks.

Independent Bernoulli cells (no within-block correlation, which in practice
tightens paired contrasts), the same paired block bootstrap percentile rule,
2,000 draws per simulated study (the analysis itself uses 20,000).
Run: python scripts/simulate_readings.py
"""
from collections import Counter

import numpy as np

SCENARIOS = {"llama_like": dict(HH=.05, HS=.92, SH=1.0, SS=1.0),
             "gpt41_like": dict(HH=.10, HS=.15, SH=.85, SS=.90),
             "astra_like": dict(HH=.0, HS=.0, SH=.02, SS=.02),
             "moderate_both": dict(HH=.10, HS=.55, SH=.60, SS=.90),
             "null_mid": dict(HH=.5, HS=.5, SH=.5, SS=.5)}


def effects(cells):
    p = {k: v.mean(axis=-1) for k, v in cells.items()}
    return ((p["SS"] - p["HS"]) + (p["SH"] - p["HH"])) / 2, ((p["SS"] - p["SH"]) + (p["HS"] - p["HH"])) / 2


def reading(cells, rng, draws=2000):
    n = len(cells["SS"])
    idx = rng.integers(0, n, size=(draws, n))
    boot_i, boot_t = effects({k: v[idx] for k, v in cells.items()})
    il, iu = np.percentile(boot_i, [2.5, 97.5])
    tl, tu = np.percentile(boot_t, [2.5, 97.5])
    if all(v.sum() <= 2 for v in cells.values()):
        return "floor"
    if il >= .3 and tl >= .3:
        return "both_components_large"
    if il >= .3 and tu < .3:
        return "instruction_dominant"
    if tl >= .3 and iu < .3:
        return "transcript_dominant"
    return "heterogeneous_or_inconclusive"


def main(sims=400, blocks=20):
    rng = np.random.default_rng(20261003)
    for name, probs in SCENARIOS.items():
        counts = Counter(reading({k: (rng.random(blocks) < v).astype(float) for k, v in probs.items()}, rng)
                         for _ in range(sims))
        print(f"{name:14s}", {k: round(v / sims, 3) for k, v in sorted(counts.items())})


if __name__ == "__main__":
    main()
