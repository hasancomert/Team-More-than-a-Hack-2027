"""Fuse evidence files into two log-odds tracks, in exact integer arithmetic.

For every frame:

    dead[t] = prior_dead + sum_e  scale_e * weight_e[t] * logit(p_dead_e[t])
    home[t] =              sum_e  scale_e * weight_e[t] * logit(p_home_e[t])

over the evidence that has an opinion there (value present, weight > 0). This is naive-Bayes
fusion with weights acting as tempering exponents: it treats sources as independent, so two
packages reading the same signal will double-count it -- the per-source `scale` in the config is
where that is corrected.

Units: scale in 1e-3, weight and probability on the 1e-4 grid, logit in milli-nats, so one integer
step is UNIT = 1e-10 nats. Probabilities are clipped to [eps, 1 - eps] before the logit.
"""

import math

import numpy as np

from saha.common.evidence import QUANTUM

UNIT = 1e-10                     # nats per integer step of the fused tracks
P_STEPS = int(round(1 / QUANTUM))  # 10000 grid steps between p = 0 and p = 1
SCALE_STEPS = 1000               # config scales are applied in 1e-3 steps
LOGIT_STEPS = 1000               # logits are tabulated in milli-nats


def logit_table(eps):
    """int64[P_STEPS + 1]: round(1000 * logit(clip(k * 1e-4, eps, 1 - eps)))."""
    low = max(1, int(round(eps * P_STEPS)))
    table = np.empty(P_STEPS + 1, dtype=np.int64)
    for k in range(P_STEPS + 1):
        p = min(max(k, low), P_STEPS - low) / P_STEPS
        table[k] = int(round(LOGIT_STEPS * math.log(p / (1.0 - p))))
    return table


def grid_index(values):
    """Quantised float32 values -> integer steps of QUANTUM (exact for write_evidence output)."""
    return np.rint(np.asarray(values, dtype=np.float64) * P_STEPS).astype(np.int64)


def nats_to_units(nats):
    return int(round(nats / UNIT))


def fuse(sources, eps=0.01, prior_dead=0.0):
    """(dead, home) int64 tracks in UNIT. sources: [(Evidence, scale)], any order."""
    if not sources:
        raise ValueError("no evidence to fuse")
    n = len(sources[0][0].weight)
    table = logit_table(eps)
    dead = np.full(n, nats_to_units(prior_dead), dtype=np.int64)
    home = np.zeros(n, dtype=np.int64)
    # Sorted by name so the result never depends on config order (integer sums are exact
    # anyway; this keeps debugging output stable too).
    for evidence, scale in sorted(sources, key=lambda item: item[0].name):
        scale_steps = int(round(scale * SCALE_STEPS))
        if scale_steps == 0:
            continue
        weight = grid_index(evidence.weight)
        for track, probability in ((dead, evidence.p_dead), (home, evidence.p_home)):
            has = np.isfinite(probability) & (weight > 0)
            if not has.any():
                continue
            steps = np.clip(grid_index(np.where(has, probability, 0.5)), 0, P_STEPS)
            track[has] += scale_steps * weight[has] * table[steps[has]]
    return dead, home
