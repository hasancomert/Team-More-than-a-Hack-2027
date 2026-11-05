"""Three-state Viterbi over H, A, D with switching costs, in exact integer arithmetic.

Per-frame scores are doubled log-probabilities, so that no division is needed. The fused tracks
are dead = log P(D) / P(alive) and home = log P(H | alive) / P(A | alive), and the scores are

    D:  +dead          H:  -dead + home - n          A:  -dead - home - n

with n = 2 log(2 cosh(home / 2)) = |home| + 2 log(1 + exp(-|home|)). That is twice the
log-probability of each state, up to a constant shared by all three. The term n makes H and A
share the alive mass. Without it, a confident p_home would also push the alive states against D
(found by WS4, PR #7). n comes from a table in milli-nats, like the fusion's logits, so the scores
stay exact integers on every machine.

Changing state costs:

    H <-> A          cost_switch      a possession change
    H or A -> D      cost_to_dead     the ball goes dead
    D -> H or A      cost_from_dead   a restart

The costs are what turn a flickering per-frame signal into few, long possessions: a change only
happens where the evidence for it, summed over the frames it covers, outweighs its cost. Ties
break deterministically: stay in the current state first, then the lower code (H < A < D).
"""

import functools
import math

import numpy as np

from saha.common.constants import A, D, H

NORM_STEP = 10 ** 7     # integer units per milli-nat (the fused tracks count 1e-10 nats)
NORM_MAX = 10_000       # milli-nats; 2 log(1 + exp(-x)) rounds to 0 milli-nats long before 10 nats


@functools.lru_cache(maxsize=1)
def softplus_table():
    """int64[NORM_MAX + 1]: round(1000 * 2 log(1 + exp(-k / 1000))), in milli-nats."""
    table = np.empty(NORM_MAX + 1, dtype=np.int64)
    for k in range(NORM_MAX + 1):
        table[k] = int(round(2000.0 * math.log1p(math.exp(-k / 1000.0))))
    return table


def normaliser(home):
    """(T,) int64: n = 2 log(2 cosh(home / 2)) in units, |home| plus the tabulated softplus."""
    magnitude = np.abs(home)
    k = np.minimum((magnitude + NORM_STEP // 2) // NORM_STEP, NORM_MAX)
    return magnitude + softplus_table()[k] * NORM_STEP


def emissions(dead, home):
    """(T, 3) int64 doubled scores for H, A, D, from fused int64 tracks."""
    norm = normaliser(home)
    scores = np.empty((len(dead), 3), dtype=np.int64)
    scores[:, H] = -dead + home - norm
    scores[:, A] = -dead - home - norm
    scores[:, D] = dead
    return scores


def decode(scores, cost_switch, cost_to_dead, cost_from_dead):
    """The best path through (T, 3) integer scores. Costs are in the same doubled units."""
    T = len(scores)
    if T == 0:
        return np.empty(0, dtype=np.int8)
    cost = [[0, cost_switch, cost_to_dead],
            [cost_switch, 0, cost_to_dead],
            [cost_from_dead, cost_from_dead, 0]]
    s = scores.tolist()
    back = np.empty((T, 3), dtype=np.int8)
    back[0] = (H, A, D)
    best = list(s[0])
    for t in range(1, T):
        h, a, d = best
        new = []
        # Candidates in tie-break order: stay, then the lower code.
        for state, order in ((H, (H, A, D)), (A, (A, H, D)), (D, (D, H, A))):
            prev = (h, a, d)
            pick, value = order[0], prev[order[0]] - cost[order[0]][state]
            for other in order[1:]:
                candidate = prev[other] - cost[other][state]
                if candidate > value:
                    pick, value = other, candidate
            back[t, state] = pick
            new.append(value + s[t][state])
        best = new

    path = np.empty(T, dtype=np.int8)
    state = max((H, A, D), key=lambda k: (best[k], -k))  # highest score, lower code on ties
    for t in range(T - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return path
