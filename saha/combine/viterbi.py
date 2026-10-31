"""Three-state Viterbi over H, A, D with switching costs, in exact integer arithmetic.

Per-frame scores (doubled, so that no division is needed):

    D:  +dead          H:  -dead + home          A:  -dead - home

which is twice the log-probability of each state under the fused log-odds, up to a constant.
Changing state costs:

    H <-> A          cost_switch      a possession change
    H or A -> D      cost_to_dead     the ball goes dead
    D -> H or A      cost_from_dead   a restart

The costs are what turn a flickering per-frame signal into few, long possessions: a change only
happens where the evidence for it, summed over the frames it covers, outweighs its cost. Ties
break deterministically: stay in the current state first, then the lower code (H < A < D).
"""

import numpy as np

from saha.common.constants import A, D, H


def emissions(dead, home):
    """(T, 3) int64 doubled scores for H, A, D, from fused int64 tracks."""
    scores = np.empty((len(dead), 3), dtype=np.int64)
    scores[:, H] = -dead + home
    scores[:, A] = -dead - home
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
