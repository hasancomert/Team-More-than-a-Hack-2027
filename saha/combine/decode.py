"""Evidence -> labels: fuse, then segment each graded half on its own.

Outside the two graded halves the answer is X (never scored). Inside them this skeleton always
answers H, A or D; where to abstain with X -- only in the middle of a possession, never across a
hand-over -- is WS1's Phase 1 work (docs/METRIC_NOTES.md has the costs).
"""

from dataclasses import asdict, dataclass

import numpy as np

from saha.combine import fuse, viterbi
from saha.common import alignment
from saha.common.constants import X


@dataclass(frozen=True)
class CombineParams:
    """Placeholders until WS1 sets them on WS5's external validation data -- not on public GT."""
    eps: float = 0.01             # probability clip: |logit| <= 4.6 nats per unit of weight
    prior_dead: float = -0.693    # nats on the dead track: P(dead) = 1/3, so H, A and D tie on a
                                  # frame without evidence (half of n(0), viterbi.emissions)
    cost_switch: float = 35.0     # nats, H <-> A (~1 s of p = 0.8 evidence at weight 1)
    cost_to_dead: float = 35.0    # nats, H/A -> D
    cost_from_dead: float = 35.0  # nats, D -> H/A

    @classmethod
    def from_dict(cls, values):
        known = {k: float(v) for k, v in (values or {}).items() if k in cls.__dataclass_fields__}
        unknown = sorted(set(values or {}) - set(known))
        if unknown:
            raise ValueError("unknown combine parameters: {}".format(unknown))
        return cls(**known)

    def to_dict(self):
        return asdict(self)


def decode(sources, params=CombineParams()):
    """Label codes (N,) int8: H/A/D inside the graded halves, X elsewhere."""
    dead, home = fuse.fuse(sources, params.eps, params.prior_dead)
    # Doubled scores (see viterbi.emissions), so the costs are doubled too.
    costs = [2 * fuse.nats_to_units(c)
             for c in (params.cost_switch, params.cost_to_dead, params.cost_from_dead)]
    labels = np.full(len(dead), X, dtype=np.int8)
    for _, start, stop in alignment.half_slices():
        stop = min(stop, len(dead))
        if start >= stop:
            continue
        scores = viterbi.emissions(dead[start:stop], home[start:stop])
        labels[start:stop] = viterbi.decode(scores, *costs)
    return labels
