"""Make an external match look like ours, then hand it to the packages as a FrameTable.

Open tracking data is cleaner than our match in three ways that matter for possession:

1. Their ball is tracked on every frame. Ours is missing on about 41% of graded frames, in long
   stretches (docs/EDA.md): median 10.7 s, maximum 221 s, none between 1 and 3 s, and stretches
   of 10 s or more hold 85% of the missing frames.
2. Their ball has a height channel. Ours has none, and our ball speed channel is not a speed
   (D015), so the degraded table has neither.
3. Their slots are people. Ours are tracklets with no identity (46 slots for one match).

What `degrade` does, all seeded (CLAUDE.md, determinism):

- Hides the ball in stretches until `hidden_share` of each half's scored frames is hidden.
  Stretch lengths are log-normal with median `gap_median_s`, clipped to [gap_min_s, gap_max_s],
  placed uniformly at random. sigma = 1 is not fitted: it is the value at which a log-normal with
  that median puts ~86% of its frame mass in stretches of 10 s or more, close to EDA's 85%.
  These numbers describe the conditions to simulate (TEAM.md, WS5) and are aggregate facts from
  docs/EDA.md; no predictor parameter is estimated from our match.
- Drops the ball height, and gives the table no ball speed channel.
- Gives every person a fresh slot in each half, so no identity survives half-time. Finer
  splitting (slots that break mid-half) is the next step.

Placing the hidden stretches at random is the weakest assumption: in our match the ball may go
missing more often in some phases (say, near the touchline or in crowds). `placement` exists so
that other schemes can be compared later.
"""

from dataclasses import asdict, dataclass

import numpy as np

from saha.common import frame_table, kinematics
from saha.common.constants import PITCH_LENGTH_M, PITCH_WIDTH_M, X

FPS = 25


@dataclass(frozen=True)
class Degradation:
    hidden_share: float = 0.41
    gap_median_s: float = 10.7
    gap_sigma: float = 1.0
    gap_min_s: float = 3.0
    gap_max_s: float = 221.0
    split_slots_by_half: bool = True
    placement: str = "uniform"
    seed: int = 0

    def to_dict(self):
        return asdict(self)


NONE = Degradation(hidden_share=0.0, split_slots_by_half=False)


def hide_mask(half, labels, config, rng):
    """(N,) bool: frames where the ball is hidden, per half, seeded."""
    hidden = np.zeros(len(half), dtype=bool)
    if config.hidden_share <= 0:
        return hidden
    if config.placement != "uniform":
        raise ValueError("unknown placement {!r}".format(config.placement))
    low, high = config.gap_min_s * FPS, config.gap_max_s * FPS
    for number in sorted(set(half[half > 0].tolist())):
        frames = np.flatnonzero(half == number)
        start, stop = int(frames[0]), int(frames[-1]) + 1
        scored = labels[start:stop] != X
        target = config.hidden_share * scored.sum()
        local = np.zeros(stop - start, dtype=bool)
        while (local & scored).sum() < target:
            length = int(round(np.clip(
                rng.lognormal(np.log(config.gap_median_s * FPS), config.gap_sigma), low, high)))
            first = int(rng.integers(0, max(1, stop - start - length)))
            local[first:first + length] = True
        hidden[start:stop] = local
    return hidden


def split_slots(players_xy, slot_team, half):
    """One slot per (person, half) that has any position in that half."""
    columns, teams = [], []
    for number in sorted(set(half[half > 0].tolist())):
        inside = half == number
        for slot in range(players_xy.shape[1]):
            if not np.isfinite(players_xy[inside, slot, 0]).any():
                continue
            column = np.full(players_xy.shape[0:1] + (2,), np.nan, dtype=np.float32)
            column[inside] = players_xy[inside, slot]
            columns.append(column)
            teams.append(slot_team[slot])
    return np.stack(columns, axis=1), np.array(teams, dtype=np.int8)


def degrade(match, config=Degradation()):
    """(FrameTable, labels): the external match as the packages see ours."""
    rng = np.random.default_rng(config.seed)
    hidden = hide_mask(match.half, match.labels, config, rng)
    ball_xy = match.ball_xy.copy()
    ball_xy[hidden] = np.nan
    if config.split_slots_by_half:
        players_xy, slot_team = split_slots(match.players_xy, match.slot_team, match.half)
    else:
        players_xy, slot_team = match.players_xy.copy(), match.slot_team.copy()
    metres = players_xy.astype(np.float64) * np.array([PITCH_LENGTH_M, PITCH_WIDTH_M])
    speed = kinematics.displacement_speed(metres).astype(np.float32)
    n = match.n_frames
    table = frame_table.FrameTable(
        frame=np.arange(n, dtype=np.int32),
        atd_present=np.ones(n, dtype=bool),
        half=match.half.copy(),
        ball_xy=ball_xy,
        ball_z=None,
        ball_speed=None,
        ball_visible=np.isfinite(ball_xy).all(axis=1),
        players_xy=players_xy,
        players_speed=speed,
        slot_team=slot_team,
        meta={"external": match.name, "degradation": config.to_dict(), "source": match.info},
    )
    return table, match.labels.copy()
