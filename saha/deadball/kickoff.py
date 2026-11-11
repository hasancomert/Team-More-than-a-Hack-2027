"""Kickoffs, from the laws of the game: before one, every player stands in his own half.

The tracker rarely sees the ball while a kickoff is set up, so the ball is no use here; the
formation is. Both teams wholly inside their own halves is a configuration open play almost
never holds for long, and it breaks the moment the ball is played -- players pour over the line.
The Smart Tagging marks a short window around every kickoff; inside it, the kickoff is where the
own-halves formation breaks.

Dead ball by definition: the set-up before every kickoff, and for the first kickoff of a half,
everything from the start of the half up to it.
"""

import numpy as np

HALF_TOLERANCE = 1.5   # m: a player may stand this far over the halfway line and still count
MAX_OVER = 1           # per side: the taker and tracking noise
BRIDGE = 10            # frames: breaks in the formation this short are tracking noise
SEARCH = 125           # frames: look this far past the tagged window for the break


def own_halves(players_m, team, attack_sign_by_side, length):
    """(n,) bool: both sides wholly in their own halves (within tolerance)."""
    ok = np.ones(len(players_m), dtype=bool)
    for side, sign in attack_sign_by_side.items():
        over = (players_m[:, team == side, 0] - length / 2.0) * sign[:, None]
        with np.errstate(invalid="ignore"):
            ok &= np.sum(over > HALF_TOLERANCE, axis=1) <= MAX_OVER
    return ok


def _bridge(mask, gap):
    """Fill False runs of at most `gap` frames that sit between True frames."""
    out = mask.copy()
    edges = np.diff(np.concatenate(([0], (~mask).astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a <= gap and a > 0 and b < len(mask):
            out[a:b] = True
    return out


def kickoffs(formation, tagged_windows):
    """[(setup_start, kickoff_frame)] for every tagged kickoff the formation confirms."""
    held = _bridge(formation, BRIDGE)
    out = []
    for start, stop in tagged_windows:
        window = np.arange(max(start, 0), min(stop + SEARCH, len(held)))
        inside = window[held[window]]
        if not len(inside):
            continue
        # the formation block that holds inside the window, and where it breaks
        end = inside[-1]
        while end + 1 < len(held) and held[end + 1]:
            end += 1
        begin = inside[0]
        while begin > 0 and held[begin - 1]:
            begin -= 1
        out.append((int(begin), int(end + 1)))
    return out


def kickoff_dead_mask(n, found, runs):
    """(n,) bool: frames that are dead ball because of a kickoff."""
    out = np.zeros(n, dtype=bool)
    for begin, kick in found:
        out[begin:kick] = True
    for first, stop in runs:
        inside = [kick for _, kick in found if first <= kick < stop]
        if inside:
            out[first:min(inside)] = True
    return out
