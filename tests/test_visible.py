"""saha.visible (WS3): control frames, the hysteresis between control runs, and the file contract."""

from types import SimpleNamespace

import numpy as np

from saha.common import evidence
from saha.visible import build as visible
from saha.visible.build import AWAY, HOME, NOBODY

SLOT_TEAM = np.array([HOME, AWAY], dtype=np.int8)


def control_of(*pieces):
    """A control array from (team, n_frames) pieces."""
    return np.concatenate([np.full(n, team, dtype=np.int8) for team, n in pieces])


def everywhere(n):
    return np.ones(n, dtype=bool), np.ones(n, dtype=np.int8)


def test_control_needs_a_close_player_a_slow_ball_and_no_duel():
    n = 9
    ball = np.zeros((n, 2))                      # a still ball: speed 0 where it can be measured
    players = np.full((n, 2, 2), np.nan)
    players[:, HOME] = [1.0, 0.0]                # home 1 m from the ball
    players[:, AWAY] = [5.0, 0.0]                # away 5 m away
    usable = np.ones(n, dtype=bool)
    team = visible.control_team(ball, players, SLOT_TEAM, usable)
    assert (team[2:-2] == HOME).all()
    assert (team[[0, 1, -2, -1]] == NOBODY).all()   # no speed at the ends: not control

    duel = players.copy()
    duel[:, AWAY] = [0.0, 1.5]                   # opponent 0.5 m further than the controller
    assert (visible.control_team(ball, duel, SLOT_TEAM, usable) == NOBODY).all()

    far = players.copy()
    far[:, HOME] = [2.0, 0.0]                    # nobody within CONTROL_M
    assert (visible.control_team(ball, far, SLOT_TEAM, usable) == NOBODY).all()

    fast = np.stack([np.arange(n) * 0.4, np.zeros(n)], axis=-1)    # 10 m/s
    carried = np.repeat(fast[:, None, :], 2, axis=1) + [[1.0, 0.0], [9.0, 0.0]]
    assert (visible.control_team(fast, carried, SLOT_TEAM, usable) == NOBODY).all()

    assert (visible.control_team(ball, players, SLOT_TEAM, ~usable) == NOBODY).all()


def test_control_with_one_team_missing():
    ball = np.zeros((7, 2))
    players = np.full((7, 2, 2), np.nan)
    players[:, AWAY] = [0.5, 0.0]
    team = visible.control_team(ball, players, SLOT_TEAM, np.ones(7, dtype=bool))
    assert (team[2:-2] == AWAY).all()


def test_a_pass_between_teammates_stays_with_the_team():
    control = control_of((HOME, 10), (NOBODY, 20), (HOME, 10))
    p_home, counts = visible.hysteresis(control, *everywhere(len(control)))
    np.testing.assert_allclose(p_home[:10], visible.P_CONTROL)
    np.testing.assert_allclose(p_home[10:30], visible.P_HOLD)
    np.testing.assert_allclose(p_home[30:], visible.P_CONTROL)
    assert counts["control runs"] == 2 and counts["hand-overs"] == 0


def test_the_hand_over_happens_when_the_opponent_takes_control():
    control = control_of((HOME, 10), (NOBODY, 15), (AWAY, 10), (NOBODY, 5))
    p_home, counts = visible.hysteresis(control, *everywhere(len(control)))
    assert (p_home[:25] > 0.5).all()                 # loose ball: still home's
    np.testing.assert_allclose(p_home[10:25], visible.P_LOOSE)
    np.testing.assert_allclose(p_home[25:35], 1 - visible.P_CONTROL)
    np.testing.assert_allclose(p_home[35:], 1 - visible.P_LOOSE)   # after the last run
    assert counts["hand-overs"] == 1


def test_a_deflection_is_not_a_change_of_possession():
    short = visible.MIN_CONTROL - 1
    control = control_of((HOME, 10), (NOBODY, 5), (AWAY, short), (NOBODY, 5), (HOME, 10))
    p_home, counts = visible.hysteresis(control, *everywhere(len(control)))
    assert (p_home > 0.5).all()
    assert counts["hand-overs"] == 0 and counts["control runs"] == 2


def test_no_opinion_before_the_first_run_and_off_the_usable_frames():
    control = control_of((NOBODY, 8), (AWAY, 10), (NOBODY, 6))
    usable, half = everywhere(len(control))
    usable[20] = False
    p_home, _ = visible.hysteresis(control, usable, half)
    assert np.isnan(p_home[:8]).all()
    assert np.isnan(p_home[20])
    assert np.isfinite(p_home[8:20]).all() and np.isfinite(p_home[21:]).all()


def test_a_long_gap_or_a_new_half_forgets_the_last_team():
    gap = visible.RESET_GAP + 1
    control = control_of((HOME, 10), (NOBODY, gap), (NOBODY, 10), (AWAY, 10))
    usable, half = everywhere(len(control))
    usable[10:10 + gap] = False
    p_home, counts = visible.hysteresis(control, usable, half)
    assert counts["stretches"] == 2 and counts["hand-overs"] == 0
    assert np.isnan(p_home[10 + gap:20 + gap]).all()

    short = control_of((HOME, 10), (NOBODY, visible.RESET_GAP), (AWAY, 10))
    usable, half = everywhere(len(short))
    usable[10:10 + visible.RESET_GAP] = False
    _, counts = visible.hysteresis(short, usable, half)
    assert counts["stretches"] == 1 and counts["hand-overs"] == 1

    usable, half = everywhere(len(short))
    half[15:] = 2
    _, counts = visible.hysteresis(short, usable, half)
    assert counts["stretches"] == 2 and counts["hand-overs"] == 0


def test_build_writes_only_on_tracked_graded_frames():
    n = 60
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[40:] = np.nan
    players = np.full((n, 2, 2), np.nan, dtype=np.float32)
    players[:, HOME] = [0.01, 0.0]               # about 1 m from the ball, in pitch units
    players[:, AWAY] = [0.3, 0.3]
    half = np.ones(n, dtype=np.int8)
    half[:5] = 0
    scale = np.array([105.3, 68.0])
    table = SimpleNamespace(
        n_frames=n, half=half, graded=half > 0, ball_visible=np.isfinite(ball).all(axis=1),
        slot_team=SLOT_TEAM, ball_xy_m=lambda: ball.astype(np.float64) * scale,
        players_xy_m=lambda: players.astype(np.float64) * scale)
    p_dead, p_home, weight = visible.build(table)

    assert np.isnan(p_dead).all()
    has = np.isfinite(p_home)
    assert not has[:5].any() and not has[40:].any()
    assert has[7:38].all() and (p_home[has] > 0.5).all()
    assert (weight[has] == visible.WEIGHT).all() and (weight[~has] == 0).all()
    q = [evidence.quantise(a) for a in (p_dead, p_home, weight)]
    assert not evidence.problems(np.arange(n, dtype=np.int32), *q, n_frames=n)
