"""saha.visible (WS3): touches, confirmed hand-overs, and the file contract."""

from types import SimpleNamespace

import numpy as np

from saha.common import evidence
from saha.visible import build as visible
from saha.visible.build import AWAY, HOME, NOBODY

SLOT_TEAM = np.array([HOME, AWAY], dtype=np.int8)


def team_of(*pieces):
    """A touch-team array from (team, n_frames) pieces."""
    return np.concatenate([np.full(n, team, dtype=np.int8) for team, n in pieces])


def everywhere(n):
    return np.ones(n, dtype=bool), np.ones(n, dtype=np.int8)


def test_a_touch_is_the_closest_team_within_touch_distance_at_any_speed():
    n = 6
    ball = np.stack([np.arange(n) * 0.6, np.zeros(n)], axis=-1)        # 15 m/s
    players = np.full((n, 2, 2), np.nan)
    players[:, HOME] = ball + [1.0, 0.0]
    players[:, AWAY] = ball + [0.0, 1.2]                               # a duel: home is closer
    usable = np.ones(n, dtype=bool)
    assert (visible.touch_team(ball, players, SLOT_TEAM, usable) == HOME).all()

    far = players.copy()
    far[:, HOME] = ball + [2.0, 0.0]
    far[:, AWAY] = ball + [0.0, 3.0]
    assert (visible.touch_team(ball, far, SLOT_TEAM, usable) == NOBODY).all()

    alone = np.full((n, 2, 2), np.nan)
    alone[:, AWAY] = ball + [0.5, 0.0]                                 # no home player tracked
    assert (visible.touch_team(ball, alone, SLOT_TEAM, usable) == AWAY).all()
    assert (visible.touch_team(ball, players, SLOT_TEAM, ~usable) == NOBODY).all()


def test_touches_join_same_team_runs_only_across_short_breaks():
    join = visible.JOIN
    team = team_of((HOME, 4), (NOBODY, join), (HOME, 4), (NOBODY, join + 1), (HOME, 4))
    assert visible.touches(team, 0, len(team)) == [
        (0, 8 + join, HOME), (9 + 2 * join, 13 + 2 * join, HOME)]
    other = team_of((HOME, 4), (NOBODY, 1), (AWAY, 4))
    assert [k for _, _, k in visible.touches(other, 0, len(other))] == [HOME, AWAY]


def test_a_pass_between_teammates_stays_with_the_team():
    team = team_of((HOME, 8), (NOBODY, 20), (HOME, 8))
    p_home, counts = visible.possession(team, *everywhere(len(team)))
    np.testing.assert_allclose(p_home[:8], visible.P_TOUCH)
    np.testing.assert_allclose(p_home[8:28], visible.P_HOLD)
    np.testing.assert_allclose(p_home[28:], visible.P_TOUCH)
    assert counts["possessions"] == 1 and counts["hand-overs"] == 0


def test_an_interception_kept_by_the_player_changes_hands_at_his_first_touch():
    long = visible.CONFIRM_FRAMES
    team = team_of((HOME, 8), (NOBODY, 15), (AWAY, long), (NOBODY, 5))
    p_home, counts = visible.possession(team, *everywhere(len(team)))
    np.testing.assert_allclose(p_home[8:23], visible.P_LOOSE)          # on its way: still home's
    np.testing.assert_allclose(p_home[23:23 + long], 1 - visible.P_TOUCH)
    np.testing.assert_allclose(p_home[23 + long:], 1 - visible.P_LOOSE)
    assert counts["hand-overs"] == 1


def test_two_short_touches_by_the_new_team_confirm_from_the_first_one():
    short = visible.CONFIRM_FRAMES - 1
    gap = visible.JOIN + 5
    team = team_of((HOME, 8), (NOBODY, 5), (AWAY, short), (NOBODY, gap), (AWAY, short))
    p_home, counts = visible.possession(team, *everywhere(len(team)))
    assert (p_home[:13] > 0.5).all()
    assert (p_home[13:] < 0.5).all()                                   # switched at frame 13
    np.testing.assert_allclose(p_home[13 + short:13 + short + gap], 1 - visible.P_HOLD)
    assert counts["hand-overs"] == 1


def test_a_deflection_is_not_a_change_of_possession():
    short = visible.CONFIRM_FRAMES - 1
    team = team_of((HOME, 8), (NOBODY, 5), (AWAY, short), (NOBODY, 5), (HOME, 8))
    p_home, counts = visible.possession(team, *everywhere(len(team)))
    assert (p_home > 0.5).all()
    np.testing.assert_allclose(p_home[13:13 + short], visible.P_LOOSE)
    assert counts["hand-overs"] == 0 and counts["unconfirmed touches"] == 1


def test_no_opinion_before_the_first_possession_and_off_the_usable_frames():
    short = visible.CONFIRM_FRAMES - 1
    team = team_of((NOBODY, 8), (AWAY, short), (NOBODY, 6), (HOME, 10), (NOBODY, 6))
    usable, half = everywhere(len(team))
    usable[-2] = False
    p_home, counts = visible.possession(team, usable, half)
    assert np.isnan(p_home[:8 + short + 6]).all()                      # the away touch: unconfirmed
    assert np.isnan(p_home[-2])
    assert (p_home[8 + short + 6:-2] > 0.5).all()
    assert counts["possessions"] == 1


def test_the_first_touch_of_a_stretch_can_start_a_possession(monkeypatch):
    short = visible.CONFIRM_FRAMES - 1
    team = team_of((NOBODY, 8), (AWAY, short), (NOBODY, 6), (HOME, 10), (NOBODY, 6))
    monkeypatch.setattr(visible, "FIRST_TOUCH_OWNS", True)
    p_home, counts = visible.possession(team, *everywhere(len(team)))
    assert np.isnan(p_home[:8]).all()
    assert (p_home[8:8 + short + 6] < 0.5).all()                       # away's from its touch
    assert (p_home[8 + short + 6:] > 0.5).all()                        # home's long touch
    assert counts["possessions"] == 2 and counts["hand-overs"] == 1


def test_a_long_gap_or_a_new_half_forgets_the_last_team():
    gap = visible.RESET_GAP + 1
    short = visible.CONFIRM_FRAMES - 1
    team = team_of((HOME, 10), (NOBODY, gap), (NOBODY, 10), (AWAY, short), (NOBODY, 5))
    usable, half = everywhere(len(team))
    usable[10:10 + gap] = False
    p_home, counts = visible.possession(team, usable, half)
    assert counts["stretches"] == 2 and counts["hand-overs"] == 0
    assert np.isnan(p_home[10 + gap:]).all()        # one short touch cannot start a possession

    team = team_of((HOME, 10), (NOBODY, visible.RESET_GAP), (AWAY, 10))
    usable, half = everywhere(len(team))
    usable[10:10 + visible.RESET_GAP] = False
    _, counts = visible.possession(team, usable, half)
    assert counts["stretches"] == 1 and counts["hand-overs"] == 1

    usable, half = everywhere(len(team))
    half[15:] = 2
    _, counts = visible.possession(team, usable, half)
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
    assert has[5:40].all() and (p_home[has] > 0.5).all()
    assert (weight[has] == visible.WEIGHT).all() and (weight[~has] == 0).all()
    q = [evidence.quantise(a) for a in (p_dead, p_home, weight)]
    assert not evidence.problems(np.arange(n, dtype=np.int32), *q, n_frames=n)
