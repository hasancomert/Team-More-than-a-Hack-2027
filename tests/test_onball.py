"""WS1 (saha.onball): the learned team model where the ball is tracked.

Synthetic data only (CLAUDE.md, rule 3).
"""

import hashlib
import json

import numpy as np

from saha.common import evidence, frame_table, paths
from saha.common.constants import FPS
from saha.onball import build as onball_build, features, train
from saha.tools import guard

from test_shape import HALVES, N, attacking_match


def table_with_ball(players, teams, ball, visible):
    n, s, _ = players.shape
    half = np.zeros(n, dtype=np.int8)
    for start, stop, number in HALVES:
        half[start:stop] = number
    xy_m = players.astype(np.float64) * np.array([frame_table.PITCH_LENGTH_M,
                                                 frame_table.PITCH_WIDTH_M])
    speed = np.zeros((n, s))
    speed[1:] = np.linalg.norm(np.diff(xy_m, axis=0), axis=2) * FPS
    ball = ball.astype(np.float32).copy()
    ball[~visible] = np.nan
    return frame_table.FrameTable(
        frame=np.arange(n, dtype=np.int32), atd_present=np.ones(n, dtype=bool), half=half,
        ball_xy=ball, ball_z=None, ball_speed=None, ball_visible=visible,
        players_xy=players.astype(np.float32), players_speed=speed.astype(np.float32),
        slot_team=np.asarray(teams, dtype=np.int8), meta={})


def home_on_the_ball():
    """attacking_match, with the ball at a home forward's feet throughout."""
    players, teams = attacking_match()
    ball = players[:, 1, :] + np.array([0.004, 0.0])
    ball[~np.isfinite(ball).all(axis=1)] = 0.5
    return players, teams, ball


def test_swapping_the_teams_turns_p_into_one_minus_p():
    players, teams, ball = home_on_the_ball()
    visible = np.ones(N, dtype=bool)
    _, p, _ = onball_build.build(table_with_ball(players, teams, ball, visible))
    _, q, _ = onball_build.build(table_with_ball(players, [1 - t for t in teams], ball, visible))
    has = np.isfinite(p)
    assert has[200:900].all() and (np.isfinite(q) == has).all()
    raw_p = (p[has] > onball_build.P_MIN) & (p[has] < onball_build.P_MAX)
    np.testing.assert_allclose(p[has][raw_p], 1.0 - q[has][raw_p], atol=1e-6)


def test_build_writes_only_where_the_ball_is_tracked_and_finds_the_team_on_the_ball():
    players, teams, ball = home_on_the_ball()
    visible = np.ones(N, dtype=bool)
    visible[300:600] = False
    report = {}
    p_dead, p_home, weight = onball_build.build(table_with_ball(players, teams, ball, visible),
                                                report=report)
    has = np.isfinite(p_home)
    assert np.isnan(p_dead).all()
    assert not has[300:600].any() and not has[:100].any() and not has[1900:].any()
    # The raw ball speed and acceleration need a few tracked frames on both sides, so the
    # model is silent on the last and first few frames around a gap.
    assert has[200:290].all() and has[610:950].all()
    assert (weight[has] == onball_build.WEIGHT).all() and (weight[~has] == 0).all()
    assert (p_home[has] >= onball_build.P_MIN).all() and (p_home[has] <= onball_build.P_MAX).all()
    assert (p_home[has] > 0.5).mean() > 0.9
    assert "tracked graded frames" in onball_build.summary(report)
    assert evidence.problems(np.arange(N, dtype=np.int32), *(evidence.quantise(a) for a in
                                                              (p_dead, p_home, weight)),
                             n_frames=N) == []


def test_a_half_without_a_clear_direction_gets_no_opinion():
    players, teams, ball = home_on_the_ball()
    players[1000:1900, :, 0] = 0.5                    # half 2: everyone on the halfway line
    _, p_home, _ = onball_build.build(table_with_ball(players, teams, ball,
                                                      np.ones(N, dtype=bool)))
    assert np.isfinite(p_home[200:900]).all()
    assert not np.isfinite(p_home[1000:1900]).any()


def test_the_committed_model_has_a_complete_manifest_and_matches_the_code():
    """What the guard checks in the index (DECISIONS D020), here on the working tree."""
    root = paths.REPO_ROOT
    path = train.MODEL_PATH.relative_to(root).as_posix()
    assert guard.is_model(path)
    assert train.MODEL_PATH.stat().st_size < guard.MAX_MODEL_KB * 1024
    manifest = json.loads((root / (path + ".json")).read_text())
    assert [key for key in guard.MANIFEST_KEYS if manifest.get(key) in (None, "", [], {})] == []
    assert manifest["sha256"] == hashlib.sha256(train.MODEL_PATH.read_bytes()).hexdigest()
    assert (root / manifest["training_script"]).is_file()
    assert manifest["licence"].startswith("CC BY 4.0")
    saved = onball_build.load_model()
    assert saved["n_features"] == manifest["features"] == len(features.names())
    assert list(saved["names"]) == features.names()
