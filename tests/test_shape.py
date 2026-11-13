"""WS1 (saha.shape): team-shape features, the symmetric model, and the evidence boundaries.

Synthetic data only (CLAUDE.md, rule 3).
"""

import json

import numpy as np

from saha.common import evidence, frame_table
from saha.common.constants import A, D, FPS, H, X
from saha.shape import build as shape_build, features, train

N = 2000
HALVES = ((100, 1000, 1), (1000, 1900, 2))


def make_table(players_xy, slot_team, ball_visible, meta=None):
    n, s, _ = players_xy.shape
    half = np.zeros(n, dtype=np.int8)
    for start, stop, number in HALVES:
        half[start:stop] = number
    xy_m = players_xy.astype(np.float64) * np.array([frame_table.PITCH_LENGTH_M,
                                                    frame_table.PITCH_WIDTH_M])
    step = np.zeros((n, s))
    step[1:] = np.linalg.norm(np.diff(xy_m, axis=0), axis=2) * FPS
    ball = np.full((n, 2), 0.5, dtype=np.float32)
    ball[~ball_visible] = np.nan
    return frame_table.FrameTable(
        frame=np.arange(n, dtype=np.int32), atd_present=np.ones(n, dtype=bool), half=half,
        ball_xy=ball, ball_z=None, ball_speed=None, ball_visible=ball_visible,
        players_xy=players_xy.astype(np.float32), players_speed=step.astype(np.float32),
        slot_team=np.asarray(slot_team, dtype=np.int8), meta=meta or {})


def attacking_match():
    """Home attacks high x in half 1 and low x in half 2, and has the ball throughout: its
    players are advanced, wide and moving forward; the away players are deep, narrow and
    retreating. Slot 0 and 3 are the goalkeepers."""
    t = np.arange(N) / FPS
    players = np.full((N, 6, 2), np.nan)
    for start, stop, number in HALVES:
        sign = 1.0 if number == 1 else -1.0           # home's direction of attack along x
        goal_home, goal_away = (0.03, 0.97) if number == 1 else (0.97, 0.03)
        k = slice(start, stop)
        drift = 0.01 * np.sin(t[k] / 3.0)
        players[k, 0] = np.stack([np.full(stop - start, goal_home), np.full(stop - start, 0.5)], 1)
        players[k, 3] = np.stack([np.full(stop - start, goal_away), np.full(stop - start, 0.5)], 1)
        forward = 0.5 + sign * (0.15 + 0.002 * (t[k] - t[start]))
        players[k, 1] = np.stack([forward + drift, np.full(stop - start, 0.1)], 1)
        players[k, 2] = np.stack([forward - sign * 0.05, np.full(stop - start, 0.9)], 1)
        back = 0.5 + sign * (0.30 + 0.002 * (t[k] - t[start]))
        players[k, 4] = np.stack([back, np.full(stop - start, 0.45)], 1)
        players[k, 5] = np.stack([back + sign * 0.02, np.full(stop - start, 0.55)], 1)
    return players, [0, 0, 0, 1, 1, 1]


def test_directions_follow_the_goalkeepers():
    players, teams = attacking_match()
    table = make_table(players, teams, np.ones(N, dtype=bool))
    d = features.directions(table)
    assert (d[100:1000, 0] == 1).all() and (d[100:1000, 1] == -1).all()
    assert (d[1000:1900, 0] == -1).all() and (d[1000:1900, 1] == 1).all()
    assert (d[:100] == 0).all() and (d[1900:] == 0).all()


def test_swapping_the_teams_turns_p_into_one_minus_p():
    players, teams = attacking_match()
    table = make_table(players, teams, np.ones(N, dtype=bool))
    swapped = make_table(players, [1 - t for t in teams], np.ones(N, dtype=bool))
    model = shape_build.load_model()
    f1, ok1, _ = features.pair_features(table)
    f2, ok2, _ = features.pair_features(swapped)
    assert (ok1 == ok2).all() and ok1[200:900].all()
    np.testing.assert_allclose(train.predict(model, f1[ok1]),
                               1.0 - train.predict(model, f2[ok2]), atol=1e-12)


def test_build_writes_only_on_missing_ball_frames_and_finds_the_attacking_team():
    players, teams = attacking_match()
    visible = np.ones(N, dtype=bool)
    visible[300:600] = False
    visible[1200:1500] = False
    visible[1950:] = False                            # not graded: no opinion there
    report = {}
    p_dead, p_home, weight = shape_build.build(make_table(players, teams, visible),
                                               report=report)
    has = np.isfinite(p_home)
    assert np.isnan(p_dead).all()
    assert has[300:600].all() and has[1200:1500].all()
    assert not has[:300].any() and not has[600:1200].any() and not has[1500:].any()
    assert (weight[has] == shape_build.WEIGHT).all() and (weight[~has] == 0).all()
    assert (p_home[has] >= shape_build.P_MIN).all() and (p_home[has] <= shape_build.P_MAX).all()
    assert (p_home[has] > 0.5).mean() > 0.9           # home has the ball
    assert "margin" in shape_build.summary(report)


def test_a_half_without_a_clear_direction_gets_no_opinion():
    players, teams = attacking_match()
    players[1000:1900, :, 0] = 0.5                    # half 2: everyone on the halfway line
    visible = np.zeros(N, dtype=bool)
    report = {}
    _, p_home, _ = shape_build.build(make_table(players, teams, visible), report=report)
    assert np.isfinite(p_home[200:900]).all()
    assert not np.isfinite(p_home[1000:1900]).any()
    assert "WARNING" in shape_build.summary(report)


def test_it_runs_on_an_external_table_too():
    players, teams = attacking_match()
    visible = np.ones(N, dtype=bool)
    visible[300:600] = False
    table = make_table(players, teams, visible, meta={"external": "toy"})
    p_dead, p_home, weight = shape_build.build(table)
    assert np.isfinite(p_home[300:600]).all()
    assert evidence.problems(np.arange(N, dtype=np.int32), *(evidence.quantise(a) for a in
                                                              (p_dead, p_home, weight)),
                             n_frames=N) == []


def test_controlled_labels_merge_short_possessions_only_between_the_other_team():
    short, long = FPS, 3 * FPS
    labels = np.concatenate([np.full(100, A), np.full(short, H), np.full(100, A),
                             np.full(long, H), np.full(100, A), np.full(short, H),
                             np.full(100, D), np.full(10, X)]).astype(np.int8)
    out = train.controlled(labels)
    assert (out[100:100 + short] == A).all()                       # A H A, H short: merged
    start = 200 + short
    assert (out[start:start + long] == H).all()                    # long: kept
    start += long + 100
    assert (out[start:start + short] == H).all()                   # A H D: kept
    assert (out[-10:] == X).all()


def test_the_model_file_is_complete_and_matches_the_features():
    model = json.loads(train.MODEL_PATH.read_text())
    _, _, names = features.pair_features(make_table(*attacking_match(), np.ones(N, dtype=bool)))
    assert [n.split(" ")[0] for n in model["features"]] == names
    assert len(model["coef"]) == len(model["mean"]) == len(model["std"]) == len(names)
    assert model["training"]["licence"] == "CC BY 4.0"
    assert model["training"]["script"] == "saha/shape/train.py"
    assert train.MODEL_PATH.stat().st_size < 10 * 1024              # the guard's JSON limit
