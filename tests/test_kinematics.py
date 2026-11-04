"""saha.common.kinematics: speeds and nearest players from positions."""

import numpy as np
import pytest

from saha.common import frame_table, kinematics
from saha.nearest.build import nearest_team


def straight_run(n_frames, metres_per_frame):
    frames = np.arange(n_frames, dtype=np.float64)
    return np.stack([metres_per_frame * frames, np.zeros(n_frames)], axis=-1)


def test_displacement_speed_of_a_steady_run():
    xy = straight_run(30, 0.2)                                   # 0.2 m a frame = 5 m/s
    speed = kinematics.displacement_speed(xy)
    assert np.isnan(speed[:2]).all() and np.isnan(speed[-2:]).all()
    np.testing.assert_allclose(speed[2:-2], 5.0)
    np.testing.assert_allclose(kinematics.displacement_speed(xy, span=25)[12:-12], 5.0)
    assert np.isnan(kinematics.displacement_speed(xy[:4])).all()    # too short to measure
    for span in (1, 4):
        with pytest.raises(ValueError):
            kinematics.displacement_speed(xy, span=span)


def test_displacement_speed_next_to_a_gap_and_with_extra_axes():
    xy = np.stack([straight_run(30, 0.2), straight_run(30, 0.4)], axis=1)   # (N, 2, 2)
    xy[10, 0] = np.nan
    speed = kinematics.displacement_speed(xy)
    assert speed.shape == (30, 2)
    assert np.isnan(speed[[8, 12], 0]).all() and np.isfinite(speed[[9, 10, 11], 0]).all()
    np.testing.assert_allclose(speed[2:-2, 1], 10.0)


def test_nearest_player():
    nan = [np.nan, np.nan]
    ball = np.array([[0.0, 0.0], [10.0, 0.0], nan, [0.0, 0.0]])
    players = np.array([[[1.0, 0.0], [0.0, 3.0], nan],
                        [[1.0, 0.0], [9.5, 0.0], [10.0, 2.0]],
                        [[1.0, 0.0], [9.5, 0.0], [10.0, 2.0]],
                        [nan, nan, nan]])
    distance, team = kinematics.nearest_player(ball, players, np.array([0, 1, 1], np.int8))
    np.testing.assert_allclose(distance[:2], [1.0, 0.5])
    assert np.isinf(distance[2:]).all() and team.tolist() == [0, 1, -1, -1]


def test_nearest_player_agrees_with_the_nearest_evidence(small_match, use_data, tmp_path):
    use_data(small_match, tmp_path / "cache")
    table = frame_table.load()
    ball, players = table.ball_xy_m(), table.players_xy_m()
    _, team = kinematics.nearest_player(ball, players, table.slot_team)
    assert (team >= 0).sum() > 1000
    np.testing.assert_array_equal(team, nearest_team(ball, players, table.slot_team))
