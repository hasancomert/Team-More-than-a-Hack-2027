"""WS2's dead-ball package on a small scripted match, with a stand-in for the trained classifier
(the real one needs SkillCorner, saha/deadball/train.py)."""

import numpy as np
import pytest

from saha.common import evidence, frame_table
from saha.deadball import build, dead, segment

N = 6000
HALVES = ((1, 200, 2799), (2, 3200, 5799))
STOPPAGE = (1200, 1700)     # the ball is dead here, and only here


def toy_table():
    """22 players and a ball. In play everyone runs and the ball moves; in the stoppage the
    players walk and the tracker has lost the ball."""
    rng = np.random.default_rng(0)
    t = np.arange(N)
    half = np.zeros(N, dtype=np.int8)
    for h, first, last in HALVES:
        half[first:last + 1] = h
    dead_frames = np.zeros(N, dtype=bool)
    dead_frames[STOPPAGE[0]:STOPPAGE[1]] = True
    pace = np.where(dead_frames, 0.5, 2.5)                       # m/s
    phase = np.cumsum(pace) / 25.0 / 10.0                        # radians along a 10 m circle
    home = np.arange(22) < 11
    base = np.stack([np.where(home, 0.30, 0.70) + rng.uniform(-0.1, 0.1, 22),
                     rng.uniform(0.2, 0.8, 22)], axis=1)
    circle = np.stack([np.cos(phase[:, None] + np.arange(22)),
                       np.sin(phase[:, None] + np.arange(22))], axis=-1) * 10.0
    players_xy = (base[None] + circle / [105.3, 68.0]).astype(np.float32)
    players_speed = np.repeat(pace[:, None], 22, axis=1).astype(np.float32)
    ball_xy = np.stack([0.5 + 0.3 * np.sin(t / 40.0), 0.5 + 0.3 * np.cos(t / 55.0)],
                       axis=1).astype(np.float32)
    ball_xy[dead_frames] = np.nan
    return frame_table.FrameTable(
        frame=t.astype(np.int32), atd_present=np.ones(N, dtype=bool), half=half,
        ball_xy=ball_xy, ball_z=None, ball_speed=None,
        ball_visible=np.isfinite(ball_xy).all(axis=1), players_xy=players_xy,
        players_speed=players_speed, slot_team=np.where(home, 0, 1).astype(np.int8),
        meta={"external": "toy"}), dead_frames


class OracleModel:
    """Stands in for ExternalDeadModel: certain, and right."""
    truth = None

    def logit(self, features, rows):
        assert features.shape == (N, 79)
        return np.where(self.truth[rows], 60.0, -60.0)


@pytest.fixture
def oracle(monkeypatch, tmp_path):
    table, truth = toy_table()
    OracleModel.truth = truth
    monkeypatch.setenv(dead.MODEL_ENV, str(tmp_path / "unused.pkl"))
    monkeypatch.setattr(segment, "ExternalDeadModel", OracleModel)
    return table, truth


def test_blocks_of():
    half = np.array([0, 1, 1, 0, 0, 2, 2, 2, 0], dtype=np.int8)
    assert segment.blocks_of(half) == [(1, 1, 2), (2, 5, 7)]


def test_model_path_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(dead.MODEL_ENV, str(tmp_path / "m.pkl"))
    assert dead.model_path() == tmp_path / "m.pkl" and not dead.model_disabled()
    monkeypatch.setenv(dead.MODEL_ENV, "none")
    assert dead.model_disabled()
    monkeypatch.delenv(dead.MODEL_ENV)
    assert dead.model_path().name == "dead_skillcorner.pkl"


def test_decoder_finds_the_stoppage(oracle):
    table, truth = oracle
    found = build.dead_ball(table, verbose=False)
    a, b = STOPPAGE
    assert found[a + 25:b - 25].all()
    assert not found[HALVES[0][1]:a - 25].any() and not found[b + 25:HALVES[0][2] + 1].any()
    assert not found[HALVES[1][1]:].any()
    assert not found[table.half == 0].any()


def test_build_fills_the_interface(oracle):
    table, truth = oracle
    p_dead, p_home, weight = build.build(table)
    graded = table.half > 0
    assert evidence.problems(table.frame, p_dead, p_home, weight, n_frames=N) == []
    assert np.isnan(p_home).all()
    assert set(np.unique(p_dead[graded])) <= {np.float32(build.P_DEAD),
                                              np.float32(1 - build.P_DEAD)}
    assert np.isnan(p_dead[~graded]).all()
    assert (weight[graded] == build.WEIGHT).all() and (weight[~graded] == 0).all()
    assert (p_dead[STOPPAGE[0] + 25] > 0.5) and (p_dead[STOPPAGE[0] - 100] < 0.5)


def test_no_classifier_means_no_opinion(monkeypatch):
    monkeypatch.setenv(dead.MODEL_ENV, "none")
    table, _ = toy_table()
    p_dead, p_home, weight = build.build(table)
    assert np.isnan(p_dead).all() and np.isnan(p_home).all() and (weight == 0).all()


def test_committed_model_loads():
    model = dead.ExternalDeadModel(dead.MODEL_FILE)
    assert model.n_features == 79 and len(model.models) == 3


def test_whistle_score_streams_like_one_pass(monkeypatch):
    from saha.deadball import audio
    rng = np.random.default_rng(0)
    hop, n_frames = audio.SAMPLE_RATE // audio.FPS, 300
    t = np.arange(n_frames * hop) / audio.SAMPLE_RATE
    signal = 0.05 * rng.standard_normal(len(t))
    blast = (t >= 100 / audio.FPS) & (t < 140 / audio.FPS)
    signal[blast] += 0.5 * np.sin(2 * np.pi * 4000.0 * t[blast])
    pcm = np.round(np.clip(signal, -1, 1) * 32767).astype("<i2")
    raw = memoryview(pcm.tobytes())
    position = [0]

    def read(n):   # an uneven stream, as from a pipe
        n = min(n, 7777)
        chunk = bytes(raw[position[0]:position[0] + n])
        position[0] += len(chunk)
        return chunk

    monkeypatch.setattr(audio, "BLOCK", 37)
    streamed = audio.whistle_score(read, n_frames)
    starts = np.arange(n_frames) * hop + hop // 2 - audio.NFFT // 2
    fits = starts + audio.NFFT <= len(pcm)
    direct = np.zeros(n_frames, dtype=np.float32)
    direct[fits] = audio.frame_scores(pcm / 32768.0, starts[fits], audio.band_mask())
    np.testing.assert_array_equal(streamed, direct)
    events = audio.whistle_events(streamed)
    assert len(events) == 1 and abs(events[0][0] - 100) <= 1 and abs(events[0][1] - 140) <= 1


def test_whistle_fingerprint():
    from saha.deadball import audio
    assert audio.fingerprint([(10, 20), (30, 45)]) == audio.fingerprint([(10, 20), (30, 45)])
    assert audio.fingerprint([(10, 20), (30, 45)]) != audio.fingerprint([(10, 20), (30, 46)])
