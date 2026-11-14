"""WS1 (saha.hiddendead): the second opinion on WS2's dead calls where the ball is missing.

Synthetic data only (CLAUDE.md, rule 3).
"""

import hashlib
import json

import numpy as np

from saha import pipeline
from saha.common import evidence, paths
from saha.hiddendead import build as hd_build, features as hd_features, train
from saha.tools import guard

from test_onball import home_on_the_ball, table_with_ball
from test_shape import N


def match_with_gap():
    """attacking_match with the ball tracked except on frames 300-600 (inside half 1)."""
    players, teams, ball = home_on_the_ball()
    visible = np.ones(N, dtype=bool)
    visible[300:600] = False
    return table_with_ball(players, teams, ball, visible)


def test_the_population_is_ws2_dead_frames_inside_play_where_the_ball_is_missing():
    table = match_with_gap()
    ws2_dead = np.zeros(N, dtype=bool)
    ws2_dead[100:150] = True          # before the half's first alive frame: kick-off wait
    ws2_dead[350:450] = True          # inside a gap: the population
    ws2_dead[700:750] = True          # ball tracked: WS2 stays final
    index, x = hd_features.inputs(table, ws2_dead)
    assert index.tolist() == list(range(350, 450))
    assert x.shape == (100, len(hd_features.names())) and x.dtype == np.float32
    assert np.isfinite(x[:, :6]).all()                 # positions in the gap and in the spell


def test_build_writes_p_dead_only_on_the_population():
    table = match_with_gap()
    ws2_dead = np.zeros(N, dtype=bool)
    ws2_dead[350:450] = True
    report = {}
    p_dead, p_home, weight = hd_build.build(table, ws2_dead, report=report)
    has = np.isfinite(p_dead)
    assert np.flatnonzero(has).tolist() == list(range(350, 450))
    assert np.isnan(p_home).all()
    assert (weight[has] == hd_build.WEIGHT).all() and (weight[~has] == 0).all()
    assert ((p_dead[has] >= hd_build.P_MIN) & (p_dead[has] <= hd_build.P_MAX)).all()
    assert "second opinion on 100" in hd_build.summary(report)


def test_an_override_drops_the_target_p_dead_only_where_the_source_has_one():
    n = 10
    nan = np.full(n, np.nan, np.float32)
    target = evidence.Evidence("deadball", np.full(n, 0.99, np.float32), nan,
                               np.full(n, 20.0, np.float32), {})
    p_dead = np.full(n, np.nan, np.float32)
    p_dead[3:6] = 0.2
    weight = np.where(np.isfinite(p_dead), 1.0, 0.0).astype(np.float32)
    source = evidence.Evidence("hiddendead", p_dead, nan, weight, {})
    entries = [{"name": "deadball"}, {"name": "hiddendead", "overrides": "deadball"}]
    out, counts = pipeline.apply_overrides([(target, 1.0), (source, 2.0)], entries)
    changed = out[0][0]
    assert counts == {"hiddendead": 3}
    assert np.isnan(changed.p_dead[3:6]).all()
    kept = np.r_[0:3, 6:n]
    assert (changed.p_dead[kept] == np.float32(0.99)).all()
    assert (changed.weight == target.weight).all() and out[1] == (source, 2.0)
    assert (target.p_dead == np.float32(0.99)).all()                    # the input is not changed


def test_the_committed_model_has_a_complete_manifest_and_matches_the_code():
    root = paths.REPO_ROOT
    path = train.MODEL_PATH.relative_to(root).as_posix()
    assert guard.is_model(path)
    assert train.MODEL_PATH.stat().st_size < guard.MAX_MODEL_KB * 1024
    manifest = json.loads((root / (path + ".json")).read_text())
    assert [key for key in guard.MANIFEST_KEYS if manifest.get(key) in (None, "", [], {})] == []
    assert manifest["sha256"] == hashlib.sha256(train.MODEL_PATH.read_bytes()).hexdigest()
    assert (root / manifest["training_script"]).is_file()
    assert manifest["licence"].startswith("CC BY 4.0")
    saved = hd_build.load_model()
    assert saved["n_features"] == manifest["features"] == len(hd_features.names())
    assert list(saved["names"]) == hd_features.names()
