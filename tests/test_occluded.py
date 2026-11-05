"""WS4 (saha.occluded): the bridge over missing-ball gaps and the Smart Tagging phase teams.

Synthetic data only (CLAUDE.md, rule 3).
"""

import numpy as np
import pandas as pd

from saha.common import alignment, evidence, frame_table
from saha.common.constants import A, H
from saha.occluded import build as occluded
from saha.occluded import selfcheck

FIRST_HALF = alignment.half_slices()[0]


def test_runs_finds_true_stretches():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
    assert occluded.runs(mask) == [(1, 3), (5, 6), (7, 8)]
    assert occluded.runs(np.zeros(4, dtype=bool)) == []


def test_bridge_follows_each_side_and_fades_in_the_middle():
    agree = occluded.bridge(0.9, 0.9, 50 * 25)
    assert agree[0] > 0.89 and agree[-1] > 0.89
    assert (agree > 0.5).all() and agree[len(agree) // 2] < agree[0]

    handover = occluded.bridge(0.9, 0.1, 10 * 25)
    assert handover[0] > 0.85 and handover[-1] < 0.15
    assert (np.diff(handover) < 0).all()                     # one crossing, no flicker
    np.testing.assert_allclose(handover, 1.0 - handover[::-1], atol=1e-12)  # symmetric

    assert np.all(occluded.bridge(0.5, 0.5, 100) == 0.5)     # no side, no opinion
    long_gap = occluded.bridge(0.9, 0.5, 200 * 25)
    assert abs(long_gap[-1] - 0.5) < 1e-3                    # 200 s on: nothing left


def test_anchor_weighs_recent_touches():
    lo, hi = 0, 200
    control = np.full(hi, occluded.NOBODY, dtype=np.int8)
    assert occluded.anchor(control, 99, -1, lo, hi) == 0.5            # no touch
    control[99] = occluded.HOME
    one = occluded.anchor(control, 99, -1, lo, hi)
    assert abs(one - (0.5 + (occluded.ANCHOR_MAX - 0.5) / 2)) < 1e-12
    control[80:99] = occluded.HOME
    assert one < occluded.anchor(control, 99, -1, lo, hi) < occluded.ANCHOR_MAX
    control[90:100] = occluded.AWAY                                      # a late hand-over
    assert occluded.anchor(control, 99, -1, lo, hi) < 0.5
    assert occluded.anchor(control, 100, +1, lo, 100) == 0.5             # never past the half


def test_gaps_stop_at_the_half_edges():
    n = FIRST_HALF[2] + 10
    control = np.full(n, occluded.NOBODY, dtype=np.int8)
    missing = np.zeros(n, dtype=bool)
    _, lo, hi = FIRST_HALF
    missing[lo - 5:lo + 30] = True        # starts before the half: only its graded part counts
    control[lo + 30:lo + 40] = occluded.AWAY
    p, informed, gaps = occluded.bridge_gaps(control, missing)
    assert gaps[0][:2] == (lo, lo + 30) and gaps[0][2] == 0.5 and gaps[0][3] < 0.5
    assert np.isnan(p[:lo]).all() and (p[lo:lo + 30] < 0.5).all() and informed[lo:lo + 30].all()


def test_tag_reliability_learns_which_team_a_code_names():
    n = 1000
    control = np.full(n, occluded.NOBODY, dtype=np.int8)
    control[0:500] = occluded.HOME
    control[500:1000] = occluded.AWAY
    instances = [("ATTACK", occluded.HOME, 0, 100), ("ATTACK", occluded.AWAY, 600, 700),
                 ("ATTACK", occluded.HOME, 100, 200), ("DEFEND", occluded.AWAY, 200, 300),
                 ("DEFEND", occluded.HOME, 700, 800), ("DEFEND", occluded.HOME, 900, 950)]
    codes = occluded.tag_reliability(instances, control)
    prior = occluded.TAG_PRIOR
    assert codes["ATTACK"]["p"] == (3 + prior) / (3 + 2 * prior)
    assert codes["DEFEND"]["p"] == (0 + prior) / (3 + 2 * prior)
    tags = occluded.tag_logit(instances, codes, n)
    assert tags[50] > 0 and tags[250] > 0 and tags[650] < 0 and tags[750] < 0
    assert np.isnan(tags[400])


def test_set_pieces_and_teamless_tags_are_left_to_others():
    tags = pd.DataFrame({
        "code": ["SET PIECES", "BUILD UP", "BUILD UP", "PROGRESSION"],
        "team": ["H", None, "A", "H"],
        "start_frame": [10, 20, 30, -5],
        "end_frame": [40, 50, 60, 15],
    })
    assert occluded.tag_instances(tags, 100) == [("BUILD UP", occluded.AWAY, 30, 60),
                                                 ("PROGRESSION", occluded.HOME, 0, 15)]


def test_build_writes_only_on_missing_ball_frames_and_finds_the_team(small_match, use_data,
                                                                     tmp_path):
    use_data(small_match, tmp_path / "cache")
    table = frame_table.load()
    report = {}
    p_dead, p_home, weight = occluded.build(table, report=report)

    has = np.isfinite(p_home)
    assert np.isnan(p_dead).all()
    assert not (has & table.ball_visible).any() and not (has & ~table.graded).any()
    assert (weight[has] == occluded.WEIGHT).all() and (weight[~has] == 0).all()
    assert (p_home[has] >= occluded.P_MIN).all() and (p_home[has] <= occluded.P_MAX).all()
    assert report["gaps"] and report["covered"] <= report["missing"]

    # The synthetic tracking stops at small_match.n_frames; judge the frames before that.
    truth = small_match.truth[:table.n_frames]
    hidden = table.graded & ~table.ball_visible & np.isin(truth, (H, A)) & table.atd_present
    assert hidden.sum() > 100 and has[hidden].mean() > 0.9
    alive = has & hidden
    right = (p_home[alive] > 0.5) == (truth[alive] == H)
    assert right.mean() > 0.9
    assert report["codes"]["BUILD UP"]["p"] > 0.5            # the synthetic phases are honest

    again = occluded.build(table)
    assert evidence.content_hash(*again) == evidence.content_hash(p_dead, p_home, weight)
    assert "missing-ball gaps" in occluded.summary(report, p_home)


def test_selfcheck_hides_the_ball_and_reports_aggregates(small_match, use_data, tmp_path,
                                                         capsys):
    use_data(small_match, tmp_path / "cache")
    mask = np.zeros(400, dtype=bool)
    mask[0:300] = True
    hidden, count = selfcheck.hidden_stretches(mask, 50)
    assert count == 1 and hidden.sum() == 50 and hidden[125:175].all()

    scale = occluded.TAG_SCALE
    assert selfcheck.main(["--seconds", "0.4"]) == 0
    out = capsys.readouterr().out
    assert "hide 0.4 s" in out and "bridge + tags" in out and "accuracy" in out
    assert occluded.TAG_SCALE == scale                       # restored


def test_main_writes_valid_evidence(small_match, use_data, tmp_path, capsys):
    use_data(small_match, tmp_path / "cache")
    assert occluded.main([]) == 0
    loaded = evidence.read_evidence("occluded")
    assert loaded.meta["params"]["MEAN_HOLD_S"] == occluded.MEAN_HOLD_S
    assert "OK" in capsys.readouterr().out
