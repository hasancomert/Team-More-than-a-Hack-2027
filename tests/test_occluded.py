"""WS4 (saha.occluded): the bridge over missing-ball gaps and the Smart Tagging phase teams.

Synthetic data only (CLAUDE.md, rule 3).
"""

import numpy as np
import pandas as pd

from saha.combine.decode import CombineParams
from saha.common import alignment, evidence, frame_table
from saha.common.constants import A, H, X
from saha.external import degrade, dfl, harness
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
    half = alignment.half_of_frames(n)
    control = np.full(n, occluded.NOBODY, dtype=np.int8)
    missing = np.zeros(n, dtype=bool)
    _, lo, hi = FIRST_HALF
    missing[lo - 5:lo + 30] = True        # starts before the half: only its graded part counts
    control[lo + 30:lo + 40] = occluded.AWAY
    p, informed, gaps = occluded.bridge_gaps(control, missing, half)
    assert gaps[0][:2] == (lo, lo + 30) and gaps[0][2] == 0.5 and gaps[0][3] < 0.5
    assert np.isnan(p[:lo]).all() and (p[lo:lo + 30] < 0.5).all() and informed[lo:lo + 30].all()


def test_gaps_follow_the_half_array_not_this_match():
    half = np.array([0] * 5 + [1] * 20 + [2] * 20 + [0] * 5, dtype=np.int8)
    assert occluded.half_runs(half) == [(5, 25), (25, 45)]          # touching halves stay apart
    control = np.full(len(half), occluded.NOBODY, dtype=np.int8)
    missing = np.zeros(len(half), dtype=bool)
    control[18:20] = occluded.HOME
    missing[20:30] = True                 # crosses from half 1 into half 2: two gaps
    control[30:32] = occluded.AWAY
    p, informed, gaps = occluded.bridge_gaps(control, missing, half)
    assert [g[:2] for g in gaps] == [(20, 25), (25, 30)]
    assert (p[20:25] > 0.5).all() and (p[25:30] < 0.5).all() and informed[20:30].all()
    assert not occluded.bridge_gaps(control, missing, np.zeros_like(half))[2]   # no half, no gap


def tag_frame(code, team):
    return pd.DataFrame({"code": [code], "team": [team], "start_frame": [10], "end_frame": [20]})


def tag_value(code, team):
    """Home log-odds of one tag on a frame it covers (NaN: no opinion)."""
    tags = tag_frame(code, team)
    return occluded.tag_logit(occluded.tag_instances(tags, 100), 100)[15]


def test_each_code_has_a_fixed_side_from_its_definition():
    value = float(occluded.logit(occluded.TAG_P))
    assert tag_value("BUILD UP", "H") == value > 0                  # with the ball, by home
    assert tag_value("DEFENDING IN MIDDLE THIRD", "H") == -value < 0   # without, by home
    assert tag_value("DEFENSIVE TRANSITION", "A") == value > 0      # without, by away: home has it
    assert tag_value("LONG BALL", "A") == -value
    assert np.isnan(tag_value("GOALS", "H"))
    assert np.isnan(tag_value("SET PIECES", "H"))
    assert np.isnan(tag_value("SOMETHING NEW", "H"))
    assert np.isnan(tag_value("BUILD UP", None))
    assert all(side in (occluded.WITH_BALL, occluded.WITHOUT_BALL)
               for side in occluded.TAG_SIDE.values())


def test_agreeing_tags_are_averaged_not_summed():
    instances = [("BUILD UP", occluded.HOME, 0, 50), ("PROGRESSION", occluded.HOME, 25, 75),
                 ("DEFENDING IN DEFENSIVE THIRD", occluded.AWAY, 40, 60)]
    tags = occluded.tag_logit(instances, 100)
    value = float(occluded.logit(occluded.TAG_P))
    np.testing.assert_allclose(tags[[10, 30, 45, 70]], value)      # one, two or three: the same
    assert np.isnan(tags[90])


def test_tag_counts_report_sides_and_left_out_codes():
    tags = pd.DataFrame({
        "code": ["BUILD UP", "BUILD UP", "DEFENSIVE TRANSITION", "SET PIECES", "GOALS",
                 "SOMETHING NEW"],
        "team": ["H", None, "A", "H", "A", "H"],
        "start_frame": [0] * 6, "end_frame": [10] * 6,
    })
    codes, left_out = occluded.tag_counts(tags)
    assert codes["BUILD UP"] == {"instances": 1, "side": "with"}
    assert codes["DEFENSIVE TRANSITION"] == {"instances": 1, "side": "without"}
    assert left_out == {"SET PIECES": 1, "GOALS": 1, "N/A": 1, "unknown": 1}
    assert all(set(entry) == {"instances", "side"} for entry in codes.values())  # no agreement
    assert occluded.tag_counts(None)[1] == {"SET PIECES": 0, "GOALS": 0, "N/A": 0, "unknown": 0}


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


def external_match(n_half=2000):
    """A tiny dfl.ExternalMatch, like toy_match in tests/test_external.py, with halves at frames
    0-1999 and 2010-4009 (not this match's). The ball is random (never a touch) except around two
    hidden stretches: a slow ball at a home player's feet on both sides of 600-800 (half 1), and
    at an away player's feet on both sides of 3000-3200 (half 2)."""
    rng = np.random.default_rng(7)
    half = np.concatenate([np.ones(n_half, np.int8), np.zeros(10, np.int8),
                           np.full(n_half, 2, np.int8)])
    n = len(half)
    labels = np.where(half > 0, H, X).astype(np.int8)
    labels[2010:] = A
    ball = rng.random((n, 2)).astype(np.float32)
    homes, aways = (0.5, 0.5), (0.3, 0.7)
    players = np.empty((n, 4, 2), np.float32)
    players[:] = np.array([homes, (0.1, 0.1), aways, (0.9, 0.9)], np.float32)
    players[half == 0] = np.nan
    ball[500:900] = (0.505, 0.5)                    # 0.5 m from the home player, standing still
    ball[2900:3300] = (0.305, 0.7)                  # the same at the away player's feet
    ball[600:800] = np.nan                          # hidden by hand
    ball[3000:3200] = np.nan
    match = dfl.ExternalMatch("toy-ws4", ball, np.zeros(n, np.float32), players,
                              np.array([0, 0, 1, 1], np.int8), np.array(["a", "b", "c", "d"]),
                              labels, half, {})
    return degrade.degrade(match, degrade.Degradation(hidden_share=0.0))


def test_build_runs_on_an_external_table_without_our_tags(monkeypatch):
    table, labels = external_match()
    assert "external" in table.meta and not table.ball_visible[600:800].any()

    def forbidden(*args, **kwargs):
        raise AssertionError("an external table must never get this match's Smart Tagging")

    monkeypatch.setattr(occluded.smart_tagging, "load", forbidden)
    report = {}
    p_dead, p_home, weight = occluded.build(table, report=report)

    has = np.isfinite(p_home)
    assert np.isnan(p_dead).all()
    assert not (has & ~(table.graded & ~table.ball_visible)).any()  # graded, ball missing only
    assert has[600:800].all() and has[3000:3200].all()
    assert (p_home[600:800] > 0.5).all() and (p_home[3000:3200] < 0.5).all()
    assert (weight[has] == occluded.WEIGHT).all() and (weight[~has] == 0).all()
    assert report["tagged"] == 0 and len(report["gaps"]) == 2
    assert "left out: 0 instances" in occluded.summary(report, p_home)

    entries = [{"name": "occluded", "module": "saha.occluded.build"}]
    rows, status = harness.evaluate_table(table, labels, entries, CombineParams(), singles=False)
    assert status["occluded"] == "ok" and rows[0]["sources"] == "all"


def test_main_writes_valid_evidence(small_match, use_data, tmp_path, capsys):
    use_data(small_match, tmp_path / "cache")
    assert occluded.main([]) == 0
    loaded = evidence.read_evidence("occluded")
    assert loaded.meta["params"]["MEAN_HOLD_S"] == occluded.MEAN_HOLD_S
    assert "OK" in capsys.readouterr().out
