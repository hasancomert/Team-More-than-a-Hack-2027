"""WS5: external data parsing, degradation and the scoring harness, on tiny synthetic inputs."""

import numpy as np

from saha.combine.decode import CombineParams
from saha.common import evidence
from saha.common.constants import A, D, H, X
from saha.external import build, degrade, dfl, harness

META = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
  <General MatchId="DFL-MAT-TEST" HomeTeamId="T-H" GuestTeamId="T-A" />
  <Environment PitchX="105.00" PitchY="68.00" />
  <Teams>
    <Team TeamId="T-H"><Players><Player PersonId="P1" /></Players></Team>
    <Team TeamId="T-A"><Players><Player PersonId="P2" /></Players></Team>
  </Teams>
</MatchInformation></PutDataRequest>
"""


def frameset(section, team, person, rows):
    lines = ['<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">'.format(section, team, person)]
    for n, x, y, extra in rows:
        lines.append('<Frame N="{}" X="{}" Y="{}" {}/>'.format(n, x, y, extra))
    lines.append("</FrameSet>")
    return "\n".join(lines)


def write_dfl(tmp_path):
    ball1 = [(10000 + i, 0.0, 0.0, 'Z="0" BallStatus="{}" BallPossession="{}"'.format(s, p))
             for i, (s, p) in enumerate([(0, 1), (1, 1), (1, 2), (1, 2)])]
    ball2 = [(100000 + i, 1.0, -1.0, 'Z="0" BallStatus="1" BallPossession="1"') for i in range(3)]
    body = "\n".join([
        frameset("firstHalf", "T-H", "P1", [(10000 + i, -52.5, 34.0, "") for i in range(4)]),
        frameset("firstHalf", "T-A", "P2", [(10001, 52.5, -34.0, "")]),
        frameset("firstHalf", "BALL", "B", ball1),
        frameset("secondHalf", "T-H", "P1", [(100000, 0.0, 0.0, "")]),
        frameset("secondHalf", "BALL", "B", ball2),
        frameset("firstHalf", "referee", "R1", [(10000, 0.0, 0.0, "")]),
    ])
    meta, tracking = tmp_path / "m.xml", tmp_path / "t.xml"
    meta.write_text(META, encoding="utf-8")
    tracking.write_text('<?xml version="1.0"?>\n<PutDataRequest><Positions>' + body +
                        "</Positions></PutDataRequest>", encoding="utf-8")
    return meta, tracking


def test_labels_from_ball():
    status = np.array([0, 1, 1, 1, -1], dtype=np.int8)
    possession = np.array([1, 1, 2, 0, 1], dtype=np.int8)
    assert dfl.labels_from_ball(status, possession).tolist() == [D, H, A, X, X]


def test_convert_layout(tmp_path):
    out = dfl.convert(*write_dfl(tmp_path))
    gap = dfl.HALF_GAP
    assert out["labels"].tolist() == [D, H, A, A] + [X] * gap + [H, H, H]
    assert out["half"].tolist() == [1] * 4 + [0] * gap + [2] * 3
    assert out["slot_person"].tolist() == ["P1", "P2"]        # the referee is not a slot
    assert out["slot_team"].tolist() == [0, 1]
    # Corner origin, y down: (-52.5, 34) is (0, 0); (52.5, -34) is (1, 1).
    assert np.allclose(out["players_xy"][0, 0], [0.0, 0.0])
    assert np.allclose(out["players_xy"][1, 1], [1.0, 1.0])
    assert np.isnan(out["players_xy"][0, 1]).all()


def toy_match(n_half=3000, seed=1):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.array([H, A, D, H, A, H], dtype=np.int8), n_half // 6)
    labels = np.concatenate([labels, np.full(10, X, np.int8), labels])
    half = np.concatenate([np.ones(n_half, np.int8), np.zeros(10, np.int8),
                           np.full(n_half, 2, np.int8)])
    n = len(labels)
    players = rng.random((n, 4, 2)).astype(np.float32)
    players[half == 0] = np.nan
    return dfl.ExternalMatch("toy", rng.random((n, 2)).astype(np.float32),
                             np.zeros(n, np.float32), players, np.array([0, 0, 1, 1], np.int8),
                             np.array(["a", "b", "c", "d"]), labels, half, {})


def test_degrade_hides_share_and_is_seeded():
    match = toy_match()
    config = degrade.Degradation(gap_median_s=4.0, gap_min_s=1.0, gap_max_s=20.0)
    table, labels = degrade.degrade(match, config)
    scored = labels != X
    assert 0.41 <= (~table.ball_visible)[scored].mean() < 0.6
    again, _ = degrade.degrade(match, config)
    assert np.array_equal(table.ball_visible, again.ball_visible)
    assert table.n_slots == 8                                 # 4 people x 2 halves
    assert table.ball_speed is None and table.ball_z is None
    clean, _ = degrade.degrade(match, degrade.NONE)
    assert clean.ball_visible.all() and clean.n_slots == 4


def oracle(labels):
    p_dead, p_home, weight = evidence.empty(len(labels))
    p_dead[:] = np.where(labels == D, 0.99, 0.01)
    p_home[:] = np.where(labels == H, 0.99, 0.01)
    weight[:] = 1.0
    return harness.Source("oracle", evidence.Evidence("oracle", *(evidence.quantise(a) for a in
                                                                  (p_dead, p_home, weight)), {}),
                          1.0)


def test_oracle_scores_perfectly():
    match = toy_match()
    codes = harness.decode([oracle(match.labels)], CombineParams(), match.half)
    assert (codes[match.half == 0] == X).all()
    assert harness.score(match.labels, codes)["spell_f1_mean"] == 1.0


def test_builders_run_on_external_table():
    match = toy_match()
    table, labels = degrade.degrade(match)
    entries = [{"name": "nearest", "module": "saha.nearest.build"},
               {"name": "missing", "module": "saha.no_such_package.build"}]
    rows, status = harness.evaluate_table(table, labels, entries, CombineParams(),
                                          sweep={"cost_switch": [10.0, 35.0]})
    assert status["nearest"] == "ok" and status["missing"].startswith("ModuleNotFoundError")
    assert [r["sources"] for r in rows] == ["all", "all", "nearest"]
    assert rows[0]["params"]["cost_switch"] == 10.0


def test_half_runs():
    assert harness.half_runs(np.array([0, 1, 1, 0, 2, 2, 2], np.int8)) == [(1, 3), (4, 7)]


def test_label_stats():
    stats = build.label_stats(toy_match().labels)
    assert stats["stoppages"] == 2 and stats["possessions"] == 10
