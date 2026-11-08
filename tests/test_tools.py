"""The WS1 tools on synthetic data: data checks, EDA, the metric study, experiment records,
the self-check."""

import datetime
import json
import subprocess
import sys

import numpy as np
import pytest

from saha.common import epts, frame_table, kinematics, paths, submission
from saha.common import experiments
from saha.common.constants import A, D, H, X
from saha import pipeline
from saha.tools import data_checks, eda, metric_behaviour as mb, selfcheck, techdoc


def test_data_checks_on_a_synthetic_bundle(full_match, use_data, tmp_path):
    use_data(full_match, tmp_path / "cache")
    starter_out = tmp_path / "starter_baselines"
    subprocess.run([sys.executable, str(paths.REPO_ROOT / "starter" / "baseline.py"),
                    "--out", str(starter_out)], check=True, capture_output=True)
    report = data_checks.Report()
    data_checks.check_starter(report)
    data_checks.check_baseline_files(report, starter_out)
    meta = data_checks.check_metadata(report)
    table = data_checks.check_frame_table(report)
    data_checks.check_our_baseline(report, meta, table, starter_out)
    rows = {row["check"]: row["ok"] for row in report.rows}
    assert rows["starter: ATD alignment"] and rows["starter: homography rows"]
    assert rows["starter: Smart Tagging time base"]
    assert rows["starter/baseline.py == results/"]
    assert rows["frame table: one record per frame"]
    assert rows["frame table: line order == <Player>"]
    assert rows["our parser reproduces the baseline"]       # frame for frame, all 4 windows
    assert not rows["EPTS: home team is Barcelona"]          # synthetic teams, as expected


def test_eda_writes_aggregates_only(full_match, use_data, tmp_path):
    use_data(full_match, tmp_path / "cache")
    out = tmp_path / "docs" / "EDA.md"
    assert eda.main(["--out", str(out)]) == 0
    text = out.read_text()
    for heading in ("## Coverage", "## Ball-missing stretches", "## Ball outside the pitch",
                    "## Player speed when the ball is dead", "## Smart Tagging restarts",
                    "## Track slots"):
        assert heading in text
    assert len(text.splitlines()) < 120                      # a summary, not a dump
    assert sorted(p.name for p in (tmp_path / "docs" / "figures").iterdir()) == [
        "eda_ball_gaps.png", "eda_speed_dead_alive.png", "eda_tracklets.png"]


@pytest.fixture(scope="module")
def toy():
    """One graded run: A 100, H 300, A 50, D 100, A 200, H 250, with X elsewhere."""
    truth = np.full(2000, X, dtype=np.int8)
    position = 100
    for code, length in ((A, 100), (H, 300), (A, 50), (D, 100), (A, 200), (H, 250)):
        truth[position:position + length] = code
        position += length
    return truth, submission.metrics().valid_runs(truth)


def spells(codes, runs):
    return submission.metrics().possession_spells(codes, runs)


def test_perturbations_do_what_they_say(toy):
    truth, runs = toy
    rng = np.random.default_rng(0)
    late = mb.late(truth, runs, rng, 10)
    assert late[110] == A and late[200] == A and late[210] == H
    assert len(spells(mb.split_blip(truth, runs, rng, (1, 25)), runs)) == 5 + 2
    assert len(spells(mb.split_dead(truth, runs, rng, (1, 25)), runs)) == 5 + 1
    merged = mb.miss_possession(truth, runs, np.random.default_rng(1), 5)
    assert len(spells(merged, runs)) < 5
    assert len(spells(mb.drop_stoppage(truth, runs, rng, 1), runs)) == 4   # A D A -> one A
    assert len(spells(mb.jitter(truth, runs, rng, 5), runs)) == 5
    held = mb.abstain_middle(truth, runs, rng, 20)
    assert (held == X).sum() > 0
    assert submission.metrics().evaluate(truth, held)["spell_f1_mean"] == 1.0


def test_tiled_truth_fills_both_halves():
    tiled = mb.tiled_truth(submission.public_gt())
    assert (tiled[:11185] == X).all() and (tiled[11185:90652] != X).all()
    assert (tiled[110808:190198] != X).all() and (tiled[190198:] == X).all()


def test_experiment_records(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setenv("SAHA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(paths, "submissions_dir", lambda: tmp_path / "submissions")
    (tmp_path / "submissions").mkdir()
    (tmp_path / "submissions" / "SAHA.summary.json").write_text(json.dumps({
        "git_commit": "0123456789abcdef", "config": str(paths.CONFIGS_DIR / "pipeline.toml"),
        "public": {"spell_f1_mean": 0.41234, "spell_bootstrap_low": 0.3,
                   "spell_bootstrap_high": 0.52, "macro_f1": 0.66},
        "match": {"n_possessions": 251, "mean_duration_s": 14.04, "d_rate": 0.2111}}))
    path = experiments.new("ws2", "restarts-v1", author="someone",
                           today=datetime.date(2026, 11, 3))
    assert path.name == "20261103_ws2_restarts-v1.md"
    fields = experiments.read_front_matter(path)
    assert fields["public_spell_f1"] == "0.4123" and fields["n_possessions"] == "251"
    assert fields["config"] == "configs/pipeline.toml" and fields["verdict"] == "open"
    assert experiments.problems() == []
    text, count = experiments.table()
    assert count == 1 and "restarts-v1" in text
    (tmp_path / "experiments" / "notes.md").write_text("free text")
    assert experiments.problems()
    with pytest.raises(FileExistsError):
        experiments.new("ws2", "restarts-v1", today=datetime.date(2026, 11, 3))


def test_frame_table_detects_stale_cache(small_match, use_data, tmp_path):
    use_data(small_match, tmp_path / "cache")
    frame_table.load()
    assert frame_table.is_current()[0]
    meta = paths.asset_path("atd_meta", small_match.data_dir)
    original = meta.read_text()
    try:
        meta.write_text(original.replace("<FrameRate>25.0</FrameRate>",
                                         "<FrameRate>25.0</FrameRate>   "))
        ok, reason = frame_table.is_current()
        assert not ok and "changed" in reason
    finally:
        meta.write_text(original)
    assert epts.read_metadata(meta).frame_rate == 25


def test_speed_ratio_tells_metres_per_second_from_km_per_hour(full_match, use_data, tmp_path):
    use_data(full_match, tmp_path / "cache")
    table = frame_table.load()
    assert table.meta["version"] == frame_table.VERSION and table.ball_speed is not None
    ratio, n = data_checks.speed_ratio(table.players_xy_m(), table.players_speed)
    assert n > 1000 and 0.9 < ratio < 1.1
    ratio, _ = data_checks.speed_ratio(table.players_xy_m(), table.players_speed * 3.6)
    assert 3.3 < ratio < 3.9
    report = data_checks.Report()
    data_checks.check_speed_units(report, table)
    assert [(row["check"], row["ok"]) for row in report.rows] == [("speed unit: players", True)]


def test_ball_report_on_a_synthetic_match(full_match, use_data, tmp_path):
    use_data(full_match, tmp_path / "cache")
    table = frame_table.load()
    lines = data_checks.ball_report(table)
    assert len(lines) == 8 and lines[-1].startswith("  nearest player (m): p10")
    seen = table.graded & table.ball_visible
    speed = kinematics.displacement_speed(table.ball_xy_m())
    channel = table.ball_speed.astype(np.float64)
    ratio, n = data_checks.channel_ratio(channel, speed, seen)
    assert n > 1000 and 0.9 < ratio < 1.1             # synthetic speeds follow the positions
    ratio, _ = data_checks.channel_ratio(channel * 0.2, speed, seen)
    assert 0.18 < ratio < 0.22
    assert data_checks.main(["--speed"]) == 0


def test_selfcheck_recovers_held_out_touches(full_match, use_data, tmp_path):
    use_data(full_match, tmp_path / "cache")
    assert pipeline.main(["--out", str(tmp_path / "SAHA.csv"), "--no-score"]) == 0
    _, rows = selfcheck.run(values=(10.0, 60.0))
    assert len(rows) == 2 and rows[0]["touches"] > 100
    counts = list(rows[0]["counts"].values())
    assert counts == sorted(counts, reverse=True) and counts[-1] > 0
    assert rows[0]["recovered"] > 0.7 and rows[0]["recovered"] >= rows[1]["recovered"]
    assert {"n_possessions", "short_possessions_lt_2s", "d_rate"} <= set(rows[0])
    assert selfcheck.main(["--values", "20"]) == 0



def test_techdoc_puts_the_package_sections_in_order_as_appendices(tmp_path):
    (tmp_path / "saha_main.md").write_text("# SAHA\n\n## Summary\n\ntext\n\n## Appendices\n")
    (tmp_path / "ws3_visible.md").write_text(
        "# WS3: ball tracked\n\n## Approach\n\n```bash\n# a shell comment\n```\n")
    (tmp_path / "ws1_pipeline.md").write_text("# WS1: pipeline\n\n## Approach\n\nbody\n")
    (tmp_path / "README.md").write_text("# how to assemble\n")
    (tmp_path / "ws9_other.md").write_text("# not a package\n")
    text = techdoc.assemble(tmp_path)
    assert text.startswith("# SAHA\n")
    assert text.index("## Appendix A: WS1: pipeline") < text.index("## Appendix C: WS3: ball")
    assert text.count("### Approach") == 2 and "\n# a shell comment\n" in text
    assert "how to assemble" not in text and "not a package" not in text


def test_techdoc_assembles_the_repository_sections(tmp_path):
    out = tmp_path / "SAHA.md"
    assert techdoc.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## Appendix A: WS1" in text and "## Appendices" in text
