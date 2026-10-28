"""The WS1 tools on synthetic data: data checks, EDA, the metric study, experiment records."""

import datetime
import json
import subprocess
import sys

import numpy as np
import pytest

from saha.common import epts, frame_table, paths, submission
from saha.common import experiments
from saha.common.constants import A, D, H, X
from saha.tools import data_checks, eda, metric_behaviour as mb


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
        meta.write_text(original.replace("<FrameRate>25</FrameRate>",
                                         "<FrameRate>25</FrameRate>   "))
        ok, reason = frame_table.is_current()
        assert not ok and "changed" in reason
    finally:
        meta.write_text(original)
    assert epts.read_metadata(meta).frame_rate == 25
