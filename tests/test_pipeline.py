"""The whole pipeline on a full-length synthetic match: runs, validates, recovers the truth,
and writes the same bytes twice."""

import json

from saha import pipeline
from saha.common import submission


def test_pipeline_end_to_end_is_correct_and_deterministic(full_match, use_data, tmp_path):
    use_data(full_match, tmp_path / "cache")
    out = tmp_path / "submissions" / "SAHA.csv"
    assert pipeline.main(["--out", str(out), "--no-score"]) == 0
    first = submission.sha256_of(out)

    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["sha256"] == first
    assert set(summary["evidence"]) == {"nearest", "deadball", "hiddendead", "visible",
                                        "occluded", "shape", "onball"}
    assert summary["match"]["x_rate"] == 0.0

    # The synthetic story is easy: the plumbing is right only if the truth comes back.
    result = submission.score(submission.read_codes(out), gt_codes=full_match.truth,
                              bootstrap=0)
    assert result["spell_f1_mean"] > 0.85
    assert result["team_acc_both_alive"] > 0.95

    assert pipeline.main(["--out", str(out), "--no-score", "--fresh"]) == 0
    assert submission.sha256_of(out) == first
