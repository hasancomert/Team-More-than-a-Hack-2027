"""The frame alignment: ATD frame_id - 1, round(seconds * 25), graded halves, homography rows."""

import random
import sys

import numpy as np
import pytest

from saha.common import alignment, frame_table, homography, paths, smart_tagging
from saha.common.constants import GRADED_HALVES, N_FRAMES
from saha.testing import synthetic

sys.path.insert(0, str(paths.REPO_ROOT / "starter"))
import load_smart_tagging  # noqa: E402  (official starter code, for equivalence checks)


def test_atd_frame_id_is_video_frame_plus_one():
    assert alignment.video_frame_of_atd(1) == 0
    assert alignment.video_frame_of_atd(199319) == 199318
    assert alignment.atd_frame_id_of(11185) == 11186
    ids = np.array([1, 2, 11186, 199319])
    assert np.array_equal(alignment.video_frame_of_atd(ids), ids - 1)
    assert np.array_equal(alignment.atd_frame_id_of(ids - 1), ids)


def test_seconds_to_frame_matches_starter_rounding():
    """Same expression as starter/load_smart_tagging.py, ties and all (half-to-even)."""
    rng = random.Random(0)
    seconds = [rng.uniform(0, 8000) for _ in range(20000)]
    seconds += [447.4, 447.42, 4549.68, 0.02, 0.06, 12.5 / 25, 1.5 / 25, 2.5 / 25]
    for s in seconds:
        assert alignment.video_frame_of_seconds(s) == int(round(s * 25))
    vector = alignment.video_frame_of_seconds(np.array(seconds))
    assert vector.tolist() == [int(round(s * 25)) for s in seconds]


def test_graded_halves_are_the_published_ones():
    assert GRADED_HALVES == ((1, 11185, 90651), (2, 110808, 190197))
    mask = alignment.graded_mask()
    assert mask.sum() == 158857
    assert not mask[11184] and mask[11185] and mask[90651] and not mask[90652]
    assert not mask[110807] and mask[110808] and mask[190197] and not mask[190198]
    half = alignment.half_of_frames()
    assert half[11185] == 1 and half[110808] == 2 and half[100000] == 0


def test_frame_table_row_is_atd_frame_id_minus_one(tmp_path, use_data):
    """Every ball x encodes its own frame_id, so any off-by-one shows up as a value mismatch."""
    match = synthetic.write_match(tmp_path / "data", n_frames=3000, seed=5,
                                  ball_x_from_frame_id=True)
    use_data(match, tmp_path / "cache")
    table = frame_table.load()
    rows = np.arange(3000)
    assert np.allclose(table.ball_xy[rows, 0], (rows + 1) / 1e6, rtol=0, atol=1e-9)
    assert table.atd_present[:3000].all() and not table.atd_present[3000:].any()
    assert np.isnan(table.ball_xy[3000:]).all()


def test_missing_atd_record_leaves_its_row_empty(tmp_path, use_data):
    match = synthetic.write_match(tmp_path / "data", n_frames=500, seed=6,
                                  ball_x_from_frame_id=True)
    raw = paths.asset_path("atd_raw", match.data_dir)
    lines = raw.read_text().splitlines()
    del lines[99]                                    # frame_id 100 -> video frame 99
    raw.write_text("\n".join(lines) + "\n")
    use_data(match, tmp_path / "cache")
    table = frame_table.load()
    assert not table.atd_present[99] and np.isnan(table.ball_xy[99]).all()
    assert table.atd_present[98] and table.atd_present[100]
    assert table.ball_xy[100, 0] == pytest.approx(101 / 1e6, abs=1e-9)
    assert table.meta["contiguous"] is False


def test_smart_tagging_frames_follow_the_starter(small_match, use_data, tmp_path):
    use_data(small_match, tmp_path / "cache")
    ours = smart_tagging.load()
    theirs = load_smart_tagging.load(data_dir=str(small_match.data_dir))
    assert len(ours) == len(theirs)
    assert ours["start_frame"].tolist() == [e["start_frame"] for e in theirs]
    assert ours["end_frame"].tolist() == [e["end_frame"] for e in theirs]
    kickoff = ours[ours["type"] == "Kick Off (Start)"].iloc[0]
    assert kickoff["start_frame"] <= 11185 <= kickoff["end_frame"]
    assert set(ours["team"].dropna()) <= {"H", "A"}
    throw_ins = ours[ours["type"] == "Throw-in"]
    assert throw_ins["team"].isna().all()           # every throw-in is N/A, as in the real file


def test_smart_tagging_rejects_unknown_team(tmp_path):
    path = tmp_path / "tags.xml"
    path.write_text("<file><ALL_INSTANCES><instance><ID>1</ID><start>1</start><end>2</end>"
                    "<code>BUILD UP</code><label><group>Team</group><text>Real Madrid</text>"
                    "</label></instance></ALL_INSTANCES></file>")
    with pytest.raises(ValueError, match="unknown Smart Tagging team"):
        smart_tagging.load(path=path)


def test_homography_rows_are_video_frames(small_match, use_data, tmp_path):
    use_data(small_match, tmp_path / "cache")
    matrices, error, valid = homography.load()
    assert matrices.shape == (N_FRAMES, 3, 3) and valid.all()
    x, y = homography.project(matrices[11185], 0.5, 0.5)
    back = homography.unproject(matrices[11185], x, y)
    assert np.allclose(back, (0.5, 0.5))
