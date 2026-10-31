"""The EPTS reader against itself, kloppy, and the official baseline's own parser."""

import sys
import warnings

import numpy as np
import pytest

from saha.common import epts, frame_table, paths
from saha.common.constants import N_FRAMES, PITCH_LENGTH_M, PITCH_WIDTH_M
from saha.nearest.build import nearest_team

sys.path.insert(0, str(paths.REPO_ROOT / "starter"))
import baseline  # noqa: E402  (official starter code)


@pytest.fixture(scope="module")
def parsed(small_match):
    meta_path = paths.asset_path("atd_meta", small_match.data_dir)
    raw_path = paths.asset_path("atd_raw", small_match.data_dir)
    meta = epts.read_metadata(meta_path)
    return meta, meta_path, raw_path, epts.read_raw(raw_path, meta)


def test_metadata(parsed):
    meta, *_ = parsed
    assert meta.frame_rate == 25 and meta.field_size == (105.0, 68.0)
    assert meta.home_team_id == meta.teams[0].team_id
    assert len(meta.players) == 4 and len(meta.specs) == 1
    assert meta.specs[0].player_order() == meta.player_ids()
    assert "none" in epts.channels_report(meta).splitlines()[-1]


def test_fast_and_exact_paths_agree(parsed):
    meta, _, raw_path, fast = parsed
    exact = epts.read_raw(raw_path, meta, force_exact=True)
    assert fast.path == "fast" and exact.path == "exact"
    assert np.array_equal(fast.frame_id, exact.frame_id)
    for channel in fast.players:
        assert np.array_equal(fast.players[channel], exact.players[channel], equal_nan=True)
    for channel in fast.ball:
        assert np.array_equal(fast.ball[channel], exact.ball[channel], equal_nan=True)


def test_values_equal_kloppy(parsed):
    meta, meta_path, raw_path, raw = parsed
    from kloppy import metrica
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = metrica.load_tracking_epts(meta_data=str(meta_path), raw_data=str(raw_path),
                                           coordinates="metrica").to_df()
    assert np.array_equal(frame["frame_id"].to_numpy(), raw.frame_id)
    assert np.array_equal(frame["ball_x"].to_numpy(), raw.ball["x"], equal_nan=True)
    for slot, player_id in enumerate(raw.slot_player_ids):
        assert np.array_equal(frame[player_id + "_x"].to_numpy(), raw.players["x"][:, slot],
                              equal_nan=True)
        assert np.array_equal(frame[player_id + "_s"].to_numpy(), raw.players["s"][:, slot],
                              equal_nan=True)


def test_nearest_team_equals_official_baseline(parsed, small_match, use_data, tmp_path):
    """Our parser + our nearest-player code reproduce starter/baseline.py frame for frame."""
    meta, meta_path, raw_path, raw = parsed
    official = baseline.nearest_player_labels(str(raw_path), baseline.team_of_each_track(
        str(meta_path)), N_FRAMES, (PITCH_LENGTH_M, PITCH_WIDTH_M))

    scale = np.array([PITCH_LENGTH_M, PITCH_WIDTH_M])
    ball = np.stack([raw.ball["x"], raw.ball["y"]], axis=-1) * scale
    players = np.stack([raw.players["x"], raw.players["y"]], axis=-1) * scale
    slot_team = (np.array([p.team_id for p in meta.players]) != meta.home_team_id).astype(np.int8)
    ours = np.full(N_FRAMES, -1, dtype=np.int8)
    ours[raw.frame_id - 1] = nearest_team(ball, players, slot_team)
    assert np.array_equal(ours, official)

    use_data(small_match, tmp_path / "cache")      # and through the float32 frame table
    table = frame_table.load()
    from_table = nearest_team(table.ball_xy_m(), table.players_xy_m(), table.slot_team)
    assert np.array_equal(from_table, official)


def test_trailing_separators_take_the_exact_path(parsed, tmp_path):
    meta, _, raw_path, fast = parsed
    lines = raw_path.read_text().splitlines()
    variant = tmp_path / "trailing.txt"
    variant.write_text("\n".join(line.replace(":", ";:", 2).replace(";:", ":", 1) + ","
                                 for line in lines) + "\n")
    read = epts.read_raw(variant, meta)
    assert read.path == "exact"
    assert np.array_equal(read.players["x"], fast.players["x"], equal_nan=True)
    assert np.array_equal(read.ball["y"], fast.ball["y"], equal_nan=True)


def test_short_line_is_an_error_that_does_not_leak_values(parsed, tmp_path):
    meta, _, raw_path, _ = parsed
    lines = raw_path.read_text().splitlines()
    lines[7] = lines[7].rsplit(":", 1)[0]          # drop the ball group of frame_id 8
    broken = tmp_path / "broken.txt"
    broken.write_text("\n".join(lines) + "\n")
    with pytest.raises(epts.EptsError) as caught:
        epts.read_raw(broken, meta)
    message = str(caught.value)
    assert "line 8" in message
    assert not any(token in message for token in lines[7].replace(":", ",").split(",")
                   if len(token) > 3)


def test_numbers_written_as_floats_in_the_metadata(small_match, tmp_path):
    """The delivered metadata writes FrameRate as '25.0'."""
    text = paths.asset_path("atd_meta", small_match.data_dir).read_text()
    assert "<FrameRate>25.0</FrameRate>" in text
    variant = tmp_path / "meta.xml"
    variant.write_text(text.replace('startFrame="1"', 'startFrame="1.0"'))
    meta = epts.read_metadata(variant)
    assert meta.frame_rate == 25 and isinstance(meta.frame_rate, int)
    assert meta.specs[0].start_frame == 1
    variant.write_text(text.replace("<FrameRate>25.0</FrameRate>", "<FrameRate>fast</FrameRate>"))
    with pytest.raises(epts.EptsError, match="FrameRate"):
        epts.read_metadata(variant)


def test_lines_beyond_the_declared_range_use_the_nearest_layout(parsed, tmp_path):
    meta, meta_path, raw_path, full = parsed
    last = int(full.frame_id.max())
    variant = tmp_path / "short_spec.xml"
    variant.write_text(meta_path.read_text().replace('endFrame="{}"'.format(last),
                                                     'endFrame="{}"'.format(last - 10)))
    short = epts.read_raw(raw_path, epts.read_metadata(variant))
    assert short.outside_spec == 10 and full.outside_spec == 0
    assert np.array_equal(short.players["x"], full.players["x"], equal_nan=True)
