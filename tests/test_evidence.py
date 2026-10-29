"""The evidence interface and its validator."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from saha.common import evidence, paths
from saha.common.constants import N_FRAMES


def arrays():
    p_dead, p_home, weight = evidence.empty()
    p_dead[11185:12000] = 0.123456
    p_home[11185:11500] = 0.98767
    weight[11185:12000] = 0.5
    return p_dead, p_home, weight


def test_round_trip_quantises_and_hashes(tmp_path, monkeypatch):
    monkeypatch.setenv("SAHA_CACHE_DIR", str(tmp_path))
    path = evidence.write_evidence("demo", *arrays(), producer="test", params={"a": 1})
    assert path == tmp_path / "evidence" / "demo.parquet"
    loaded = evidence.read_evidence("demo")
    assert loaded.p_dead.dtype == np.float32
    assert loaded.p_dead[11185] == np.float32(0.1235)          # quantised to 1e-4
    assert loaded.p_home[11185] == np.float32(0.9877)
    assert np.isnan(loaded.p_home[0]) and loaded.weight[0] == 0
    assert loaded.meta["producer"] == "test" and loaded.meta["params"] == {"a": 1}
    assert loaded.meta["content_sha256"] == loaded.content_hash()
    again = evidence.write_evidence("demo", *arrays())
    assert evidence.read_evidence(again).content_hash() == loaded.content_hash()


@pytest.mark.parametrize("breakage, message", [
    (lambda d, h, w: (d[:-1], h, w), "rows"),
    (lambda d, h, w: (np.where(np.arange(N_FRAMES) == 5, 1.5, d), h, w), "outside [0, 1]"),
    (lambda d, h, w: (d, h, np.where(np.arange(N_FRAMES) == 5, -1.0, w)), "negative"),
    (lambda d, h, w: (d, h, np.full(N_FRAMES, np.nan)), "NaN or infinite"),
    (lambda d, h, w: (d, np.where(np.arange(N_FRAMES) == 5, np.inf, h), w), "infinite"),
])
def test_invalid_arrays_are_rejected(tmp_path, monkeypatch, breakage, message):
    monkeypatch.setenv("SAHA_CACHE_DIR", str(tmp_path))
    with pytest.raises(evidence.EvidenceError, match=message.replace("[", r"\[").replace(
            "]", r"\]")):
        evidence.write_evidence("bad", *breakage(*arrays()))


def test_names_are_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("SAHA_CACHE_DIR", str(tmp_path))
    with pytest.raises(evidence.EvidenceError, match="must match"):
        evidence.write_evidence("Bad-Name", *arrays())


def test_files_written_by_hand_are_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("SAHA_CACHE_DIR", str(tmp_path))
    p_dead, p_home, weight = arrays()
    base = {"frame": np.arange(N_FRAMES, dtype=np.int32), "p_dead": p_dead,
            "p_home": p_home, "weight": weight}
    paths.evidence_dir().mkdir(parents=True)

    pq.write_table(pa.table({**base, "debug": weight}), paths.evidence_path("extra"))
    assert not evidence.validate_file("extra")[0]
    pq.write_table(pa.table({**base, "p_home": p_home.astype(np.float64)}),
                   paths.evidence_path("dtype"))
    ok, lines = evidence.validate_file("dtype")
    assert not ok and "float32" in "\n".join(lines)
    pq.write_table(pa.table({**base, "frame": base["frame"][::-1].copy()}),
                   paths.evidence_path("order"))
    assert not evidence.validate_file("order")[0]
    pq.write_table(pa.table(base), paths.evidence_path("good"))
    ok, lines = evidence.validate_file("good")
    assert ok and "OK" in lines[0]
    assert evidence.main([]) == 1                   # three bad files among four
    assert evidence.main(["good"]) == 0


def test_home_overlap_counts_shared_graded_frames():
    def make(name, start, stop, weight=1.0):
        p_dead, p_home, w = evidence.empty()
        p_home[start:stop] = 0.7
        w[start:stop] = weight
        return evidence.Evidence(name, p_dead, p_home, w, {})

    visible = make("visible", 11185, 11300)
    occluded = make("occluded", 11290, 11400)
    assert evidence.home_overlap(visible, occluded) == 10
    assert evidence.home_overlap(visible, make("occluded", 11300, 11400)) == 0
    assert evidence.home_overlap(visible, make("occluded", 11185, 11300, weight=0.0)) == 0
    assert evidence.home_overlap(make("a", 0, 100), make("b", 0, 100)) == 0   # ungraded frames
