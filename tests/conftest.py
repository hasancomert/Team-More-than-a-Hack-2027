"""Shared fixtures. Tests run on synthetic data only, in temporary directories.

The real cache/ and data/ are never touched: SAHA_CACHE_DIR and MTAH_DATA_DIR are pointed at
pytest temp dirs for the whole session before any test runs.
"""

import os

import pytest

from saha.testing import synthetic


@pytest.fixture(scope="session", autouse=True)
def isolated_dirs(tmp_path_factory):
    saved = {k: os.environ.get(k) for k in ("SAHA_CACHE_DIR", "MTAH_DATA_DIR",
                                             "SAHA_DEADBALL_MODEL")}
    os.environ["SAHA_CACHE_DIR"] = str(tmp_path_factory.mktemp("cache"))
    os.environ["MTAH_DATA_DIR"] = str(tmp_path_factory.mktemp("no_data_here"))
    # WS2's classifier learned from real matches with twenty-two players; on the synthetic match
    # (four players) it calls almost every frame dead, and its decision outweighs the rest. So
    # the pipeline tests run WS2's builder without it, with no opinion; tests/test_deadball.py
    # tests the package.
    os.environ["SAHA_DEADBALL_MODEL"] = "none"
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def small_match(tmp_path_factory):
    """13,000 frames: the start of the first graded half (from 11185), 2+2 slots."""
    return synthetic.write_match(tmp_path_factory.mktemp("small") / "data", n_frames=13000,
                                 seed=1)


@pytest.fixture(scope="session")
def full_match(tmp_path_factory):
    """Every frame of the video, 2+2 slots. Slower; used by the pipeline test."""
    return synthetic.write_match(tmp_path_factory.mktemp("full") / "data", seed=2)


@pytest.fixture
def use_data(monkeypatch):
    """Point MTAH_DATA_DIR (and a fresh cache) at a synthetic match for one test."""
    def point(match, cache):
        monkeypatch.setenv("MTAH_DATA_DIR", str(match.data_dir))
        monkeypatch.setenv("SAHA_CACHE_DIR", str(cache))
        return match
    return point
