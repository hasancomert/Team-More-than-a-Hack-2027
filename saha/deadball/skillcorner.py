"""SkillCorner open data: the labelled external matches the dead-ball classifier is trained on.

Twenty A-League 2024/25 matches, broadcast tracking at 10 fps, MIT licence, (c) SkillCorner,
github.com/SkillCorner/opendata. Downloaded once from a fixed commit of that repository to
cache/external/skillcorner/ (gitignored), about 1.8 GB; set SAHA_SKILLCORNER_DIR to use a copy
that is already on disk. Each tracking file is checked against the SHA256 its Git LFS pointer
publishes.

load(match) resamples to 25 fps, so every frame-based constant keeps its meaning, and returns:

    ball      (n, 3)      x, y in 0-1 pitch units, height in metres; NaN where SkillCorner did not
                          detect the ball (its extrapolated positions are not observations)
    players   (n, k, 3)   x, y in 0-1 pitch units, speed in m/s from the positions
    team      (k,)        0 home, 1 away
    alive     (n,)        1 inside a phase of play; SkillCorner defines phases only while the ball
                          is in play, so the gaps between them are dead ball
    valid     (n,)        frames with tracking: the broadcast cuts to replays and close-ups during
                          stoppages, and those frames have no data at all, which would give the
                          dead ball away; train only where this is True
    runs      (2, 2)      [start, stop) of each half
    pitch     (2,)        length, width in metres
"""

import csv
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from saha.common import paths

SOURCE_FPS, FPS = 10, 25
MIN_PLAYERS = 10            # a frame with fewer tracked players is a replay or a close-up
CONVERTED_VERSION = 2

REPOSITORY = "SkillCorner/opendata"
COMMIT = "4340d274572876239c154c90bc507a9b3250a656"   # master on 14 Sep 2026
RAW_URL = "https://raw.githubusercontent.com/{}/{}/data/matches/{{m}}/{{m}}_{{f}}".format(
    REPOSITORY, COMMIT)
LFS_URL = "https://media.githubusercontent.com/media/{}/{}/data/matches/{{m}}/{{m}}_{{f}}".format(
    REPOSITORY, COMMIT)
USER_AGENT = "saha-ws2-deadball/1"
LICENCE = "MIT"
FILES = ("match.json", "phases_of_play.csv")
TRACKING = "tracking_extrapolated.jsonl"
MATCHES = ("1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846",
           "1986691", "1996435", "1996436", "2006229", "2006363", "2007448", "2007721",
           "2010085", "2011166", "2013725", "2015213", "2016236", "2017461")


def raw_dir():
    env = os.environ.get("SAHA_SKILLCORNER_DIR")
    return Path(env).expanduser().resolve() if env else (
        paths.cache_dir() / "external" / "skillcorner")


def converted_path(match):
    return paths.cache_dir() / "external" / "skillcorner_25fps" / "{}.npz".format(match)


def _file(match, name):
    return raw_dir() / match / "{}_{}".format(match, name)


def _sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=120)


def _fetch(url, target, attempts=3):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with _get(url) as response, open(partial, "wb") as out:
                expected = int(response.headers.get("Content-Length") or 0)
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            size = partial.stat().st_size
            # A dropped connection can end the stream early without an error: check the length.
            if size == 0 or (expected and size != expected):
                raise OSError("got {} of {} bytes".format(size, expected))
            partial.replace(target)
            return
        except OSError as error:
            print("  attempt {} failed: {}".format(attempt, error), flush=True)
            if partial.exists():
                partial.unlink()
    raise OSError("could not download {}".format(url))


def _lfs_pointer(match):
    """(sha256, size) that the repository publishes for a match's tracking file."""
    with _get(RAW_URL.format(m=match, f=TRACKING)) as response:
        fields = dict(line.split(" ", 1) for line in response.read().decode().splitlines()
                      if " " in line)
    return fields["oid"].strip().split(":", 1)[1], int(fields["size"])


def download(match):
    """The raw files of one match, downloaded if missing and the tracking file checked."""
    if match not in MATCHES:
        raise KeyError("unknown SkillCorner match {!r}".format(match))
    for name in FILES:
        path = _file(match, name)
        if not path.is_file() or path.stat().st_size == 0:
            _fetch(RAW_URL.format(m=match, f=name), path)
    path = _file(match, TRACKING)
    sha, size = _lfs_pointer(match)
    if not path.is_file() or path.stat().st_size != size:
        print("  downloading {} ({} MB)".format(path.name, size >> 20), flush=True)
        _fetch(LFS_URL.format(m=match, f=TRACKING), path)
    if _sha256(path) != sha:
        path.unlink()
        raise OSError("{} does not match its published SHA256; deleted it, run again to "
                      "re-download".format(path.name))


def _resample(values, n_out):
    """(n_src, ...) at 10 fps -> (n_out, ...) at 25 fps, linear; NaN unless both neighbours exist."""
    t = np.arange(n_out) * (SOURCE_FPS / FPS)
    j = np.minimum(np.floor(t).astype(np.int64), len(values) - 2)
    w = (t - j).reshape((-1,) + (1,) * (values.ndim - 1))
    return ((1 - w) * values[j] + w * values[j + 1]).astype(np.float32)


def _parse(match):
    with open(_file(match, "match.json")) as handle:
        meta = json.load(handle)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home = meta["home_team"]["id"]
    people = sorted(p["id"] for p in meta["players"])
    slot = {pid: k for k, pid in enumerate(people)}
    team = np.array([0 if next(p for p in meta["players"] if p["id"] == pid)["team_id"] == home
                     else 1 for pid in people], dtype=np.int8)

    with open(_file(match, TRACKING)) as handle:
        rows = [json.loads(line) for line in handle]
    n10 = rows[-1]["frame"] + 1
    ball = np.full((n10, 3), np.nan, dtype=np.float64)
    players = np.full((n10, len(people), 2), np.nan, dtype=np.float64)
    period = np.zeros(n10, dtype=np.int8)
    tracked = np.zeros(n10, dtype=np.int16)
    for r in rows:
        f = r["frame"]
        period[f] = r["period"] or 0
        b = r.get("ball_data") or {}
        if b.get("is_detected") and b.get("x") is not None:
            ball[f] = (b["x"], b["y"], b.get("z") or 0.0)
        data = r.get("player_data") or []
        tracked[f] = len(data)
        for p in data:
            k = slot.get(p["player_id"])
            if k is not None:
                players[f, k] = (p["x"], p["y"])

    alive10 = np.zeros(n10, dtype=np.int8)
    with open(_file(match, "phases_of_play.csv")) as handle:
        for p in csv.DictReader(handle):
            alive10[int(p["frame_start"]):int(p["frame_end"]) + 1] = 1

    n = int(np.floor((n10 - 1) * FPS / SOURCE_FPS)) + 1
    nearest = np.minimum(np.round(np.arange(n) * SOURCE_FPS / FPS).astype(np.int64), n10 - 1)
    half = np.array([length / 2.0, width / 2.0])
    ball25 = _resample(ball, n)
    ball25[:, :2] = (ball25[:, :2] + half) / (2 * half)
    pos = _resample(players, n)
    players25 = np.empty(pos.shape[:2] + (3,), dtype=np.float32)
    players25[..., :2] = (pos + half) / (2 * half)
    speed = np.full(pos.shape[:2], np.nan, dtype=np.float32)
    d = pos[4:] - pos[:-4]
    speed[2:-2] = np.hypot(d[..., 0], d[..., 1]) * (FPS / 4.0)
    players25[..., 2] = speed
    period25 = period[nearest]
    valid = (tracked[nearest] >= MIN_PLAYERS) & (period25 > 0)
    runs = []
    for number in (1, 2):
        idx = np.flatnonzero(period25 == number)
        runs.append([int(idx[0]), int(idx[-1]) + 1])
    return {"ball": ball25, "players": players25, "team": team, "alive": alive10[nearest],
            "valid": valid, "runs": np.array(runs), "pitch": np.array([length, width], np.float32),
            "version": np.array(CONVERTED_VERSION)}


def load(match):
    """One match at 25 fps, downloaded and converted on first use."""
    path = converted_path(match)
    if path.is_file():
        with np.load(path) as cached:
            if int(cached["version"]) == CONVERTED_VERSION:
                return {key: cached[key] for key in cached.files}
    download(match)
    print("  converting SkillCorner match {}".format(match), flush=True)
    out = _parse(match)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out
