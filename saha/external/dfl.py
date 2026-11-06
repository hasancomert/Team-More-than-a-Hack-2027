"""DFL/Sportec open tracking data (the IDSSE dataset), labelled H/A/D on every frame.

Source: Bassek, Weber, Rein and Memmert, "An integrated dataset of spatiotemporal and event data
in elite soccer", Scientific Data (2025). Seven Bundesliga / 2. Bundesliga matches of 2022/23,
published on figshare under CC BY 4.0. Licence and attribution: docs/tech/ws5_external.md.

Why this dataset first: it is 25 fps like ours, and its position file carries two per-frame
fields on the ball, BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 guest). Those map
straight to our labels:

    BallStatus 0                      -> D
    BallStatus 1, BallPossession 1    -> H
    BallStatus 1, BallPossession 2    -> A
    anything else                     -> X (not scored)

Files are downloaded once to cache/external/dfl/ (gitignored) and converted to one .npz per match,
so later runs take seconds. We parse the XML ourselves (streaming, standard library) instead of
through kloppy, because kloppy builds one Python object per frame and player and we only need
arrays. Coordinates become our 0-1 convention: x along the length, y across, origin at a corner.
"""

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from saha.common import paths
from saha.common.constants import A, D, H, X

FPS = 25
CONVERTED_VERSION = 1
# Frames of X (no data, not scored) put between the halves, so the scorer never joins them.
HALF_GAP = 50
SECTIONS = ("firstHalf", "secondHalf")  # extra time, if any, is ignored
BALL_TEAM = "BALL"

DOWNLOAD_URL = "https://api.figshare.com/v2/file/download/{}"
USER_AGENT = "saha-ws5-external/1"
LICENCE = "CC BY 4.0"
CITATION = ("Bassek, M., Weber, H., Rein, R., & Memmert, D. (2025). An integrated dataset of "
            "spatiotemporal and event data in elite soccer. Scientific Data.")

# match id -> (metadata file id, position file id) on figshare. The same ids kloppy 3.18 uses in
# sportec.load_open_tracking_data.
MATCHES = {
    "J03WMX": (51643475, 51643514),   # 1. FC Koeln - FC Bayern Muenchen
    "J03WN1": (51643472, 51643517),   # VfL Bochum 1848 - Bayer 04 Leverkusen
    "J03WOH": (51643478, 51643520),   # Fortuna Duesseldorf - SSV Jahn Regensburg
    "J03WOY": (51643481, 51643523),   # Fortuna Duesseldorf - F.C. Hansa Rostock
    "J03WPY": (51643487, 51643526),   # Fortuna Duesseldorf - 1. FC Nuernberg
    "J03WQQ": (51643484, 51643529),   # Fortuna Duesseldorf - FC St. Pauli
    "J03WR9": (51643490, 51643532),   # Fortuna Duesseldorf - 1. FC Kaiserslautern
}
DEFAULT_MATCHES = ("J03WPY",)


def dfl_dir():
    return paths.cache_dir() / "external" / "dfl"


def raw_paths(match_id):
    folder = dfl_dir()
    return folder / "{}_meta.xml".format(match_id), folder / "{}_tracking.xml".format(match_id)


def converted_path(match_id):
    return dfl_dir() / "{}.npz".format(match_id)


def _sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_once(file_id, partial):
    request = urllib.request.Request(DOWNLOAD_URL.format(file_id),
                                     headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        expected = int(response.headers.get("Content-Length") or 0)
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    size = partial.stat().st_size
    # A dropped connection can end the stream early without an error: check the length.
    if size == 0 or (expected and size != expected):
        raise OSError("figshare file {}: got {} of {} bytes".format(file_id, size, expected))


def _fetch(file_id, target, attempts=3):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            _fetch_once(file_id, partial)
            partial.replace(target)
            return
        except OSError as error:
            print("  attempt {} failed: {}".format(attempt, error), flush=True)
            if partial.exists():
                partial.unlink()
    raise OSError("could not download figshare file {}".format(file_id))


def download(match_id):
    """Both raw files of one match, downloaded if missing. Returns (meta, tracking) paths."""
    if match_id not in MATCHES:
        raise KeyError("unknown DFL match {!r}; known: {}".format(match_id, sorted(MATCHES)))
    meta_path, tracking_path = raw_paths(match_id)
    for file_id, path in zip(MATCHES[match_id], (meta_path, tracking_path)):
        if not path.is_file() or path.stat().st_size == 0:
            print("  downloading {} (figshare file {})".format(path.name, file_id), flush=True)
            _fetch(file_id, path)
    return meta_path, tracking_path


# -- parsing ---------------------------------------------------------------------------------

def _xml_root(path):
    data = Path(path).read_bytes()
    try:
        data.decode("utf-8")
        return ET.fromstring(data)
    except UnicodeDecodeError:
        # Some DFL files declare UTF-8 but hold Latin-1 names; only ids are read from them.
        text = data.decode("latin-1")
        return ET.fromstring(text[text.index("?>") + 2:] if text.startswith("<?xml") else text)


def read_meta(path):
    """{'home', 'away', 'pitch', 'team_of'} from a DFL match-information file."""
    root = _xml_root(path)
    general = root.find(".//General")
    environment = root.find(".//Environment")
    team_of = {}
    for team in root.iter("Team"):
        for player in team.iter("Player"):
            team_of[player.get("PersonId")] = team.get("TeamId")
    return {
        "match_id": general.get("MatchId"),
        "home": general.get("HomeTeamId"),
        "away": general.get("GuestTeamId"),
        "pitch": (float(environment.get("PitchX")), float(environment.get("PitchY"))),
        "team_of": team_of,
    }


def read_positions(path):
    """{(section, team_id, person_id): dict of arrays} from a DFL position file, streamed."""
    sets = {}
    key, rows = None, None
    for event, element in ET.iterparse(str(path), events=("start", "end")):
        tag = element.tag
        if event == "start" and tag == "FrameSet":
            key = (element.get("GameSection"), element.get("TeamId"), element.get("PersonId"))
            rows = []
        elif event == "end" and tag == "Frame":
            get = element.get
            rows.append((int(get("N")), float(get("X")), float(get("Y")),
                         float(get("Z") or "nan"), int(get("BallStatus") or -1),
                         int(get("BallPossession") or -1)))
            element.clear()
        elif event == "end" and tag == "FrameSet":
            array = np.array(rows, dtype=np.float64).reshape(-1, 6)
            sets[key] = {"n": array[:, 0].astype(np.int64), "x": array[:, 1], "y": array[:, 2],
                         "z": array[:, 3], "status": array[:, 4].astype(np.int8),
                         "possession": array[:, 5].astype(np.int8)}
            key, rows = None, None
            element.clear()
    return sets


def labels_from_ball(status, possession):
    """Label codes from the ball's BallStatus and BallPossession (module docstring)."""
    labels = np.full(len(status), X, dtype=np.int8)
    labels[status == 0] = D
    alive = status == 1
    labels[alive & (possession == 1)] = H
    labels[alive & (possession == 2)] = A
    return labels


def convert(meta_path, tracking_path):
    """Arrays for one match, both halves on one time line with HALF_GAP frames of X between."""
    meta = read_meta(meta_path)
    sets = read_positions(tracking_path)
    length, width = meta["pitch"]
    team_of = {person: team for (section, team, person) in sets
               if team in (meta["home"], meta["away"]) and section in SECTIONS}
    persons = sorted(team_of)
    slot_team = np.array([0 if team_of[p] == meta["home"] else 1 for p in persons],
                         dtype=np.int8)
    slot_index = {p: i for i, p in enumerate(persons)}

    blocks, offset = [], 0
    for number, section in enumerate(SECTIONS, start=1):
        balls = [v for (s, team, _), v in sets.items() if s == section and team == BALL_TEAM]
        if not balls:
            continue
        ball = balls[0]
        first, last = int(ball["n"].min()), int(ball["n"].max())
        n = last - first + 1
        ball_xy = np.full((n, 2), np.nan)
        ball_z = np.full(n, np.nan)
        status = np.full(n, -1, dtype=np.int8)
        possession = np.full(n, -1, dtype=np.int8)
        at = ball["n"] - first
        ball_xy[at, 0], ball_xy[at, 1] = ball["x"], ball["y"]
        ball_z[at], status[at], possession[at] = ball["z"], ball["status"], ball["possession"]
        players = np.full((n, len(persons), 2), np.nan)
        for (s, team, person), v in sets.items():
            if s != section or person not in slot_index:
                continue
            inside = (v["n"] >= first) & (v["n"] <= last)
            at = v["n"][inside] - first
            players[at, slot_index[person], 0] = v["x"][inside]
            players[at, slot_index[person], 1] = v["y"][inside]
        blocks.append((number, offset, ball_xy, ball_z, players,
                       labels_from_ball(status, possession)))
        offset += n + HALF_GAP

    total = offset - HALF_GAP
    out = {
        "ball_xy": np.full((total, 2), np.nan, dtype=np.float32),
        "ball_z": np.full(total, np.nan, dtype=np.float32),
        "players_xy": np.full((total, len(persons), 2), np.nan, dtype=np.float32),
        "labels": np.full(total, X, dtype=np.int8),
        "half": np.zeros(total, dtype=np.int8),
    }
    for number, start, ball_xy, ball_z, players, labels in blocks:
        stop = start + len(labels)
        out["ball_xy"][start:stop] = _normalise(ball_xy, length, width)
        out["ball_z"][start:stop] = ball_z
        out["players_xy"][start:stop] = _normalise(players, length, width)
        out["labels"][start:stop] = labels
        out["half"][start:stop] = number
    info = {"version": CONVERTED_VERSION, "source": "DFL/IDSSE", "match_id": meta["match_id"],
            "licence": LICENCE, "pitch_m": [length, width], "fps": FPS, "half_gap": HALF_GAP,
            "n_slots": len(persons),
            "raw_sha256": {"meta": _sha256(meta_path), "tracking": _sha256(tracking_path)}}
    out["slot_team"] = slot_team
    out["slot_person"] = np.array(persons)
    out["info"] = np.array(json.dumps(info, sort_keys=True))
    return out


def _normalise(xy_m, length, width):
    """DFL metres (centre origin, y up) -> 0-1 (corner origin, y down), like Metrica's EPTS."""
    out = np.empty_like(xy_m, dtype=np.float32)
    out[..., 0] = (xy_m[..., 0] + length / 2.0) / length
    out[..., 1] = (width / 2.0 - xy_m[..., 1]) / width
    return out


# -- the converted match ---------------------------------------------------------------------

@dataclass
class ExternalMatch:
    name: str
    ball_xy: np.ndarray       # (N, 2) float32, 0-1, the ball is always tracked here
    ball_z: np.ndarray        # (N,) float32 metres (our match has no height channel)
    players_xy: np.ndarray    # (N, S, 2) float32, 0-1, one slot per real person
    slot_team: np.ndarray     # (S,) int8, 0 home, 1 away
    slot_person: np.ndarray   # (S,) str
    labels: np.ndarray        # (N,) int8 H/A/D, X where not scored
    half: np.ndarray          # (N,) int8 1/2, 0 in the gap between halves
    info: dict

    @property
    def n_frames(self):
        return len(self.labels)


def prepare(match_id, rebuild=False):
    """Download and convert one match if needed; return it as an ExternalMatch."""
    path = converted_path(match_id)
    if rebuild or not path.is_file() or _version_of(path) != CONVERTED_VERSION:
        meta_path, tracking_path = download(match_id)
        print("  converting {} (a few minutes, once)".format(match_id), flush=True)
        try:
            arrays = convert(meta_path, tracking_path)
        except ET.ParseError as error:
            # Almost always a truncated download: delete it so the next run fetches it again.
            tracking_path.unlink()
            raise OSError("{} is not valid XML ({}); deleted it, run again to re-download"
                          .format(tracking_path.name, error)) from error
        with open(path, "wb") as handle:
            np.savez_compressed(handle, **arrays)
    return load(match_id)


def _version_of(path):
    try:
        with np.load(path) as data:
            return json.loads(str(data["info"])).get("version")
    except (OSError, KeyError, ValueError):
        return None


def load(match_id):
    with np.load(converted_path(match_id)) as data:
        return ExternalMatch(
            name="dfl-" + match_id, ball_xy=data["ball_xy"], ball_z=data["ball_z"],
            players_xy=data["players_xy"], slot_team=data["slot_team"],
            slot_person=data["slot_person"], labels=data["labels"], half=data["half"],
            info=json.loads(str(data["info"])))


def converted_matches():
    """Match ids already converted in the cache, sorted."""
    return sorted(m for m in MATCHES if converted_path(m).is_file())
