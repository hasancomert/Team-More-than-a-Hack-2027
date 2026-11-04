"""The shared frame table: one row per video frame, built once from the ATD and cached as parquet.

    python -m saha.common.frame_table            build if missing or stale, print a summary
    python -m saha.common.frame_table --rebuild

Row i is video frame i (= submission row i = ATD frame_id i + 1). Columns:

    frame          int32    0 .. 199318
    atd_present    bool     the ATD has a record for this frame
    half           int8     1 or 2 inside a graded half, 0 elsewhere
    ball_x/_y      float32  normalised 0-1, NaN when the ball is not tracked
    ball_z         float32  only if the export declares a ball z channel (BAR-ATH has none)
    ball_s         float32  ball speed in m/s, if the export declares it (BAR-ATH does)
    ball_visible   bool     ball_x and ball_y both present
    n_home/n_away  int8     tracked slots of each team on this frame
    sNN_x/_y/_s    float32  track slot NN (01..46): position 0-1 and speed in m/s
                            (the metadata says 'kph'; the values are m/s, DECISIONS D012)

Slot NN is the NN-th <Player> of the metadata ("Track NN"): a tracklet, not a person. Its team is
in the parquet metadata (key "saha"), along with the source file hashes, so a stale cache is
detected instead of silently used. Values come from the raw text through saha.common.epts, which
follows the declared EPTS layout.

Never print rows of this table into a chat or an LLM session; summaries only (CLAUDE.md).
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from saha.common import alignment, epts, paths
from saha.common.constants import GAME, N_FRAMES, PITCH_LENGTH_M, PITCH_WIDTH_M

VERSION = 2
FILENAME = "frame_table.parquet"
META_KEY = b"saha"
HOME, AWAY = 0, 1

# Channel ids as Metrica's EPTS uses them (kloppy reads the same ids), with channel names as a
# fallback in case an export names them differently.
CHANNEL_ALIASES = {
    "x": ("x", "position_x"),
    "y": ("y", "position_y"),
    "z": ("z", "position_z"),
    "s": ("s", "speed"),
}


def frame_table_path():
    return paths.cache_dir() / FILENAME


def sha256_of(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_info(path):
    path = Path(path)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_of(path)}


def pick_channel(arrays, meta, canonical):
    """The array for a canonical channel (x, y, z, s), or None."""
    ids, names = CHANNEL_ALIASES[canonical][:1], CHANNEL_ALIASES[canonical]
    for channel_id in ids:
        if channel_id in arrays:
            return arrays[channel_id]
    for channel in meta.channels:
        if channel.name.lower() in names and channel.channel_id in arrays:
            return arrays[channel.channel_id]
    return None


def build(data=None, out=None):
    """Parse the ATD and write the frame table. Returns the path written."""
    meta_path, raw_path = paths.require_assets(["atd_meta", "atd_raw"], data)
    meta = epts.read_metadata(meta_path)
    raw = epts.read_raw(raw_path, meta)

    video = alignment.video_frame_of_atd(raw.frame_id)
    if video.min() < 0 or video.max() >= N_FRAMES:
        raise epts.EptsError("ATD frame ids map outside video frames 0..{}".format(N_FRAMES - 1))
    if len(np.unique(video)) != len(video):
        raise epts.EptsError("ATD has duplicated frame ids")

    def place(values):
        if values is None:
            return None
        placed = np.full((N_FRAMES,) + values.shape[1:], np.nan, dtype=np.float32)
        placed[video] = values
        return placed

    ball_x = place(pick_channel(raw.ball, meta, "x"))
    ball_y = place(pick_channel(raw.ball, meta, "y"))
    if ball_x is None or ball_y is None:
        raise epts.EptsError("the data spec declares no ball x/y channels")
    ball_z = place(pick_channel(raw.ball, meta, "z"))
    ball_s = place(pick_channel(raw.ball, meta, "s"))
    px = place(pick_channel(raw.players, meta, "x"))
    py = place(pick_channel(raw.players, meta, "y"))
    if px is None or py is None:
        raise epts.EptsError("the data spec declares no player x/y channels")
    speed = place(pick_channel(raw.players, meta, "s"))
    has_speed = speed is not None
    if not has_speed:
        speed = np.full(px.shape, np.nan, dtype=np.float32)

    team_by_player = {p.player_id: p.team_id for p in meta.players}
    slot_team = np.array([HOME if team_by_player[pid] == meta.home_team_id else AWAY
                          for pid in raw.slot_player_ids], dtype=np.int8)
    present = np.zeros(N_FRAMES, dtype=bool)
    present[video] = True
    tracked = np.isfinite(px) & np.isfinite(py)

    columns = {
        "frame": np.arange(N_FRAMES, dtype=np.int32),
        "atd_present": present,
        "half": alignment.half_of_frames(N_FRAMES),
        "ball_x": ball_x,
        "ball_y": ball_y,
    }
    if ball_z is not None:
        columns["ball_z"] = ball_z
    if ball_s is not None:
        columns["ball_s"] = ball_s
    columns["ball_visible"] = np.isfinite(ball_x) & np.isfinite(ball_y)
    columns["n_home"] = tracked[:, slot_team == HOME].sum(axis=1).astype(np.int8)
    columns["n_away"] = tracked[:, slot_team == AWAY].sum(axis=1).astype(np.int8)
    for slot in range(len(slot_team)):
        stem = "s{:02d}".format(slot + 1)
        columns[stem + "_x"] = px[:, slot]
        columns[stem + "_y"] = py[:, slot]
        columns[stem + "_s"] = speed[:, slot]

    players = {p.player_id: p for p in meta.players}
    info = {
        "version": VERSION,
        "game": GAME,
        "n_frames": N_FRAMES,
        "sources": {"atd_meta": _source_info(meta_path), "atd_raw": _source_info(raw_path)},
        "parse_path": raw.path,
        "lines_outside_spec": raw.outside_spec,
        "n_records": int(len(raw.frame_id)),
        "frame_id_range": [int(raw.frame_id.min()), int(raw.frame_id.max())],
        "contiguous": bool(np.array_equal(np.sort(raw.frame_id),
                                          np.arange(raw.frame_id.min(), raw.frame_id.max() + 1))),
        "teams": {"H": meta.team_name(meta.home_team_id), "A": meta.team_name(meta.away_team_id)},
        "home_is_first_team": meta.home_team_id == meta.teams[0].team_id,
        "slots": [{"slot": i + 1, "player_id": pid, "name": players[pid].name,
                   "team": "H" if slot_team[i] == HOME else "A"}
                  for i, pid in enumerate(raw.slot_player_ids)],
        "spec_order_matches_metadata": all(
            spec.player_order() == [pid for pid in meta.player_ids()
                                    if pid in set(spec.player_order())]
            for spec in meta.specs),
        "ball_channels": sorted(raw.ball),
        "player_channels": sorted(raw.players),
        "has_speed": has_speed,
    }

    table = pa.table(columns)
    table = table.replace_schema_metadata({META_KEY: json.dumps(info, sort_keys=True).encode()})
    out = Path(out) if out else frame_table_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    return out


@dataclass
class FrameTable:
    frame: np.ndarray           # (N,) int32
    atd_present: np.ndarray     # (N,) bool
    half: np.ndarray            # (N,) int8
    ball_xy: np.ndarray         # (N, 2) float32, normalised
    ball_z: object              # (N,) float32 or None
    ball_speed: object          # (N,) float32 m/s, or None
    ball_visible: np.ndarray    # (N,) bool
    players_xy: np.ndarray      # (N, S, 2) float32, normalised
    players_speed: np.ndarray   # (N, S) float32, m/s
    slot_team: np.ndarray       # (S,) int8, 0 = home (H), 1 = away (A)
    meta: dict

    @property
    def n_frames(self):
        return len(self.frame)

    @property
    def n_slots(self):
        return len(self.slot_team)

    @property
    def graded(self):
        return self.half > 0

    @property
    def players_present(self):
        return np.isfinite(self.players_xy).all(axis=-1)

    def ball_xy_m(self):
        """Ball position in metres, float64 (pitch size from games.py)."""
        return self.ball_xy.astype(np.float64) * np.array([PITCH_LENGTH_M, PITCH_WIDTH_M])

    def players_xy_m(self):
        return self.players_xy.astype(np.float64) * np.array([PITCH_LENGTH_M, PITCH_WIDTH_M])


def read_info(path=None):
    """The 'saha' metadata of a frame table file, without loading the arrays."""
    schema = pq.read_schema(Path(path) if path else frame_table_path())
    return json.loads((schema.metadata or {}).get(META_KEY, b"{}"))


def is_current(data=None, path=None):
    """(ok, reason): the cached table exists, has this VERSION and matches the data files."""
    path = Path(path) if path else frame_table_path()
    if not path.is_file():
        return False, "no frame table at {}".format(path)
    info = read_info(path)
    if info.get("version") != VERSION:
        return False, "frame table version {} != {}".format(info.get("version"), VERSION)
    meta_path, raw_path = paths.require_assets(["atd_meta", "atd_raw"], data)
    for key, source in (("atd_meta", meta_path), ("atd_raw", raw_path)):
        recorded = info.get("sources", {}).get(key, {})
        if recorded.get("bytes") != source.stat().st_size or \
                recorded.get("sha256") != sha256_of(source):
            return False, "{} changed since the frame table was built".format(source.name)
    return True, "up to date"


def load(data=None, build_if_needed=True, path=None):
    """The frame table, rebuilt first if it is missing or stale."""
    ok, reason = is_current(data, path)
    if not ok:
        if not build_if_needed:
            raise FileNotFoundError(reason + " -- run: python -m saha.common.frame_table")
        build(data, path)
    table = pq.read_table(Path(path) if path else frame_table_path())
    info = json.loads(table.schema.metadata[META_KEY])
    n_slots = len(info["slots"])

    def col(name):
        return table.column(name).to_numpy()

    players_xy = np.stack(
        [np.stack([col("s{:02d}_x".format(s + 1)), col("s{:02d}_y".format(s + 1))], axis=-1)
         for s in range(n_slots)], axis=1)
    speed = np.stack([col("s{:02d}_s".format(s + 1)) for s in range(n_slots)], axis=1)
    return FrameTable(
        frame=col("frame"),
        atd_present=col("atd_present"),
        half=col("half"),
        ball_xy=np.stack([col("ball_x"), col("ball_y")], axis=-1),
        ball_z=col("ball_z") if "ball_z" in table.column_names else None,
        ball_speed=col("ball_s") if "ball_s" in table.column_names else None,
        ball_visible=col("ball_visible"),
        players_xy=players_xy,
        players_speed=speed,
        slot_team=np.array([HOME if s["team"] == "H" else AWAY for s in info["slots"]],
                           dtype=np.int8),
        meta=info,
    )


def summary(table):
    """Aggregate description, safe to share: counts and percentages only."""
    graded = table.graded
    tracked = table.players_present
    info = table.meta
    lines = [
        "frame table v{}  {} frames  parsed via the {} path".format(
            info["version"], table.n_frames, info["parse_path"]),
        "ATD records {}  frame_id {}..{}  contiguous {}  outside the declared layout {}".format(
            info["n_records"], *info["frame_id_range"], info["contiguous"],
            info.get("lines_outside_spec", 0)),
        "teams  H = {}  A = {}  (home is the first <Team>: {})".format(
            info["teams"]["H"], info["teams"]["A"], info["home_is_first_team"]),
        "slots  {} ({} home, {} away)  data-spec order == <Player> order: {}".format(
            table.n_slots, int((table.slot_team == HOME).sum()),
            int((table.slot_team == AWAY).sum()), info["spec_order_matches_metadata"]),
        "channels  ball {}  players {}  speed {}".format(
            info["ball_channels"], info["player_channels"],
            "found" if info["has_speed"] else "MISSING (speeds are NaN)"),
        "ball visible  {:.1f}% of all frames, {:.1f}% of graded frames".format(
            100.0 * table.ball_visible.mean(), 100.0 * table.ball_visible[graded].mean()),
        "tracked slots per graded frame  mean {:.1f}  (home {:.1f}, away {:.1f})".format(
            tracked[graded].sum(axis=1).mean(),
            tracked[graded][:, table.slot_team == HOME].sum(axis=1).mean(),
            tracked[graded][:, table.slot_team == AWAY].sum(axis=1).mean()),
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if up to date")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)
    if args.rebuild:
        build()
    table = load()
    print(summary(table))
    print("cached at {}".format(frame_table_path()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
