"""Phase 0 checks that need the real data. Run them on your own machine, never in a cloud session.

    python -m saha.tools.data_checks              # all checks, then docs/EDA.md
    python -m saha.tools.data_checks --no-eda

  1. the data bundle is complete (sizes as delivered)
  2. the official starter checks: ATD alignment, Smart Tagging time base, homography rows
  3. starter/baseline.py rebuilt into a temporary folder == evaluation/results/, byte for byte
  4. every channel and parameter declared in the EPTS metadata, flagging ball state or ownership
  5. the shared frame table, with its consistency checks
  6. the speed unit, from displacements (DECISIONS D012)
  7. our own data layer reproduces the official baseline frame for frame
  8. the pipeline end to end, scored on the public window
  9. docs/EDA.md (aggregate statistics and figures)

Everything printed is aggregate (counts, percentages, hashes, names of channels); the report is
also written to cache/phase0_checks.json. Read it before sharing it with anyone or anything, and
share no more than the summary (CLAUDE.md).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from saha.common import epts, frame_table, paths
from saha.common.constants import N_FRAMES, PITCH_LENGTH_M, PITCH_WIDTH_M
from saha.nearest.build import nearest_team
from saha.tools.setup_dev import EXPECTED_BYTES

STARTER = paths.REPO_ROOT / "starter"
if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))
VARIANTS = ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125")
RESULTS = paths.EVALUATION_DIR / "results"
SPAN = 5            # frames between the two positions a displacement speed is measured over
MAX_STEP_M = 1.0    # a larger single-frame step is an identity switch, not running
MIN_SPEED = 2.0     # m/s; slower movement is dominated by tracking jitter
M_PER_S_RANGE = (0.6, 1.6)   # median ratio for "the channel is m/s"; km/h would read ~3.6


class Report:
    def __init__(self):
        self.rows, self.details = [], {}

    def add(self, name, ok, detail):
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        print("{:<6} {:<34} {}".format("PASS" if ok else "FAIL", name, detail), flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_files(report):
    missing = []
    for asset, expected in EXPECTED_BYTES.items():
        path = paths.asset_path(asset)
        if not path.is_file() or path.stat().st_size != expected:
            missing.append(path.name)
    report.add("data bundle complete", not missing, ", ".join(missing) or "4 tactical files")
    return not missing


def check_starter(report):
    import load_atd
    import load_homography
    import load_smart_tagging

    data = str(paths.data_dir())
    atd = load_atd.check_alignment_raw(data_dir=data)
    report.add("starter: ATD alignment", atd["agrees"],
               "{} records, frame_id {}..{}, contiguous {}".format(
                   atd["n_records"], atd["frame_id_range"][0], atd["frame_id_range"][1],
                   atd["contiguous"]))
    tags = load_smart_tagging.check_time_base(load_smart_tagging.load(data_dir=data))
    report.add("starter: Smart Tagging time base", tags["agrees"],
               "{} kick-off tags; both real kick-offs inside a tag: {}".format(
                   tags["n_kickoff_tags"], tags["agrees"]))
    homography = load_homography.check_alignment(data_dir=data)
    report.add("starter: homography rows", homography["agrees"],
               "{} rows".format(homography["n_rows"]))


def check_baseline_files(report, results=RESULTS):
    with tempfile.TemporaryDirectory() as folder:
        done = subprocess.run([sys.executable, str(STARTER / "baseline.py"), "--out", folder],
                              cwd=paths.REPO_ROOT, capture_output=True, text=True)
        if done.returncode:
            report.add("starter/baseline.py rebuild", False,
                       "exit code {}".format(done.returncode))
            return
        differing = [name for name in VARIANTS
                     if sha256(Path(folder) / (name + ".csv")) !=
                     sha256(Path(results) / (name + ".csv"))]
    report.add("starter/baseline.py == results/", not differing,
               "differ: {}".format(differing) if differing else "4/4 files identical")


def check_metadata(report):
    meta = epts.read_metadata(paths.asset_path("atd_meta"))
    text = epts.channels_report(meta)
    print("\n" + text + "\n")
    report.details["channels"] = text
    flagged = text.splitlines()[-1]
    report.add("EPTS: ball state / ownership", True, flagged.split(": ", 1)[1])
    home = meta.team_name(meta.home_team_id)
    report.add("EPTS: home team is Barcelona", "barcelona" in home.lower(), home)
    return meta


def check_frame_table(report):
    started = time.time()
    table = frame_table.load()
    info = table.meta
    print("\n" + frame_table.summary(table) + "\n")
    report.add("frame table built", True, "{:.0f} s, {} path".format(
        time.time() - started, info["parse_path"]))
    report.add("frame table: one record per frame", info["n_records"] == N_FRAMES
               and info["contiguous"] and info["frame_id_range"] == [1, N_FRAMES],
               "{} records, {}..{}".format(info["n_records"], *info["frame_id_range"]))
    report.add("frame table: line order == <Player>", info["spec_order_matches_metadata"],
               "the official baseline assumes it")
    seen = table.ball_visible
    report.add("frame table: ball visible 56.9 / 59.3%",
               abs(100 * seen.mean() - 56.9) < 0.1 and abs(100 * seen[table.graded].mean() -
                                                            59.3) < 0.1,
               "{:.1f}% all, {:.1f}% graded (DATA.md)".format(
                   100 * seen.mean(), 100 * seen[table.graded].mean()))
    tracked = table.players_present[table.graded].sum(axis=1).mean()
    report.add("frame table: ~20.5 tracked per frame", abs(tracked - 20.5) < 0.1,
               "{:.2f} (DATA.md says 20.5)".format(tracked))
    return table


def speed_ratio(xy_m, speed, span=SPAN, max_step=MAX_STEP_M, min_speed=MIN_SPEED):
    """(median of provided speed / displacement speed, number of samples).

    xy_m is (N, ..., 2) metres and speed (N, ...) as provided. A sample is a pair of frames
    `span` apart with the object tracked throughout, no single step above `max_step` and a
    displacement speed of at least `min_speed`; the provided speed is read halfway between.
    About 1 means the channel is in m/s, about 3.6 means km/h.
    """
    step = np.hypot(*np.moveaxis(np.diff(xy_m, axis=0), -1, 0))         # (N-1, ...)
    step = np.where(np.isfinite(step), step, np.inf)
    worst = np.lib.stride_tricks.sliding_window_view(step, span, axis=0).max(axis=-1)
    moved = xy_m[span:] - xy_m[:-span]
    disp = np.hypot(moved[..., 0], moved[..., 1]) * 25.0 / span
    given = speed[span // 2:len(speed) - span + span // 2]
    ok = (worst <= max_step) & (disp >= min_speed) & np.isfinite(given) & (given > 0)
    if not ok.any():
        return float("nan"), 0
    return float(np.median(given[ok] / disp[ok])), int(ok.sum())


def check_speed_units(report, table):
    """D012: the metadata says km/h, DATA.md says m/s. The displacements decide."""
    ratio, n = speed_ratio(table.players_xy_m(), table.players_speed)
    report.add("speed unit: players", M_PER_S_RANGE[0] <= ratio <= M_PER_S_RANGE[1],
               "median provided/displacement {:.2f} over {} samples (1 = m/s, 3.6 = km/h)"
               .format(ratio, n))
    if table.ball_speed is not None:
        ratio, n = speed_ratio(table.ball_xy_m(), table.ball_speed)
        report.add("speed unit: ball", M_PER_S_RANGE[0] <= ratio <= M_PER_S_RANGE[1],
                   "median provided/displacement {:.2f} over {} samples".format(ratio, n))


def check_our_baseline(report, meta, table, results=RESULTS):
    """Our parser + nearest_team + the official smoothing == the published baseline files."""
    import baseline

    raw = epts.read_raw(paths.asset_path("atd_raw"), meta)
    scale = np.array([PITCH_LENGTH_M, PITCH_WIDTH_M])
    slot_team = np.array([0 if p.team_id == meta.home_team_id else 1 for p in meta.players],
                         dtype=np.int8)
    codes64 = np.full(N_FRAMES, -1, dtype=np.int8)
    codes64[raw.frame_id - 1] = nearest_team(
        np.stack([frame_table.pick_channel(raw.ball, meta, "x"),
                  frame_table.pick_channel(raw.ball, meta, "y")], axis=-1) * scale,
        np.stack([frame_table.pick_channel(raw.players, meta, "x"),
                  frame_table.pick_channel(raw.players, meta, "y")], axis=-1) * scale,
        slot_team)
    codes32 = nearest_team(table.ball_xy_m(), table.players_xy_m(), table.slot_team)

    from saha.common import submission
    mismatches = {}
    for name, window in zip(VARIANTS, (0, 25, 50, 125)):
        official = submission.read_codes(Path(results) / (name + ".csv"))
        ours = submission.metrics().label_codes(baseline.smooth(codes64, window))
        mismatches[name] = int((ours != official).sum())
    report.add("our parser reproduces the baseline", not any(mismatches.values()),
               "frames differing: {}".format(mismatches))
    report.add("float32 frame table vs float64", True,
               "nearest team differs on {} frames".format(int((codes32 != codes64).sum())))


def run_pipeline(report):
    done = subprocess.run([sys.executable, "-m", "saha.pipeline"], cwd=paths.REPO_ROOT,
                          capture_output=True, text=True)
    tail = [line for line in done.stdout.splitlines() if "public 5 min" in line
            or "match-wide" in line]
    for line in tail:
        print(line)
    report.add("pipeline end to end", done.returncode == 0,
               "submissions/SAHA.csv validated" if done.returncode == 0
               else "exit {}: {}".format(done.returncode, done.stderr.strip()[-300:]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--no-eda", action="store_true")
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)

    report = Report()
    if not check_files(report):
        print("\nThe data bundle is incomplete; nothing else can run. See TEAM.md.")
        return 1
    check_starter(report)
    check_baseline_files(report)
    meta = check_metadata(report)
    table = check_frame_table(report)
    check_speed_units(report, table)
    check_our_baseline(report, meta, table)
    del table
    run_pipeline(report)
    if not args.no_eda:
        from saha.tools import eda
        eda.main([])
        report.add("docs/EDA.md written", True, "review it before committing")

    out = paths.cache_dir() / "phase0_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": report.rows, "details": report.details}, indent=2))
    failed = [row["check"] for row in report.rows if not row["ok"]]
    print("\n{} checks, {} failed{}".format(len(report.rows), len(failed),
                                           ": " + ", ".join(failed) if failed else ""))
    print("report: {}".format(out))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
