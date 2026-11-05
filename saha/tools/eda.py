"""Exploratory statistics of the match -> docs/EDA.md and docs/figures/eda_*.png.

    python -m saha.tools.eda

Needs the real data, so run it on your own machine. The output is aggregate only: distributions,
counts, percentages, never a frame-level table. That is what makes it committable. Read the file
before committing it anyway.

The public ground truth (public, MIT) is used to split statistics by dead/alive in the first five
minutes. That is description, not tuning: nothing here feeds a parameter (CLAUDE.md).
"""

import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np

from saha.common import alignment, frame_table, paths, smart_tagging, submission
from saha.common.constants import A, D, FPS, H, PITCH_LENGTH_M, PITCH_WIDTH_M, PUBLIC_WINDOW

GAP_BUCKETS = ((0, 0.2), (0.2, 1), (1, 3), (3, 10), (10, np.inf))   # seconds
JUMP_M = 1.0   # metres between consecutive frames (25 m/s): no player runs that fast


def runs(mask):
    """(starts, lengths) of the True runs of a boolean array."""
    padded = np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0]))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def graded_runs(mask):
    """Run lengths of mask inside each graded half (runs never cross halftime)."""
    lengths = []
    for _, start, stop in alignment.half_slices():
        lengths.append(runs(mask[start:stop])[1])
    return np.concatenate(lengths)


def quantiles(values, qs=(50, 75, 90, 99)):
    if not len(values):
        return "n/a"
    return ", ".join("p{} {:.2f}".format(q, np.percentile(values, q)) for q in qs) + \
        ", max {:.2f}".format(np.max(values))


class Figures:
    """Writes figures next to the report (<out dir>/figures) and remembers their names."""

    def __init__(self, folder):
        self.folder, self.names = Path(folder), []

    def add(self, name, draw):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2), dpi=100)
        draw(ax)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self.folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.folder / name)
        plt.close(fig)
        self.names.append(name)
        return name


def section_coverage(t):
    graded = t.graded
    tracked = t.players_present.sum(axis=1)
    home = t.players_present[:, t.slot_team == 0].sum(axis=1)
    away = t.players_present[:, t.slot_team == 1].sum(axis=1)
    lines = ["## Coverage", "",
             "| | all frames | graded | 1st half | 2nd half |", "|---|---|---|---|---|"]
    halves = [t.half == 1, t.half == 2]
    lines.append("| ball visible | {:.1%} | {:.1%} | {:.1%} | {:.1%} |".format(
        t.ball_visible.mean(), t.ball_visible[graded].mean(),
        *[t.ball_visible[h].mean() for h in halves]))
    lines.append("| tracked slots per frame (mean) | {:.1f} | {:.1f} | {:.1f} | {:.1f} |".format(
        tracked.mean(), tracked[graded].mean(), *[tracked[h].mean() for h in halves]))
    lines += ["",
              "On graded frames: home {:.1f} and away {:.1f} tracked slots on average. "
              "{:.1%} of frames have more than 22 tracked (duplicates) and {:.1%} have fewer "
              "than 18.".format(home[graded].mean(), away[graded].mean(),
                                (tracked[graded] > 22).mean(), (tracked[graded] < 18).mean()),
              ""]
    return lines


def section_gaps(t, figures):
    gaps = graded_runs(~t.ball_visible) / FPS
    missing = gaps.sum()
    lines = ["## Ball-missing stretches (graded frames)", "",
             "{} stretches. Durations in seconds: {}.".format(len(gaps), quantiles(gaps)), "",
             "| stretch length | stretches | share of missing frames |", "|---|---|---|"]
    for low, high in GAP_BUCKETS:
        inside = (gaps >= low) & (gaps < high)
        label = "≥ {:g} s".format(low) if np.isinf(high) else "{:g}–{:g} s".format(low, high)
        lines.append("| {} | {} | {:.1%} |".format(label, int(inside.sum()),
                                                  gaps[inside].sum() / missing if missing else 0))
    name = figures.add("eda_ball_gaps.png", lambda ax: (
        ax.hist(gaps, bins=np.logspace(np.log10(1 / FPS), np.log10(max(gaps.max(), 1)), 40)
                if len(gaps) else [], weights=gaps if len(gaps) else None, color="#1f77b4"),
        ax.set_xscale("log"), ax.set_xlabel("ball-missing stretch (s, log scale)"),
        ax.set_ylabel("missing seconds"),
        ax.set_title("Where the missing ball time is: short blips vs long stretches")))
    lines += ["", "![ball gaps](figures/{})".format(name), ""]
    return lines


def section_out_of_pitch(t):
    graded = t.graded & t.ball_visible
    x, y = t.ball_xy[:, 0], t.ball_xy[:, 1]
    outside = graded & ((x < 0) | (x > 1) | (y < 0) | (y > 1))
    margin_x, margin_y = 1.0 / PITCH_LENGTH_M, 1.0 / PITCH_WIDTH_M
    near = graded & ~outside & ((x < margin_x) | (x > 1 - margin_x) | (y < margin_y) |
                                (y > 1 - margin_y))
    lengths = graded_runs(outside) / FPS
    lines = ["## Ball outside the pitch lines", "",
             "Of graded frames with a visible ball, {:.2%} place it outside the 0–1 pitch and "
             "{:.2%} within 1 m inside a line. Ranges seen: x {:.3f}..{:.3f}, y {:.3f}..{:.3f}."
             .format(outside.sum() / max(graded.sum(), 1), near.sum() / max(graded.sum(), 1),
                     np.nanmin(x[graded]) if graded.any() else np.nan,
                     np.nanmax(x[graded]) if graded.any() else np.nan,
                     np.nanmin(y[graded]) if graded.any() else np.nan,
                     np.nanmax(y[graded]) if graded.any() else np.nan),
             "", "{} excursions outside; durations (s): {}.".format(len(lengths),
                                                                   quantiles(lengths)), ""]
    if t.ball_z is not None and np.isfinite(t.ball_z[graded]).any():
        z = t.ball_z[graded]
        lines += ["Ball z is present on {:.1%} of those frames; {}.".format(
            np.isfinite(z).mean(), quantiles(z[np.isfinite(z)])), ""]
    return lines


def section_speeds(t, gt, figures):
    speed = np.where(t.players_present, t.players_speed, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames with nobody tracked
        mean_speed = np.nanmean(speed, axis=1)
    start, stop = PUBLIC_WINDOW[0], PUBLIC_WINDOW[1] + 1
    window = np.zeros(len(gt), dtype=bool)
    window[start:stop] = True
    dead = window & (gt == D) & np.isfinite(mean_speed)
    alive = window & ((gt == H) | (gt == A)) & np.isfinite(mean_speed)
    lines = ["## Player speed when the ball is dead", "",
             "Per-frame mean speed of tracked players (m/s), public window split by the public "
             "ground truth. It helps to understand the data, but it is never a threshold or a "
             "parameter (DECISIONS D016):", "",
             "| | frames | median | p10 | p90 | ball visible |", "|---|---|---|---|---|---|"]
    for label, mask in (("dead (D)", dead), ("alive (H/A)", alive)):
        values = mean_speed[mask]
        lines.append("| {} | {} | {} | {} | {} | {:.1%} |".format(
            label, int(mask.sum()), *["{:.2f}".format(np.percentile(values, q))
                                      if len(values) else "n/a" for q in (50, 10, 90)],
            t.ball_visible[mask].mean() if mask.any() else 0))
    graded = t.graded & np.isfinite(mean_speed)
    lines += ["", "Whole match, graded frames: {}.".format(quantiles(mean_speed[graded],
                                                                       (10, 50, 90))), ""]
    name = figures.add("eda_speed_dead_alive.png", lambda ax: (
        ax.hist([mean_speed[alive], mean_speed[dead]], bins=40, density=True,
                label=["alive", "dead"], color=["#1f77b4", "#d62728"], alpha=0.7),
        ax.set_xlabel("mean speed of tracked players (m/s)"), ax.set_ylabel("density"),
        ax.legend(frameon=False), ax.set_title("Public window: players slow down when dead")))
    lines += ["![speeds](figures/{})".format(name), ""]
    return lines


def section_restarts(gt):
    pm = submission.metrics()
    tags = smart_tagging.load()
    restarts = smart_tagging.restarts(tags)
    lines = ["## Smart Tagging restarts against the public ground truth", "",
             "Restart tags in the whole match: {}.".format(
                 ", ".join("{} {}".format(n, kind) for kind, n in
                           restarts["type"].value_counts().items()) or "none"), ""]
    runs_public = pm.valid_runs(gt)
    starts = [(f, to) for f, frm, to in pm.transitions(gt, runs_public) if frm == D and to != D]
    start, stop = PUBLIC_WINDOW[0], PUBLIC_WINDOW[1] + 1
    inside = restarts[(restarts["end_frame"] > start) & (restarts["start_frame"] < stop)]
    if not len(inside) or not starts:
        return lines + ["No restart tag overlaps the public window.", ""]
    offsets_start, offsets_end, agree, unknown = [], [], 0, 0
    for _, tag in inside.iterrows():
        middle = (tag["start_frame"] + tag["end_frame"]) / 2
        frame, team = min(starts, key=lambda s: abs(s[0] - middle))
        offsets_start.append(frame - tag["start_frame"])
        offsets_end.append(frame - tag["end_frame"])
        if tag["team"] is None:
            unknown += 1
        else:
            agree += int({"H": H, "A": A}[tag["team"]] == team)
    lines += [
        "{} restart tags overlap the public window, which has {} D→H/A restarts. For each tag, "
        "the nearest restart in the ground truth lies (frames, + = after):".format(
            len(inside), len(starts)), "",
        "| | median | min | max |", "|---|---|---|---|",
        "| after the tag's start | {:.0f} | {} | {} |".format(
            np.median(offsets_start), min(offsets_start), max(offsets_start)),
        "| after the tag's end | {:.0f} | {} | {} |".format(
            np.median(offsets_end), min(offsets_end), max(offsets_end)), "",
        "Team tag agrees with the restarting team in {} of {} tags with a team ({} are N/A)."
        .format(agree, len(inside) - unknown, unknown), ""]
    return lines


def section_tracklets(t, figures):
    xy_m = t.players_xy.astype(np.float64) * np.array([PITCH_LENGTH_M, PITCH_WIDTH_M])
    lengths, jumps, minutes = [], 0, t.graded.sum() / FPS / 60
    for slot in range(t.n_slots):
        present = t.players_present[:, slot]
        lengths.append(graded_runs(present))
        step = np.hypot(*np.diff(xy_m[:, slot], axis=0).T)
        both = present[1:] & present[:-1] & t.graded[1:] & t.graded[:-1]
        jumps += int((step[both] > JUMP_M).sum())
    lengths = np.concatenate(lengths) / FPS
    used = int((t.players_present[t.graded].any(axis=0)).sum())
    lines = ["## Track slots: fragmentation and identity switches", "",
             "{} of {} slots are used on graded frames. They form {} tracklets (unbroken "
             "presence) of duration (s): {}.".format(used, t.n_slots, len(lengths),
                                                     quantiles(lengths)), "",
             "{:.1f} tracklets start per graded minute. Jumps of more than {:g} m between "
             "consecutive frames inside a tracklet (a proxy for an identity switch): {} in "
             "total, {:.2f} per graded minute.".format(len(lengths) / minutes, JUMP_M, jumps,
                                                      jumps / minutes), ""]
    name = figures.add("eda_tracklets.png", lambda ax: (
        ax.hist(lengths, bins=np.logspace(np.log10(1 / FPS), np.log10(max(lengths.max(), 1)),
                                          40), color="#2ca02c"),
        ax.set_xscale("log"), ax.set_xlabel("tracklet duration (s, log scale)"),
        ax.set_ylabel("tracklets"), ax.set_title("How long a track slot follows someone")))
    lines += ["![tracklets](figures/{})".format(name), ""]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--out", default=str(paths.DOCS_DIR / "EDA.md"))
    args = parser.parse_args(argv)
    if args.data:
        paths.use_data_dir(args.data)

    table = frame_table.load()
    gt = submission.public_gt()
    sources = table.meta["sources"]
    figures = Figures(Path(args.out).parent / "figures")
    lines = ["# EDA: BAR-ATH tactical tracking", "",
             "Generated by `python -m saha.tools.eda` on {} from the data bundle with ATD "
             "sha256 {}… (raw) / {}… (metadata). Aggregates only: no frame-level data. "
             "Do not edit by hand; re-run the tool.".format(
                 datetime.date.today().isoformat(), sources["atd_raw"]["sha256"][:12],
                 sources["atd_meta"]["sha256"][:12]), ""]
    lines += section_coverage(table)
    lines += section_gaps(table, figures)
    lines += section_out_of_pitch(table)
    lines += section_speeds(table, gt, figures)
    lines += section_restarts(gt)
    lines += section_tracklets(table, figures)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("wrote {} and {} figures".format(args.out, len(figures.names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
