"""How spellF1 reacts to each kind of mistake, measured with the official scorer.

    python -m saha.tools.metric_behaviour          -> docs/METRIC_NOTES.md, docs/figures/

Takes the public ground truth, applies one deliberate mistake at a time (late boundaries,
jitter, false splits, missed or swapped possessions, wrong stoppages, abstentions) and scores the
damaged copy with evaluation/possession_metrics.py, unmodified.

Two references:
  public  the 5 public minutes as they are: 17 possessions, so every number is coarse.
  tiled   the public segment pattern repeated to fill both graded halves (~360 possessions):
          a scale model for "cost per mistake on a full match". It is not a claim about the real
          match -- only that costs per mistake shrink roughly as 1/possessions.

This is a study of the metric, not of a model: nothing here is fitted, and nothing here may be
used to choose between two versions of our pipeline (CLAUDE.md).
"""

import argparse
from pathlib import Path

import numpy as np

from saha.common import alignment, paths, submission
from saha.common.constants import A, D, FPS, H, N_FRAMES, X

SEEDS_PUBLIC = 50
SEEDS_TILED = 20


# -- references --------------------------------------------------------------------------------

def tiled_truth(public):
    """The public window's label pattern repeated over both graded halves, X elsewhere."""
    first, last = 11185, 18685
    pattern = public[first:last]
    truth = np.full(N_FRAMES, X, dtype=np.int8)
    for _, start, stop in alignment.half_slices():
        reps = -(-(stop - start) // len(pattern))
        truth[start:stop] = np.tile(pattern, reps)[:stop - start]
    return truth


def runs_of(truth):
    return submission.metrics().valid_runs(truth)


def segments(codes, runs):
    return submission.metrics().segments(codes, runs)


def possession_indices(segs):
    return [i for i, (_, _, k) in enumerate(segs) if k in (H, A)]


# -- perturbations: (truth, runs, rng, param) -> predicted codes --------------------------------

def late(truth, runs, rng, k):
    """Every label k frames late (negative = early)."""
    pred = truth.copy()
    for start, stop in runs:
        window = truth[start:stop]
        if k >= 0:
            pred[start:stop] = np.concatenate((np.full(k, window[0]), window[:len(window) - k]))
        else:
            pred[start:stop] = np.concatenate((window[-k:], np.full(-k, window[-1])))
    return pred


def _move_boundaries(truth, runs, delta):
    """Rebuild with each boundary moved by delta(from, to, rng-free index); keeps order."""
    pred = truth.copy()
    for start, stop in runs:
        segs = segments(truth, [(start, stop)])
        bounds = [s for s, _, _ in segs[1:]]
        moved, previous = [], start
        for index, b in enumerate(bounds):
            new = b + delta(segs[index][2], segs[index + 1][2], index)
            limit = bounds[index + 1] - 1 if index + 1 < len(bounds) else stop - 1
            new = min(max(new, previous + 1), limit)
            moved.append(new)
            previous = new
        edges = [start] + moved + [stop]
        for (a, b), (_, _, code) in zip(zip(edges[:-1], edges[1:]), segs):
            pred[a:b] = code
    return pred


def jitter(truth, runs, rng, k):
    draws = rng.integers(-k, k + 1, size=len(segments(truth, runs)) + 1)
    return _move_boundaries(truth, runs, lambda f, t, i: int(draws[i]))


def handover_late(truth, runs, rng, k):
    return _move_boundaries(truth, runs, lambda f, t, i: k if D not in (f, t) else 0)


def stoppage_late(truth, runs, rng, k):
    """Both ends of every dead-ball stretch k frames late (ball out noticed late, restart too)."""
    return _move_boundaries(truth, runs, lambda f, t, i: k if D in (f, t) else 0)


def _pick(rng, candidates, n):
    if len(candidates) <= n:
        return list(candidates)
    return sorted(rng.choice(candidates, size=n, replace=False).tolist())


def split_blip(truth, runs, rng, param):
    """n possessions each get a fake opposite-team possession of L frames in the middle."""
    n, length = param
    pred, segs = truth.copy(), segments(truth, runs)
    long_enough = [i for i in possession_indices(segs) if segs[i][1] - segs[i][0] >= 3 * length]
    for i in _pick(rng, long_enough, n):
        a, b, code = segs[i]
        middle = (a + b) // 2 - length // 2
        pred[middle:middle + length] = A if code == H else H
    return pred


def split_dead(truth, runs, rng, param):
    """n possessions each get a fake stoppage of L frames in the middle."""
    n, length = param
    pred, segs = truth.copy(), segments(truth, runs)
    long_enough = [i for i in possession_indices(segs) if segs[i][1] - segs[i][0] >= 3 * length]
    for i in _pick(rng, long_enough, n):
        a, b, _ = segs[i]
        middle = (a + b) // 2 - length // 2
        pred[middle:middle + length] = D
    return pred


def _absorbable(segs):
    """Possessions directly preceded by the other team's possession (a hand-over in)."""
    return [i for i in possession_indices(segs)
            if i > 0 and segs[i - 1][2] in (H, A) and segs[i - 1][1] == segs[i][0]]


def miss_possession(truth, runs, rng, n):
    """n hand-overs missed: the possession is absorbed by the team before it (A H A -> A A A)."""
    pred, segs = truth.copy(), segments(truth, runs)
    for i in _pick(rng, _absorbable(segs), n):
        a, b, _ = segs[i]
        pred[a:b] = segs[i - 1][2]
    return pred


def miss_short(truth, runs, rng, seconds):
    """Every possession shorter than `seconds` absorbed by the team before it."""
    pred, segs = truth.copy(), segments(truth, runs)
    for i in _absorbable(segs):
        a, b, _ = segs[i]
        if b - a < seconds * FPS:
            pred[a:b] = pred[a - 1]
    return pred


def swap_team(truth, runs, rng, n):
    pred, segs = truth.copy(), segments(truth, runs)
    for i in _pick(rng, possession_indices(segs), n):
        a, b, code = segs[i]
        pred[a:b] = A if code == H else H
    return pred


def drop_stoppage(truth, runs, rng, n):
    """n stoppages missed: each half of the dead stretch goes to the possession beside it."""
    pred, segs = truth.copy(), segments(truth, runs)
    inner = [i for i, (_, _, k) in enumerate(segs)
             if k == D and 0 < i < len(segs) - 1
             and segs[i - 1][2] in (H, A) and segs[i + 1][2] in (H, A)]
    for i in _pick(rng, inner, n):
        a, b, _ = segs[i]
        pred[a:(a + b) // 2] = segs[i - 1][2]
        pred[(a + b) // 2:b] = segs[i + 1][2]
    return pred


def abstain_middle(truth, runs, rng, length):
    """X over `length` frames in the middle of every possession at least 3x that long."""
    pred, segs = truth.copy(), segments(truth, runs)
    for i in possession_indices(segs):
        a, b, _ = segs[i]
        if b - a >= 3 * length:
            middle = (a + b) // 2 - length // 2
            pred[middle:middle + length] = X
    return pred


def abstain_handover(truth, runs, rng, length):
    """X over the first `length` frames after every hand-over (H<->A)."""
    pred, segs = truth.copy(), segments(truth, runs)
    for i in _absorbable(segs):
        a, b, _ = segs[i]
        pred[a:min(a + length, b - 1)] = X
    return pred


# -- the study -----------------------------------------------------------------------------------

CASES = [
    # (group, label, function, param, description)
    ("timing", "late", late, 1, "everything 1 frame late"),
    ("timing", "late", late, 3, "3 frames late"),
    ("timing", "late", late, 8, "8 frames late"),
    ("timing", "late", late, 12, "12 frames (0.5 s) late"),
    ("timing", "late", late, 25, "1 s late"),
    ("timing", "late", late, 50, "2 s late"),
    ("timing", "jitter", jitter, 5, "every boundary moved by a random -5..+5 frames"),
    ("timing", "jitter", jitter, 12, "random -12..+12 frames"),
    ("timing", "jitter", jitter, 25, "random -25..+25 frames"),
    ("timing", "handover_late", handover_late, 12, "only H<->A hand-overs 12 frames late"),
    ("timing", "handover_late", handover_late, 25, "only hand-overs 1 s late"),
    ("timing", "stoppage_late", stoppage_late, 12, "both ends of every stoppage 12 frames late"),
    ("timing", "stoppage_late", stoppage_late, 50, "both ends of every stoppage 2 s late"),
    ("split", "split_blip", split_blip, (1, 5), "1 possession split by a 5-frame fake possession"),
    ("split", "split_blip", split_blip, (1, 25), "1 split by a 1 s fake possession"),
    ("split", "split_blip", split_blip, (5, 25), "5 splits by 1 s fake possessions"),
    ("split", "split_dead", split_dead, (1, 25), "1 possession split by a 1 s fake stoppage"),
    ("split", "split_dead", split_dead, (5, 25), "5 splits by 1 s fake stoppages"),
    ("merge", "miss_possession", miss_possession, 1, "1 hand-over missed (A H A -> A A A)"),
    ("merge", "miss_possession", miss_possession, 5, "5 hand-overs missed"),
    ("merge", "miss_short", miss_short, 1, "every possession < 1 s missed"),
    ("merge", "miss_short", miss_short, 2, "every possession < 2 s missed"),
    ("merge", "miss_short", miss_short, 4, "every possession < 4 s missed"),
    ("merge", "drop_stoppage", drop_stoppage, 1, "1 stoppage missed (split between neighbours)"),
    ("team", "swap_team", swap_team, 1, "1 possession given to the wrong team"),
    ("team", "swap_team", swap_team, 5, "5 possessions given to the wrong team"),
    ("abstain", "abstain_middle", abstain_middle, 50, "X for 2 s mid-possession, everywhere"),
    ("abstain", "abstain_handover", abstain_handover, 12, "X for 0.5 s after every hand-over"),
    ("abstain", "abstain_handover", abstain_handover, 25, "X for 1 s after every hand-over"),
]

RANDOM = {jitter, split_blip, split_dead, miss_possession, swap_team, drop_stoppage}

# EVALUATION.md, "perfect but k frames late", scored by the organisers on the full match.
OFFICIAL_LATENESS = ((1, 1.0000), (3, 0.9818), (8, 0.8933), (25, 0.6539))


def measure(truth, function, param, seeds):
    """Mean (spellF1, macroF1, predicted possessions) over seeds; one draw if deterministic."""
    pm = submission.metrics()
    runs = runs_of(truth)
    draws = seeds if function in RANDOM else 1
    spell, macro, count = [], [], []
    for seed in range(draws):
        pred = function(truth, runs, np.random.default_rng(seed), param)
        result = pm.evaluate(truth, pred, bootstrap_n=0)
        spell.append(result["spell_f1_mean"])
        macro.append(result["macro_f1"])
        count.append(result["n_pred_spells"])
    return float(np.mean(spell)), float(np.mean(macro)), float(np.mean(count))


def lateness_curve(truth):
    pm = submission.metrics()
    runs = runs_of(truth)
    ks = list(range(0, 51, 1))
    return ks, [pm.evaluate(truth, late(truth, runs, None, k))["spell_f1_mean"] for k in ks]


def iou_tolerance_table():
    """Frames of lateness a possession of a given length survives at each IoU threshold."""
    rows = []
    for seconds in (1, 2, 4, 8, 15, 30):
        length = seconds * FPS
        # A possession delayed by k keeps IoU (L - k) / (L + k).
        rows.append((seconds, [int(np.floor(length * (1 - t) / (1 + t)))
                               for t in (0.3, 0.4, 0.5, 0.6, 0.7)]))
    return rows


def write_figure(path, public_curve, tiled_curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=110)
    ax.plot(public_curve[0], public_curve[1], color="#1f77b4", lw=2, label="public 5 min")
    ax.plot(tiled_curve[0], tiled_curve[1], color="#d62728", lw=2, ls="--",
            label="tiled to a full match")
    ax.plot([k for k, _ in OFFICIAL_LATENESS], [v for _, v in OFFICIAL_LATENESS], "o",
            color="#222222", label="real full match (EVALUATION.md)")
    for x in (8, 25):
        ax.axvline(x, color="#999999", lw=0.8)
    ax.set_xlabel("every boundary late by k frames (25 frames = 1 s)")
    ax.set_ylabel("spellF1")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 50)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("spellF1 against a uniform delay")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def render(rows, public, tiled, figure_name):
    pm = submission.metrics()
    pub_spells = pm.possession_spells(public, runs_of(public))
    tiled_spells = pm.possession_spells(tiled, runs_of(tiled))
    n_tiled = len(tiled_spells)
    lines = [
        "# How spellF1 reacts to mistakes",
        "",
        "Generated by `python -m saha.tools.metric_behaviour` from the official scorer "
        "(`evaluation/possession_metrics.py`, unmodified). Do not edit by hand: change the tool "
        "and re-run it.",
        "",
        "This is a study of the **metric**, not of a model. Use it to reason about which "
        "mistakes to avoid. Never use the public window to choose between two versions of the "
        "pipeline (CLAUDE.md).",
        "",
        "## What the scorer does, in seven lines",
        "",
        "1. Only frames where the ground truth is not X are graded. That means the two halves "
        "11185–90651 and 110808–190197; on the public file it means 11185–18684 only.",
        "2. Your X labels are forward-filled with your last label in the same half (back-filled "
        "at the head). spellF1 never sees an X. macroF1 counts every X on a graded frame as a "
        "miss.",
        "3. A possession is a maximal run of H or of A. D separates possessions: `H D H` is two "
        "possessions.",
        "4. Predicted and true possessions of the **same team** are paired by temporal IoU. "
        "The pairing is greedy, highest IoU first, and one-to-one.",
        "5. F1 = 2TP / (2TP + FP + FN) at IoU 0.3, 0.4, 0.5, 0.6 and 0.7. spellF1 is the mean "
        "of the five.",
        "6. Every possession counts once. A 1-second possession weighs as much as a 40-second "
        "one.",
        "7. `validate_submission.py` counts possessions with each X as a gap. The scorer bridges "
        "X, so its count can be lower.",
        "",
        "## References used below",
        "",
        "| reference | possessions | mean duration |",
        "|---|---|---|",
        "| public (the 5 minutes as published) | {} | {:.1f} s |".format(
            len(pub_spells), np.mean([b - a for a, b, _ in pub_spells]) / FPS),
        "| tiled (public pattern repeated over both halves) | {} | {:.1f} s |".format(
            len(tiled_spells), np.mean([b - a for a, b, _ in tiled_spells]) / FPS),
        "",
        "The public window has possessions of 0.8 s and 1.3 s. The real match will have many "
        "short ones too. The tiled reference only rescales the counts; its possession count is "
        "an extrapolation, not the real match's (which is not published).",
        "",
        "Random mistakes (jitter, picked possessions) are averaged over {} seeds on public and {} "
        "on tiled. **Δ/mistake** is the tiled loss divided by the number of mistakes: read it as "
        "the full-match price of one more such mistake.".format(SEEDS_PUBLIC, SEEDS_TILED),
        "",
    ]
    groups = {"timing": "Timing: late, early and noisy boundaries",
              "split": "False splits: one possession reported as several",
              "merge": "Merges: possessions or stoppages missed",
              "team": "Wrong team",
              "abstain": "Abstaining with X"}
    for group, title in groups.items():
        lines += ["## " + title, "",
                  "| mistake | public spellF1 | public macroF1 | tiled spellF1 | Δ/mistake (tiled) "
                  "| tiled possessions |", "|---|---|---|---|---|---|"]
        for row in rows:
            if row["group"] != group:
                continue
            per = ""
            if row["count"]:
                per = "{:+.4f}".format((row["tiled_spell"] - 1.0) / row["count"])
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {} | {:.0f} vs {} |".format(
                row["description"], row["public_spell"], row["public_macro"],
                row["tiled_spell"], per, row["tiled_count"], len(tiled_spells)))
        lines.append("")

    lines += [
        "![spellF1 against a uniform delay](figures/{})".format(figure_name),
        "",
        "## How late can a boundary be?",
        "",
        "A possession of L frames delayed by k frames keeps IoU (L − k)/(L + k). It still matches "
        "at threshold τ while k ≤ L(1 − τ)/(1 + τ). That gives these frame budgets:",
        "",
        "| possession | τ=0.3 | τ=0.4 | τ=0.5 | τ=0.6 | τ=0.7 |",
        "|---|---|---|---|---|---|",
    ]
    for seconds, budget in iou_tolerance_table():
        lines.append("| {} s | {} |".format(seconds, " | ".join(str(b) for b in budget)))
    lines += [
        "",
        "Short possessions decide the score. A 1 s possession must be within 4 frames to count "
        "at τ=0.7, and a 2 s one within 8.",
        "",
        "## The real match is more sensitive than these references",
        "",
        "EVALUATION.md publishes the same 'everything late' test scored on the **full match**. "
        "It loses more than either of our references:",
        "",
        "| everything late by | full match (EVALUATION.md) | public | tiled |",
        "|---|---|---|---|",
    ]
    by_key = {(r["label"], r["param"]): r for r in rows}
    for frames, official in OFFICIAL_LATENESS:
        row = by_key.get(("late", frames))
        if row:
            lines.append("| {} frames | {:.4f} | {:.4f} | {:.4f} |".format(
                frames, official, row["public_spell"], row["tiled_spell"]))
    lines += [
        "",
        "So the full match is more sensitive to boundary timing than the public window. The "
        "likely cause is that it has proportionally more very short possessions. Treat every "
        "number on this page as **optimistic**: boundary precision is worth more than it "
        "looks here.",
        "",
        "Do not reverse-engineer private ground-truth statistics (possession count, duration "
        "distribution) from published numbers like these and tune against them. The organisers "
        "withhold those statistics on purpose (OVERVIEW.md), and the team rule is the same "
        "(CLAUDE.md).",
        "",
        "## Rules of thumb for the team",
        "",
        "Each price below is measured on the tiled reference ({} possessions). It is given "
        "both as spellF1 and as the number of possessions lost (Δ × possessions). "
        "Re-run the tool if the scorer changes.".format(n_tiled),
        "",
    ]

    def price(label, param):
        row = by_key[(label, param)]
        delta = (row["tiled_spell"] - 1.0) / row["count"]
        return "{:+.4f} (≈ {:.1f} possessions)".format(delta, -delta * n_tiled)

    lines += [
        "- **Every mistake costs about one to two possessions.**",
        "    - Inventing a short opposite-team possession inside a real one: {}.".format(
            price("split_blip", (1, 25))),
        "    - Missing a hand-over (`A H A → A A A`): {}.".format(
            price("miss_possession", 1)),
        "    - Giving a possession to the wrong team: {}.".format(price("swap_team", 1)),
        "    - Inventing a stoppage inside a possession: {}.".format(
            price("split_dead", (1, 25))),
        "    - Missing a stoppage: {}.".format(price("drop_stoppage", 1)),
        "  When the evidence for a short possession is a coin flip, reporting it and missing it "
        "cost about the same. Decide on the evidence, not on a rule of thumb.",
        "- **Short possessions are where the score is.** Losing every possession under 2 s "
        "costs {:.3f} on the tiled reference, and more on the real match (see above). A model "
        "that never reports short possessions caps itself.".format(
            1.0 - by_key[("miss_short", 2)]["tiled_spell"]),
        "- **Boundary error is graded per possession.** 8 frames is invisible on a 30 s "
        "possession. It pushes a 2 s possession out of the τ=0.7 match. A uniform 8-frame delay "
        "costs {:.3f} here and {:.3f} on the real match.".format(
            1.0 - by_key[("late", 8)]["tiled_spell"], 1.0 - dict(OFFICIAL_LATENESS)[8]),
        "- **X in mid-possession is free for spellF1.** It still costs macroF1, the first "
        "tie-break: 2 s of X in every possession drops public macroF1 to {:.3f}.".format(
            by_key[("abstain_middle", 50)]["public_macro"]),
        "- **X after a hand-over costs exactly as much as being late.** The fill carries the "
        "old team forward. Abstain only where the team is not about to change.",
        "- **Do not trust small gaps.** On the public window one possession is about 1/17 of "
        "the score. On the full match, gaps below ~0.04 are noise (EVALUATION.md). Run "
        "`--bootstrap 1000` before believing any difference.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(paths.DOCS_DIR / "METRIC_NOTES.md"))
    args = parser.parse_args(argv)

    public = submission.public_gt()
    tiled = tiled_truth(public)
    rows = []
    for group, label, function, param, description in CASES:
        p_spell, p_macro, _ = measure(public, function, param, SEEDS_PUBLIC)
        t_spell, _, t_count = measure(tiled, function, param, SEEDS_TILED)
        count = param if label in ("swap_team", "miss_possession", "drop_stoppage") else (
            param[0] if label.startswith("split") else None)
        rows.append({"group": group, "label": label, "param": param,
                     "description": description, "public_spell": p_spell,
                     "public_macro": p_macro, "tiled_spell": t_spell, "tiled_count": t_count,
                     "count": count})
        print("{:<52} public {:.4f}  tiled {:.4f}".format(description, p_spell, t_spell))

    out = Path(args.out)
    figure_name = "metric_lateness.png"
    write_figure(out.parent / "figures" / figure_name, lateness_curve(public),
                 lateness_curve(tiled))
    out.write_text(render(rows, public, tiled, figure_name), encoding="utf-8")
    print("wrote {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
