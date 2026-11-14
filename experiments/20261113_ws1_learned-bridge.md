---
date: 2026-11-13
package: ws1
name: learned-bridge
author: Hasan (WS1), Claude Code (code, runs and record)
commit: 480d284
config: configs/pipeline.toml (unchanged)
public_spell_f1:
public_spell_low:
public_spell_high:
public_macro_f1:
n_possessions:
mean_duration_s:
d_rate:
verdict: rejected
---

## What changed

Nothing in the submission. WS1 tested, in its scratchpad, a learned bridge where the ball is
missing, as a replacement for `shape`.

- **Inputs** on every frame inside a gap in the ball's tracking: `shape`'s 24 features (home
  minus away); where in the gap the frame is (seconds since its start, until its end, its length,
  capped at 60 s); and each edge of the gap over 1 s of tracking (nearest home minus nearest away
  distance to the ball, the ball's x and x-velocity in home's direction, the ball's speed).
- **Model:** the on-ball model's gradient boosting, symmetric between the teams.
- **Training:** the six other DFL matches degraded at seeds 100–102 (not the evaluation seeds),
  alive frames inside gaps, every 3rd frame, controlled-2s labels.

## Why

After the on-ball model, the true team where the ball is missing would raise spellF1 from 0.531
to 0.662 (raw) and from 0.587 to 0.677 (controlled-2s). `shape` reads the formation but not the
edges of the gap; WS4's bridge reads the edges but not the formation.

**The rule, fixed before any run:** replace `shape` at scale 2 (same frames), keep WS4,
`onball`, `nearest` and `visible`; adopt if, on raw and controlled-2s labels, the 7-match mean over
3 seeds is above the on-ball version's and the bridge is ahead on at least 6 of 7 matches for
every seed. Reported and not chosen from: frame accuracy, scales 1 and 4, the bridge next to
`shape`.

## Result

Each DFL match held out in turn, evaluation seeds 0–2:

| variant | raw | Δ | ahead | controlled-2s | Δ | ahead |
|---|---|---|---|---|---|---|
| the on-ball version | 0.5309 | | | 0.5868 | | |
| **bridge x2, replacing `shape` (the rule)** | 0.5340 | +0.003 | 5, 5, 5 | 0.5892 | +0.002 | 5, 5, 5 |
| bridge x1, replacing `shape` | 0.5238 | −0.007 | 1, 1, 1 | 0.5821 | −0.005 | 3, 3, 1 |
| bridge x4, replacing `shape` | 0.5363 | +0.005 | 6, 5, 4 | 0.5900 | +0.003 | 5, 4, 5 |
| bridge x2 next to `shape` | 0.5367 | +0.006 | 6, 6, 5 | 0.5906 | +0.004 | 5, 6, 5 |

- **Frame accuracy where the ball is missing** (seed 0): 0.884 for the bridge, 0.880 for `shape`.
  The edges and the position in the gap add almost nothing to the formation.
- **The rule does not hold** (5 of 7 matches per seed on both label sets).

## Decision

Rejected. The submission keeps `shape`. The public 5 minutes played no part.
