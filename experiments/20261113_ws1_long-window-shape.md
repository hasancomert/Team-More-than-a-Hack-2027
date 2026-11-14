---
date: 2026-11-13
package: ws1
name: long-window-shape
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

Nothing in the submission. WS1 tested, in its scratchpad, the shape model with two longer
smoothing windows (10 s and 30 s, next to 1 s and 4 s: 40 features instead of 24), fitted exactly
as `saha.shape.train` fits the current model, as a replacement for `shape`. It was the second
candidate where the ball is missing, tested because the learned bridge
(`experiments/20261113_ws1_learned-bridge.md`) did not pass its rule; the order was fixed before
either was run.

## Why

The gaps in the ball's tracking are long (in WS5's degraded DFL matches, half of the missing
frames lie in gaps longer than about 45 s), and `shape` smooths over 4 s at most.

**The rule, fixed before any run:** replace `shape` at scale 2 (same frames), keep WS4, `onball`,
`nearest` and `visible`; adopt if, on raw and controlled-2s labels, the 7-match mean over 3 seeds
is above the on-ball version's and the variant is ahead on at least 6 of 7 matches for every seed.

## Result

Each DFL match held out in turn, evaluation seeds 0–2:

| variant | raw | Δ | ahead | controlled-2s | Δ | ahead |
|---|---|---|---|---|---|---|
| the on-ball version | 0.5309 | | | 0.5868 | | |
| **long shape x2 (the rule)** | 0.5288 | −0.002 | 1, 2, 0 | 0.5855 | −0.001 | 2, 2, 1 |
| long shape x1 | 0.5207 | −0.010 | 0, 0, 0 | 0.5797 | −0.007 | 2, 1, 0 |
| long shape x4 | 0.5338 | +0.003 | 4, 4, 4 | 0.5891 | +0.002 | 5, 5, 3 |

- **Frame accuracy where the ball is missing** (seed 0): 0.882, against 0.880 for `shape`.
- **The rule does not hold.**
- **Together with the learned bridge,** this suggests that the formation says what it can about
  the team in possession while the ball is missing (about 88% of frames); the errors left are
  hand-overs inside the gaps, which need the ball.

## Decision

Rejected. The submission keeps `shape`. The public 5 minutes played no part.
