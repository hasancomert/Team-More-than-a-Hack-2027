---
date: 2026-11-13
package: ws1
name: shape-model
author: Hasan (WS1), Claude Code (code, runs and record); Seyyid (WS5, the same idea in parallel, review and independent reproduction)
commit: 757ba6b
config: configs/pipeline.toml
public_spell_f1:
public_spell_low:
public_spell_high:
public_macro_f1:
n_possessions:
mean_duration_s:
d_rate:
verdict: adopted
---

## What changed

A new evidence source, `saha/shape` (DECISIONS D026). It says which team has the ball where the
ball is missing, from the two teams' shape and motion, and never reads the ball.

- **The model.** A symmetric logistic model with 24 coefficients, in `saha/shape/model.json`.
- **Training.** `python -m saha.shape.train` fits it on WS5's seven DFL matches only.
- **Where it writes.** `p_home` goes on graded frames without a tracked ball, clipped to
  [0.1, 0.9]. `configs/pipeline.toml` gives it scale 2, next to WS4.
- **The orientation guard.** In a half where the two teams' players nearest one goal are less
  than 10 m apart (median), the direction of attack is unclear, and the half gets no opinion.

Details: `docs/tech/ws1_pipeline.md`.

## Why

WS1 measured where the frozen version (`225cae0`) loses spellF1. The runs used WS5's seven DFL
matches, 3 degradation seeds, and the truth substituted for one part at a time:

| that part perfect | spellF1 (raw labels, seed 0) |
|---|---|
| nothing: the frozen version | 0.394 |
| the dead ball | 0.451 |
| the team, only where the ball is missing | 0.629 |
| the team, only where the ball is tracked | 0.661 |
| the team everywhere | 0.849 |

- **The team where the ball is missing is a large loss.** 33% of the true possessions lie almost
  entirely in hidden-ball stretches, and the frozen version matched 3% of them.
- **WS4's bridge was close to a coin flip there.** It got the team right on 62–73% of those
  frames.
- **No setting fixed it.** `cost_switch`, every scale and two post-filters could not move the
  score by more than ±0.01.

So the missing part was a new signal: something that sees the team in possession without the
ball.

**The rule, fixed before the remaining runs** (8 of 21 pipeline runs were done when it was
written).

- **Variant:** the logistic model trained on the controlled-2s labels, on hidden graded frames
  only, at scale 2 (as strong as WS3's weight 2), with WS4 kept. It was chosen by reasoning,
  not by the numbers.
- **Adoption:** adopt it if, on raw labels and on controlled-2s labels, the 7-match mean over 3
  seeds is above the frozen version's, and it is ahead on at least 6 of 7 matches for every seed.
- **The other variants** (scales 1 and 4, without WS4, on all graded frames, gradient boosting)
  are reported and were not chosen from.

## Result

Every number is out of sample. For each DFL match, the model is fitted by `saha.shape.train` on
the six others, then run by `saha.shape.build` inside the frozen pipeline. The degradation used
seeds 0, 1 and 2; point estimates.

**Frames where the ball is missing** (held-out match): the team is right on 82–91% of alive
frames (by match: 0.879, 0.823, 0.888, 0.865, 0.891, 0.888, 0.907). WS4's bridge gets 62–73% on
the same frames.

**spellF1, mean of the 7 matches:**

| labels | seed 0 | seed 1 | seed 2 | 3-seed mean | ahead |
|---|---|---|---|---|---|
| raw, frozen | 0.3937 | 0.4020 | 0.3970 | 0.3976 | |
| raw, + shape | **+0.0989** | **+0.0866** | **+0.0943** | **0.4909** | 7, 7, 7 of 7 |
| controlled-1s, frozen | 0.4169 | 0.4263 | 0.4214 | 0.4216 | |
| controlled-1s, + shape | +0.1035 | +0.0914 | +0.1016 | 0.5204 | 7, 7, 7 of 7 |
| controlled-2s, frozen | 0.4273 | 0.4346 | 0.4356 | 0.4325 | |
| controlled-2s, + shape | **+0.1032** | **+0.0944** | **+0.1031** | **0.5327** | 7, 7, 7 of 7 |

- **Every run gains.** All 21 runs gain, by +0.062 to +0.137 on raw labels and by +0.062 to
  +0.135 on controlled-2s.
- **macroF1 and team accuracy rise too.** On seed 0, macroF1 goes from 0.785 to 0.846, and the
  team accuracy where both are alive from 0.784 to 0.880.
- **Predicted possessions** rise from 1,660 to 2,000 over the seven matches. The true count is
  3,061 raw, or 2,335 on controlled-2s.
- **Scale.** Scale 4 adds another +0.003 to +0.008 and scale 1 loses about 0.01; scale 2 stays
  as fixed.
- **The orientation guard never triggered on DFL.** The two teams' medians were 21–27 m apart in
  every half, and the model covered 100% of the hidden graded frames.

The rule holds on both label sets, so the variant is adopted.

**Training with missing players too** (second rule, fixed before its run).

- **The test.** DFL tracks all 22 players; our tracking has about 20.5 per graded frame. WS1
  removed players in runs (mean 5 s) from each held-out match. The model trained on complete
  matches lost accuracy where the ball is hidden: 0.884 with every player, 0.845 at 7% of
  player-frames removed (about our level), 0.801 at 15%.
- **The rule.** Add each training match once more with 10% of player-frames removed (its own
  seed), and adopt this if, held out, the accuracy at 7% rises by at least 0.010 while the
  accuracy with every player falls by no more than 0.005.

The result, held out (mean of 7 matches, alive frames where the ball is hidden):

| players removed | trained on complete matches | trained with missing players too |
|---|---|---|
| 0% | 0.884 | 0.882 (−0.002) |
| 7% | 0.856 | **0.876 (+0.021)** |
| 15% | 0.818 | **0.864 (+0.045)** |

- **The rule holds,** so `saha/shape/model.json` is trained this way.
- **Inside the pipeline on complete DFL matches** (leave one match out, 3 seeds), the version
  adopted here scores 0.4898 on raw labels and 0.5325 on controlled-2s. The frozen version scores
  0.3976 and 0.4325. The new version is ahead on 21 of 21 runs on both label sets.

**On our match** (checks only). This version was not run alone: the submission at `d276615`
holds it with D027 and D028 on top, identical on Hasan's and Baki's machines. There
`saha.shape.build` finds the direction of attack with margins of 35.2 m and 30.0 m (no
warning), has an opinion on 96.3% of the 64,643 graded frames without the ball, home on 42.7%
of them.

## Caveats

- **Seven matches, one league.** All are German matches of 2022/23, five of them with the same
  home team. Barcelona and Athletic play differently.
- **Tracking completeness.** DFL tracks all 22 players on every frame; our tracking has about
  20.5 per graded frame. The features use what is present, and a half without a clear
  orientation gets no opinion.
- **The labels are DFL's,** made stricter by WS1's proxy, not WS5's relabelling.
- **The degradation is synthetic:** the ball is hidden at random, and the players stay perfectly
  tracked.
- **This is a change after the freeze** (D026). If it is not merged and reproduced on two
  machines by 14 November 18:00 Turkey time, the frozen version ships.

## Decision

Adopted under D026: `shape` at scale 2, next to WS4, in `configs/pipeline.toml`, trained with
missing players too. Both rules were fixed before their runs, and both hold out of sample. The
public 5 minutes played no part (CLAUDE.md, rule 2).
