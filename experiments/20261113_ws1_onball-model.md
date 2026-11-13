---
date: 2026-11-13
package: ws1
name: onball-model
author: Hasan (WS1), Claude Code (code, runs and record)
commit:
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

A new evidence source, `saha/onball` (DECISIONS D027). It says which team has the ball where the
ball is tracked, from the ball and the players around it, with the two teams' shape as context.

- **The model.** scikit-learn's HistGradientBoostingClassifier (300 trees of up to 31 leaves), in
  `saha/onball/models/onball.pkl` with its manifest (D020).
- **88 inputs** (`saha/onball/features.py`):
    - **per team, in that team's direction of attack:** the distance from the ball to its nearest
      and second-nearest player, its players within 5 m and within 10 m, the nearest player's
      speed towards the ball, the ball's x and the ball's velocity towards the team's goal. Each
      is averaged over four windows: centred 0.5 s, centred 2 s, the second before and the second
      after;
    - **the ball:** its speed, acceleration and position, raw and over 1 s;
    - **the shape:** `saha.shape`'s 24 features, home minus away.
- **Symmetry.** Every training frame is added a second time with the teams exchanged, and the
  answer is the mean of the model's answers in both orders. Exchanging the teams turns `p` into
  `1 − p`.
- **Training.** `python -m saha.onball.train` fits it on WS5's seven DFL matches only, every 5th
  alive frame (221,142 rows with the exchanged copies). The labels are DFL's possession, with
  every possession under 2 s between two of the other team merged into them. One thread,
  `random_state` 0, no early stopping.
- **Where it writes.** `p_home` goes on graded frames where the ball is tracked, clipped to
  [0.1, 0.9], at weight 1. `configs/pipeline.toml` gives it scale 2, next to WS3 and `nearest`. It
  is silent on the few frames at each edge of a tracking gap, where the ball's speed is
  undefined; WS3 and `nearest` still cover them.

Details: `docs/tech/ws1_pipeline.md`.

## Why

With the shape model in (D026), WS1 measured again where the version loses spellF1. The runs used
WS5's seven DFL matches and 3 degradation seeds, with the truth substituted for one part at a
time (the shape model fitted out of sample):

| that part perfect | spellF1, raw labels | controlled-2s |
|---|---|---|
| nothing: the shape version | 0.490 | 0.533 |
| the dead ball | 0.570 | 0.625 |
| the team, only where the ball is missing | 0.621 | 0.623 |
| **the team, only where the ball is tracked** | **0.738** | **0.747** |
| the team everywhere | 0.845 | 0.831 |
| the dead ball, and the team where the ball is tracked | 0.840 | 0.866 |

- **The team where the ball is tracked is now the largest single loss,** +0.25 on raw labels.
- **The nearest player is wrong too often there.** Held out, it is right on 78% of the alive
  frames where the ball is tracked. WS3 and `nearest` both start from it, so they inherit its
  errors around passes, duels and loose balls.
- **A learned model does better on the same frames.** Leave one match out (seed 0, alive tracked
  frames, controlled-2s labels): nearest player 0.780, a logistic model on the inputs above
  0.887, gradient boosting 0.908.

**The rule, fixed before any pipeline result was read.** The runs had started; none of their
output had been read.

- **Variant:** gradient boosting, chosen on held-out frame accuracy only. It runs on graded frames
  where the ball is tracked, at scale 2 (as for WS3 and the shape model), and keeps `nearest` and
  `visible`.
- **Adoption:** on raw labels and on controlled-2s labels, the 7-match mean over 3 seeds must be
  above the shape version's, and the variant must be ahead on at least 6 of the 7 matches for
  every seed.
- **The other variants** are reported and were not chosen from: scales 1 and 4, without `nearest`
  and `visible`, and the logistic model.

## Result

Every number is out of sample. For each DFL match, the model is fitted on the six others and run
inside the shape version's pipeline, with the shape model also fitted on the six others. The
degradation used seeds 0, 1 and 2; point estimates.

**spellF1, mean of 7 matches × 3 seeds** ("ahead": matches ahead of the shape version, per seed):

| variant | raw | Δ | ahead | controlled-2s | Δ | ahead |
|---|---|---|---|---|---|---|
| the shape version | 0.4898 | | | 0.5325 | | |
| **gradient boosting, scale 2 (the rule)** | **0.5303** | **+0.041** | **7, 7, 7** | **0.5857** | **+0.053** | **7, 7, 7** |
| gradient boosting, scale 1 | 0.5101 | +0.020 | 7, 7, 7 | 0.5589 | +0.026 | 7, 7, 7 |
| gradient boosting, scale 4 | 0.5398 | +0.050 | 7, 7, 7 | 0.5914 | +0.059 | 7, 7, 7 |
| gradient boosting, scale 2, no `nearest`/`visible` | 0.5253 | +0.036 | 7, 7, 6 | 0.6044 | +0.072 | 7, 7, 7 |
| gradient boosting, scale 4, no `nearest`/`visible` | 0.5365 | +0.047 | 7, 7, 6 | 0.6008 | +0.068 | 7, 7, 7 |
| logistic, scale 1 | 0.5038 | +0.014 | 6, 7, 6 | 0.5572 | +0.025 | 7, 7, 7 |
| logistic, scale 2 | 0.5182 | +0.028 | 7, 7, 6 | 0.5784 | +0.046 | 7, 7, 7 |
| logistic, scale 4 | 0.5245 | +0.035 | 7, 7, 7 | 0.5873 | +0.055 | 7, 7, 7 |
| logistic, scale 2, no `nearest`/`visible` | 0.4909 | +0.001 | 3, 4, 5 | 0.5817 | +0.049 | 7, 6, 6 |
| logistic, scale 4, no `nearest`/`visible` | 0.5092 | +0.019 | 5, 6, 6 | 0.5845 | +0.052 | 7, 7, 7 |

- **The rule holds on both label sets,** so the variant is adopted. On controlled-1s labels it
  gains +0.047 (7, 7, 7).
- **Fewer, better possessions.** The mean predicted count per match falls from 282 to 265: the
  model removes false hand-overs more than it adds true ones.
- **Scale.** Scale 4 adds another +0.010 on raw labels and +0.006 on controlled-2s. Scale 2 stays
  as fixed.

**The repository package, checked the same way.** The table above comes from WS1's prototype.
`saha.onball.train.fit` and `saha.onball.build` were then run on the same 21 held-out cases,
and they reproduce it:

| labels | the shape version | + `saha.onball`, scale 2 | Δ | ahead |
|---|---|---|---|---|
| raw | 0.4898 | 0.5309 | +0.041 | 7, 7, 7 |
| controlled-1s | 0.5195 | 0.5666 | +0.047 | 7, 7, 7 |
| controlled-2s | 0.5325 | 0.5868 | +0.054 | 7, 7, 7 |

- Each run is within 0.010 of the prototype's; on average the package is 0.001 higher.
- **Frame accuracy** at seed 0, on alive frames with the ball: 0.908 (by match: 0.907, 0.888,
  0.911, 0.894, 0.913, 0.914, 0.928), as the prototype's.
- **Coverage:** the model has an opinion on 98.5% of the graded frames with the ball; it is
  silent on the frames next to a gap.

**Missing players** (second rule, fixed before its run).

- **The test.** DFL tracks all 22 players; our tracking about 20.5 per graded frame. For each
  held-out match, players were removed in runs (mean 5 s, `saha.shape.train.drop_players`) at
  0%, 7% (about our level) and 15% of player-frames, and the accuracy was measured on alive
  frames with the ball (seed 0).
- **The rule.** Retrain with every match also added once with 10% of player-frames removed, as
  for the shape model, and adopt this if, held out, the accuracy at 7% rises by at least 0.010
  while the accuracy with every player falls by no more than 0.005.

| players removed | trained on complete matches | trained with missing players too |
|---|---|---|
| 0% | 0.908 | 0.908 (−0.000) |
| 7% | 0.897 | 0.902 (+0.005) |
| 15% | 0.888 | 0.899 (+0.011) |

- **The rule does not hold** (+0.005 at 7%, against the 0.010 required), so the model stays
  trained on complete matches. The variant was ahead on all 7 matches at 7%, by little.
- **The model is robust to missing players:** it loses 0.011 at our level, where the shape
  model lost 0.03 before it was trained with missing players.

**On our match:** to be run by the team leader with `python -m saha.pipeline --fresh`. Checks
only:

- `saha.onball.build`'s line: its opinion should cover nearly all tracked graded frames;
- the match-wide statistics;
- the new SHA256, then reproduced on a second machine.

## Caveats

- **Seven matches, one league,** as for the shape model (D026).
- **The ball.** DFL tracks it on every frame. Ours is tracked on about 59% of graded frames, and
  may be noisier. The windows skip frames without the ball, and the held-out runs hide it at
  random; real tracking loses it more often in crowded play, so our tracked frames may be harder.
- **Missing players.** DFL tracks all 22 players. A missing player near the ball changes the
  nearest-player distances. At our level (7% of player-frames removed at random) the model
  loses 0.011 of accuracy; if our tracking misses players near the ball more often than at
  random, it loses more.
- **The labels are DFL's,** made stricter by WS1's proxy, not WS5's relabelling.
- **This is a second change after the freeze** (D027). If it is not merged and reproduced on two
  machines by 14 November 18:00 Turkey time, the frozen version ships.

## Decision

Adopted under D027: `onball` at scale 2, next to WS3 and `nearest`, in `configs/pipeline.toml`,
trained on complete matches. The first rule was fixed before the pipeline results were read and
holds out of sample; the second, fixed before its run, does not hold, so the training stays as
it was. The public 5 minutes played no part (CLAUDE.md, rule 2).
