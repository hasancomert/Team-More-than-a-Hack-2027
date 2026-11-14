---
date: 2026-11-13
package: ws1
name: hiddendead
author: Hasan (WS1), Claude Code (code, runs and record)
commit: d276615
config: configs/pipeline.toml
public_spell_f1: 0.6588
public_spell_low: 0.49
public_spell_high: 0.81
public_macro_f1: 0.8950
n_possessions: 361
mean_duration_s: 9.7
d_rate: 0.448
verdict: adopted
---

## What changed

A new evidence source, `saha/hiddendead` (DECISIONS D028): a second opinion on WS2's dead calls
where the ball is missing.

- **Where:** graded frames where the ball is missing, WS2 calls the ball dead, and the frame lies
  between WS2's first and last alive frame of its half. Every frame before a half's kick-off
  stays as WS2 decided.
- **What:** on those frames the pipeline drops WS2's `p_dead` (`overrides = "deadball"` in
  `configs/pipeline.toml`, `saha.pipeline.apply_overrides`), and the decoder weighs the model's
  P(dead) instead, clipped to [0.1, 0.9], at weight 1 and scale 2. The source never adds a dead
  call where WS2 says alive. WS2's code is unchanged.
- **The model.** scikit-learn's HistGradientBoostingClassifier with the on-ball model's
  parameters, on 27 inputs that do not depend on which team is which
  (`saha/hiddendead/features.py`):
    - where the frame is in its gap in the ball's tracking and in WS2's dead spell;
    - the edges of the gap over 1 s of tracking: the ball's speed, its distance from the halfway
      line and from the nearest touchline, the nearest player's distance to it;
    - the players over 1 s and 4 s: mean speed, the slower team's speed, forward speed, the
      teams' length, width and advancement, and the change of the mean speed.
- **Training.** `python -m saha.hiddendead.train`: WS5's seven DFL matches, each degraded at
  seeds 100–102 (never the evaluation seeds), on the frames defined under "Where", every 3rd
  frame; the target is DFL's ball status. One thread, `random_state` 0, no early stopping. The
  pickle and its D020 manifest are in `saha/hiddendead/models/`.

## Why

With the on-ball model in (7 DFL matches × 3 seeds, held out), the dead ball costs this much
spellF1, by kind of error (each corrected perfectly, the rest as it is):

| corrected | raw | controlled-2s |
|---|---|---|
| nothing: the on-ball version | 0.531 | 0.587 |
| WS2's false dead calls where the ball is missing | 0.569 | 0.632 |
| every false dead call | 0.594 | 0.661 |
| every missed stoppage | 0.549 | 0.607 |
| the whole dead ball | 0.611 | 0.680 |

WS2 calls 21.7% of the live frames dead where the ball is missing, against 4.6% where it is
tracked. Its classifier learned from SkillCorner's 10 fps broadcast tracking; DFL's ball status
on every frame, at 25 fps over the full pitch, can teach a second opinion for exactly those
calls. Each false stoppage splits a possession.

**The rule, fixed before any run** (at the team leader's request):

- **Variant:** as described above, at scale 2 like the other WS1 models.
- **Adoption:** on raw and controlled-2s labels, the 7-match mean over 3 seeds above the on-ball
  version's, and ahead on at least 6 of 7 matches for every seed.
- **Reported and not chosen from:** scales 1 and 4; a full replacement of WS2's decision on all
  missing-ball frames (which can also add dead calls); the frame-level error rates.

## Result

Each DFL match held out in turn (the model fitted on the six others), evaluation seeds 0–2,
inside the on-ball version; point estimates:

| variant | raw | Δ | ahead | controlled-2s | Δ | ahead |
|---|---|---|---|---|---|---|
| the on-ball version | 0.5309 | | | 0.5868 | | |
| **second opinion, scale 2 (the rule)** | **0.5491** | **+0.018** | **7, 7, 7** | **0.6085** | **+0.022** | **7, 7, 7** |
| second opinion, scale 1 | 0.5496 | +0.019 | 7, 7, 7 | 0.6096 | +0.023 | 7, 7, 7 |
| second opinion, scale 4 | 0.5484 | +0.017 | 7, 7, 7 | 0.6074 | +0.021 | 7, 7, 7 |
| full replacement, scale 2 | 0.5473 | +0.016 | 7, 7, 6 | 0.6064 | +0.020 | 7, 7, 6 |

- **Every run gains:** by +0.003 to +0.034 on raw labels and by +0.007 to +0.037 on
  controlled-2s. On controlled-1s labels the gain is +0.019 (7, 7, 7).
- **The rule holds on both label sets,** so the variant is adopted.
- **Frame level,** on the frames where WS2 calls dead with the ball missing (mean of the 21 runs):
  of the live ones, 21.6% stay dead with WS2 alone and 5.6% with the second opinion; of the truly
  dead ones, 92.0% and 87.8% are called dead.
- **Alive/dead F1** of the decoded labels rises from 0.923 to 0.954, and macroF1 (raw, seed 0)
  from 0.858 to 0.881. The mean number of predicted possessions per match rises from 264 to 285.
- **The repository's package reproduces the test exactly.** `saha.hiddendead.features.inputs`
  gives the same frames and the same float32 inputs, bit for bit, as the scripts of the test
  (checked on one match and seed), and training uses the same rows in the same order, so each
  held-out model is the one tested.

**On our match** (checks only; Hasan's and Baki's runs at `d276615` are identical, SHA256
`af770330…`):

- WS2 calls 76,664 graded frames dead; the second opinion covers 55,300 of them (ball missing,
  inside play) and finds 9.7% of those rather alive. On DFL it found about a quarter: here the
  players' movement says that most of WS2's stoppages with the ball missing are real.
- The D rate falls from 48.3% (frozen version) to 44.8% of graded frames; 361 possessions, mean
  9.7 s, median 5.1 s; home share 59.7%.
- The public 5 minutes: spellF1 0.6588 [0.49–0.81], macroF1 0.8950, 17 predicted possessions
  against 17 true ones. Too few possessions to say more; the window chose nothing.

## Caveats

- **The ball goes missing at random in the test.** In our match it may vanish more often when it
  goes out of play, so a missing ball may mean a stoppage more often than on DFL. The model reads
  the players' movement, which shows a stoppage either way, but the second opinion may be too
  ready to call play live there. It can only soften WS2's dead calls, never add one.
- **Seven matches, one league,** as for the other WS1 models.
- **WS2's ownership of `p_dead`** (D011, D022) now has one exception, recorded in D028.
- **This is a third change after the freeze.** If it is not merged and reproduced on two
  machines by 14 November 18:00 Turkey time, the frozen version ships.

## Decision

Adopted under D028: `hiddendead` at scale 2, overriding WS2's `p_dead` on its frames, in
`configs/pipeline.toml`. The rule was fixed before the runs and holds out of sample on every
run. The public 5 minutes played no part (CLAUDE.md, rule 2).
