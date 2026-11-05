---
date: 2026-11-05
package: ws1
name: exact-emissions
author: Claude Code (WS1), run by Hasan
commit:
config: configs/pipeline.toml
public_spell_f1:
public_spell_low:
public_spell_high:
public_macro_f1:
n_possessions:
mean_duration_s:
d_rate:
verdict: open
---

## What changed

- `saha/combine/viterbi.py`: the H and A scores now subtract n = 2 log(2 cosh(home / 2)), so
  H and A share the alive mass.
- `prior_dead` goes from 0 to -0.693 nats, in `CombineParams` and in `configs/pipeline.toml`.

## Why

The evidence interface defines `p_dead` = P(dead) and `p_home` = P(home | alive). So the fused
tracks are dead = log P(D) / P(alive) and home = log P(H | alive) / P(A | alive), and the exact
doubled log-probabilities are:

    D: dead      H: -dead + home - n      A: -dead - home - n

- **The bug.** Without n, max(H, A) = -dead + |home|. A confident `p_home` therefore also pushed
  the alive states against D, by up to |home| per frame. WS4 found it (PR #7). WS2's `p_dead`
  would have had to fight every confident `p_home` from WS3 and WS4.
- **The prior.** With n and the old prior of 0, a frame without evidence would favour D:
  P(D) = 1/2 and P(H) = P(A) = 1/4. A prior of -0.693 nats, so P(dead) = 1/3, makes the three
  labels tie on such a frame, as they did before. The value is exactly half of n(0) as
  tabulated (1.386 nats), so the tie is exact.

Both values follow from the model's definition, not from this match (CLAUDE.md, rule 2).

## Expected effect

- **H versus A:** unchanged, because both scores get the same n.
- **Frames without `p_home`** (no ball): unchanged, because the prior cancels n(0).
- **Frames with `p_home`:** knowing the team gives the alive states at most 2 log 2 = 1.386
  (doubled) nats over D, instead of |home|.
- **The current submission should barely change.** With the nearest-player evidence (`p_home`
  0.8, and `p_dead` 0.1 where the ball is seen), the alive margin on a tracked frame goes from
  5.78 to 5.33 doubled nats.

## Result

TO DO (team leader, on the data): run `python -m saha.pipeline`, then fill in the front matter.
These numbers are checks only:

- the pipeline validates;
- the possession count, mean duration and D rate stay close to 368, 11.7 s and 32.3%.

## Decision

Open until the run. Adopt if the pipeline validates and the match-wide statistics show no
breakage. The public score does not decide (CLAUDE.md, rule 2).
