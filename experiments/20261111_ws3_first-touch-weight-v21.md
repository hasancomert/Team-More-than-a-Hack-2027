---
date: 2026-11-11
package: ws3
name: first-touch-weight-v21
author: Osman (owner, runs on the data), Claude Code (code and record)
commit: 5e105e7
config: configs/pipeline.toml
public_spell_f1: 0.4400
public_spell_low: 0.29
public_spell_high: 0.59
public_macro_f1: 0.8556
n_possessions: 406
mean_duration_s: 8.1
d_rate: 0.483
verdict: open
---

## What changed

v2.1: two settings of `saha/visible/build.py` v2 (20261111_ws3_touch-confirm-v2.md).

- `FIRST_TOUCH_OWNS = True`: a stretch's first touch starts a possession without confirmation.
  A restart is taken by the team with the ball, and after a ball gap nobody else has touched it
  yet. In v2 those frames had no opinion until a hand-over was confirmed.
- `WEIGHT = 2.0`: WS3's per-frame weight, twice `nearest`'s. Both read the same nearest player;
  WS3 adds the possession rules on top, so it should be the louder of the two.
- Tests: two tests now set `FIRST_TOUCH_OWNS = False` explicitly (they test confirmation from a
  stretch's start); the first-touch test also covers a reset. `python -m saha.tools.ci`: 6 of 6
  (92 tests).

## Why

The variants were scored on all seven of WS5's DFL matches (open data), degraded like our match
(seed 0), fused with WS2's `deadball` and the `occluded` stub, official metric. Mean spellF1, and
on how many matches the variant beats v2 base at the same `nearest` scale:

| variant | nearest 1.0 | nearest 0.5 | wins at 0.5 |
|---|---|---|---|
| v2 base | 0.3564 | 0.3558 | |
| first touch owns | 0.3597 | 0.3573 | 4/7 |
| weight 2 | 0.3578 | 0.3811 | 7/7 |
| weight 3 | 0.3564 | 0.3818 | 6/7 |
| **first touch + weight 2** | 0.3617 | **0.3869** | **7/7** |
| first touch + weight 3 | 0.3602 | 0.3870 | 7/7 |

Weight 3 is no better than 2, so the smaller change stays. On two matches only, `nearest` at 0.5
looked like a gain on its own (20261111_ws3_touch-confirm-v2.md); on seven it is not (0.3564 vs
0.3558). The gain comes from WS3 being louder than `nearest`, which needs both settings.

## Result

All seven DFL matches, the same setup, every WS3 version:

| configuration | spellF1 | macroF1 | team acc | possessions p/gt | above `main` |
|---|---|---|---|---|---|
| `main` today: `nearest` 1.0 + WS2, no WS3 | 0.3186 | 0.7675 | 0.756 | 180/437 | |
| v1, nearest 1.0 | 0.3216 | 0.7739 | 0.766 | 169/437 | 4/7 |
| v2, nearest 1.0 | 0.3564 | 0.7737 | 0.766 | 221/437 | 7/7 |
| v2.1, nearest 1.0 | 0.3617 | 0.7728 | 0.764 | 255/437 | 7/7 |
| **v2.1, nearest 0.5** | **0.3869** | 0.7804 | 0.776 | 236/437 | **7/7** |

On our match (checks only; `main` of 10 November with WS2's second classifier):

| | v2, nearest 1.0 | v2.1, nearest 1.0 | v2.1, nearest 0.5 |
|---|---|---|---|
| possessions | 357 | 406 | 376 |
| mean / median s | 9.2 / 5.5 | 8.1 / 4.5 | 8.7 / 4.9 |
| under 2 s | 55 | 90 | 76 |
| D | 48.3% | 48.3% | 48.3% |
| home share | 60.8% | 60.9% | 60.3% |

`python -m saha.visible.build`: 542 possessions in 134 stretches, 412 hand-overs, 119 unconfirmed
touches, opinion on 97.6% of the tracked graded frames, mean weight 2.0.

**Watch: short possessions.** v2.1 makes more of them on our match (76 under 2 s at `nearest` 0.5,
20% of possessions; v2 had 55). DFL's raw labels change on every touch (29.5% of its possessions
are under 2 s, docs/tech/ws5_external.md), so DFL rewards a louder WS3 partly by construction.
Ours ask for controlled possession. This check does not choose anything (D016), but it is the risk
of v2.1: if our ground truth has fewer short possessions than DFL, part of the DFL gain does not
transfer. WS5's relabelling of DFL to a controlled definition is the test that would settle it.

## Decision

Adopted by the owner on 11 November, with the risk above written down:

- v2.1 replaces v2 in PR #12.
- Proposal for WS1, now on seven matches: `nearest` at 0.5. With v2.1, 0.5 beats 1.0 by 0.025 and
  wins on 7 of 7 matches.
- To revisit when WS5's relabelled DFL exists: `WEIGHT` and `FIRST_TOUCH_OWNS` against a
  controlled definition, and WS1's switching costs with them.

The public 5 minutes do not decide (CLAUDE.md, rule 2).
