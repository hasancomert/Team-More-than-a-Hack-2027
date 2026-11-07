---
date: 2026-11-07
package: ws2
name: dfl-check
author: Claude Code (WS2), run by Alparslan
commit: 685f21aa1bef
config: configs/pipeline.toml
public_spell_f1: 0.4848
public_spell_low: 0.315
public_spell_high: 0.667
public_macro_f1: 0.7693
n_possessions: 305
mean_duration_s: 10.3
d_rate: 0.504
verdict: open
---

## What changed

No change to the submission. The check WS1 asked for on PR #14: WS2 scored on WS5's seven DFL
matches (25 fps, full pitch), degraded by WS5 (ball hidden on 41% of frames in long stretches
placed at random). `saha/deadball/external_check.py` adds the dead-ball scores alone, and the
same with DFL's frozen ball removed first: positions unchanged to the last bit for 5 frames or
more become missing, with no labels involved.

## Why

The classifier learned from SkillCorner's 10 fps broadcast tracking. DFL is the external data
closest to our recording, so it shows whether the classifier transfers.

## Result

`python -m saha.external.score` (configs/pipeline.toml, seed 0):

| match | all (nearest + deadball) | nearest alone | dead/alive F1, all |
|---|---|---|---|
| J03WMX | 0.2641 | 0.1890 | 0.701 |
| J03WN1 | 0.3012 | 0.2106 | 0.737 |
| J03WOH | 0.3395 | 0.2034 | 0.685 |
| J03WOY | 0.2762 | 0.1715 | 0.735 |
| J03WPY | 0.3304 | 0.2335 | 0.768 |
| J03WQQ | 0.3281 | 0.2385 | 0.726 |
| J03WR9 | 0.3386 | 0.2231 | 0.739 |
| **mean** | **0.3112** | **0.2099** | **0.727** |

`python -m saha.deadball.external_check`, means of the seven matches:

| ball | P | R | F1 | stoppage F1 | dead, DFL | dead, ours | live called dead, ball hidden / in view |
|---|---|---|---|---|---|---|---|
| as delivered | 0.655 | 0.997 | 0.790 | 0.570 | 44.9% | 68.3% | 92.5% / 5.2% |
| frozen ball removed | 0.649 | 0.995 | 0.785 | 0.566 | 44.9% | 68.7% | 92.3% / 6.7% |

- DFL's frozen ball is on 100% of dead frames and under 1% of live ones. Removing it costs
  this package 0.005 dead-ball F1, so it is not what the numbers rest on.
- Almost no stoppage is missed, but a third of live play is called dead. 93% of the wrongly
  dead frames lie under the harness's hidden-ball stretches: the classifier learned on
  SkillCorner that a ball unseen for long means a stoppage.
- DFL's own dead share is 41–52% per match. Ours on our match is 50.4%, inside that range.
  Whether our match's missing ball falls in live play, like WS5's random stretches, or mostly
  in stoppages, decides which picture applies. No allowed check on our match can say.

## Decision

Open, then acted on: the classifier trained under WS5's degradation as well passed the rule
fixed before its run (held-out SkillCorner under the degradation improves with its 95% interval
above zero, as delivered it loses no more than 0.005 spellF1, DFL's dead-ball F1 rises). See
experiments/20261107_ws2_hidden-training.md.
