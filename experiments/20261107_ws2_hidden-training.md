---
date: 2026-11-07
package: ws2
name: hidden-training
author: Claude Code (WS2), run by Alparslan
commit: 797baf038b34
config: configs/pipeline.toml
public_spell_f1: 0.4824
public_spell_low: 0.306
public_spell_high: 0.648
public_macro_f1: 0.8667
n_possessions: 293
mean_duration_s: 11.2
d_rate: 0.483
verdict: adopted
---

## What changed

The dead-ball classifier is retrained on every SkillCorner match twice: as delivered, and with
the ball hidden by WS5's own `saha.external.degrade.hide_mask` (its default settings, seed 0).
`saha/deadball/train.py` builds both copies; `skillcorner.py` keeps the half boundaries the
hiding needs. New model `saha/deadball/models/dead_skillcorner.pkl`, SHA256 `e25874c8…`
(878,894 frames). Decoder unchanged (weight 32, tau 204).

## Why

The DFL check (experiments/20261107_ws2_dfl-check.md) found the classifier calling 92% of live
frames dead under WS5's hidden-ball stretches. On SkillCorner the ball is missing on 5.7% of live
frames and 55% of dead ones, so the classifier learned that a missing ball means a stoppage. Our
match misses the ball on 41% of graded frames, against SkillCorner's 20%.

## Result

SkillCorner, five folds of four matches held out (external/hidden_training.py on branch
`ws2/event-deadball`), decoder as shipped, spellF1 on the prototype's team side:

| trained on | as delivered | WS5's hiding applied | live called dead, hidden |
|---|---|---|---|
| matches as delivered | 0.6262 | 0.4250 | 93.3% |
| hidden copies only | 0.6197 (−0.0065 [−0.0102, −0.0027]) | 0.5851 (+0.1601, 20/20) | 13.4% |
| **both** | **0.6236 (−0.0026 [−0.0059, +0.0005])** | **0.5828 (+0.1578 [+0.1455, +0.1712], 20/20)** | **15.1%** |

DFL, the shipped file (python -m saha.deadball.external_check, python -m saha.external.score):

| | dead-ball F1 | stoppage F1 | dead, ours (DFL 44.9%) | fused spellF1 |
|---|---|---|---|---|
| trained as delivered | 0.790 | 0.570 | 68.3% | 0.311 |
| **trained on both** | **0.917** | **0.783** | **49.2%** | **0.319** |

The rule was fixed before the run: adopt if the held-out gain under WS5's hiding has its 95%
interval above zero, the matches as delivered lose no more than 0.005 spellF1, and DFL's
dead-ball F1 rises. "Both" passes all three. "Hidden only" fails the second.

The experiment's models were trained under numpy 2.5.4, the shipped file under the team's pins
(numpy 2.3.5). One percentile of the players' speeds differs by one float32 step between the
two, so the trees differ. The shipped file's DFL numbers above are its own; they match the
experiment's model to the third decimal.

Our match, checks only: D 50.4% → 48.3%, possessions 305 → 293, public spellF1 0.4848 → 0.4824
(intervals overlap: noise). `pipeline --fresh` gives the same `SAHA.csv`, SHA256 `3b74ae89…`.

## Decision

Adopted: it passed the rule fixed in advance. The submission's SHA256 changes from `081f57c1…`
to `3b74ae89…`, so WS1's reproduction check compares against the new value.
