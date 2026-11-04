---
date: 2026-11-04
package: ws1
name: cost-switch-sweep-1
author: Hasan (run), Claude Code (record)
commit: 8d7ca72d37d5
config: configs/pipeline.toml
public_spell_f1:
public_spell_low:
public_spell_high:
public_macro_f1:
n_possessions: 368
mean_duration_s: 11.7
d_rate: 0.323
verdict: rejected
---

## What changed

Nothing in the submission changed. This record is the first run of
`python -m saha.tools.selfcheck` (PR #4) on the real data:

- on `main` at 8d7ca72, with the nearest-player evidence only;
- sweeping `cost_switch` over 5, 10, 20, 35 and 60 nats, where 35 is the placeholder in
  `configs/pipeline.toml`.

## Why

At 35 nats, no possession shorter than 2 s survives. The self-check asks, without any ground
truth, whether another cost recovers the team at the ball better when the evidence for that
stretch is withheld (CLAUDE.md, rule 2).

## Result

The run found 109 touches in the whole match and held them out in two folds.

| cost_switch | recovered | frames | possessions | mean s | median s | short < 2 s | D |
|---|---|---|---|---|---|---|---|
| 5 | 0.761 | 0.843 | 1484 | 2.9 | 1.8 | 786 | 32.3% |
| 10 | 0.752 | 0.825 | 950 | 4.5 | 2.8 | 342 | 32.3% |
| 20 | 0.716 | 0.806 | 549 | 7.8 | 5.6 | 66 | 32.3% |
| 35 | 0.706 | 0.816 | 368 | 11.7 | 8.2 | 0 | 32.3% |
| 60 | 0.661 | 0.826 | 264 | 16.3 | 10.9 | 0 | 32.3% |

- **Too few touches.** With 109 touches, each recovered share has a standard error of about
  0.04. One touch moves it by 0.009.
- **The touch definition was wrong.** It read the ATD's ball speed channel. The same day's
  `data_checks` run showed that this channel does not follow the ball's positions: the median
  ratio is 0.19 (DECISIONS D015).
- **Frame agreement is flat**, between 0.81 and 0.84.
- **The recovered share rises as the cost falls, only by following the evidence.** At 5 and 10
  nats, the match has 950 to 1,484 possessions with a median of 2 to 3 s, and 342 to 786 of
  them are shorter than 2 s. That is the flicker the metric punishes (docs/METRIC_NOTES.md).
- **The check favours low costs by construction.** The touches are measured with the same
  signal that the nearest-player evidence uses, the nearest player. Following that evidence
  closely therefore looks good on touches.
- **The public window was not used.** That is why the public fields above are empty.

## Decision

Rejected: `cost_switch` stays at 35.

Next steps:

- Find touches from the ball's positions (`saha.common.kinematics`), then run the sweep again.
- Choose the costs from WS5's external data and from match-wide consistency. The self-check
  serves only as a sanity check.
- Short possessions need stronger evidence from WS3 more than they need a lower cost.
