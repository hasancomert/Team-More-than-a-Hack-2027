---
date: 2026-11-04
package: ws1
name: cost-switch-sweep-2
author: Hasan (run), Claude Code (record)
commit: 841947698176
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

Nothing in the submission changed. The self-check now finds its touches from the ball's
positions (PR #5). It ran the same sweep as
[sweep 1](20261104_ws1_cost-switch-sweep-1.md):

- on `main` at 8419476, with the nearest-player evidence only;
- sweeping `cost_switch` over 5, 10, 20, 35 and 60 nats.

## Why

Sweep 1 found only 109 touches, too few to read. It took the ball's speed from the ATD's ball
speed channel, and that channel turned out to be mostly empty (below).

## Result

The ball, from `python -m saha.tools.data_checks --speed`, on the 94,214 graded frames where the
ball is tracked:

- **The positions are clean.**
    - The median step between frames is 0.20 m.
    - 0.9% of steps are over 1 m, and 0.1% are exactly 0.
    - The speed from positions is almost the same over a 5-frame window and over a 25-frame
      window: a median of 5.1 m/s with both, a 90th percentile of 14.3 and 13.3 m/s, and a 99th
      percentile of 24.4 and 22.1 m/s. Jitter does not inflate it.
- **The ATD's ball speed channel is mostly empty.**
    - It has a value on only 7.8% of these frames.
    - Where it has one, the value is 0.29 to 0.30 times the speed from positions.
    - Its 99th percentile is 5.9, against 24.4 m/s for the speed from positions.
    - Sweep 1's touches could only come from that 7.8%. D015 stands.
- **The nearest player is close.** It is within 1.5 m of the ball on 41.8% of these frames, at
  a median distance of 1.91 m.

Touches, condition by condition:

| step | frames left |
|---|---|
| graded | 158,857 |
| ball tracked | 94,214 |
| a player within 1.5 m | 39,408 |
| ball under 5 m/s | 27,430 |

That leaves 942 touches of at least 6 frames.

| cost_switch | recovered | frames | possessions | mean s | median s | short < 2 s | D |
|---|---|---|---|---|---|---|---|
| 5 | 0.931 | 0.933 | 1484 | 2.9 | 1.8 | 786 | 32.3% |
| 10 | 0.923 | 0.926 | 950 | 4.5 | 2.8 | 342 | 32.3% |
| 20 | 0.902 | 0.915 | 549 | 7.8 | 5.6 | 66 | 32.3% |
| 35 | 0.890 | 0.908 | 368 | 11.7 | 8.2 | 0 | 32.3% |
| 60 | 0.866 | 0.880 | 264 | 16.3 | 10.9 | 0 | 32.3% |

- **The sample is large enough now.** With 942 touches, each recovered share has a standard
  error of about 0.01.
- **The possession counts match sweep 1, as they should.** The evidence did not change, only the
  touches did.
- **The recovered share falls steadily as the cost rises**, from 0.931 at 5 nats to 0.866 at 60.
- **Lowering the cost buys little.** Going from 35 to 20 recovers 0.012 more touches. In
  exchange the decode makes 181 more possessions (+49%), 66 of them shorter than 2 s. Going from
  35 to 60 recovers 0.024 fewer touches, with 104 fewer possessions.
- **The bias found in sweep 1 is still there.** The touches come from the nearest player, so the
  more closely a decode follows the nearest-player evidence, the better it scores. On its own,
  the check cannot choose the cost.

## Decision

Rejected: `cost_switch` stays at 35. At 35 nats, the decoder recovers 89% of the withheld
touches.

The self-check is now a working sanity check. Its uses from here:

- **Choosing the costs:** the costs come from WS5's external data (possession counts and
  durations in labelled matches) and from match-wide consistency.
- **Checking new evidence:** a new evidence version (WS3) should keep or raise the recovered share
  at the same possession count.
