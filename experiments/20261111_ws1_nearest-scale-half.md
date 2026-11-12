---
date: 2026-11-11
package: ws1
name: nearest-scale-half
author: Hasan (WS1), Claude Code (runs and record)
commit: 25a5fa4
config: configs/pipeline.toml
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

Nothing in the submission. This record tests WS3's proposal to lower `nearest` in
`configs/pipeline.toml` from 1.0 to 0.5 (experiments/20261111_ws3_touch-confirm-v2.md), and
keeps 1.0. No run on our match was needed: the configuration does not change.

## Why

- **WS3's argument.** WS3 v2 and `nearest` read the same nearest player, so the fusion counts it
  twice.
- **WS3's evidence.** On WS3's two DFL matches the lower scales scored higher. The two-match means
  were 0.3596 at 1.0, 0.3665 at 0.5 and 0.3688 at 0.0.
- **What was still open.** The proposal came from those two matches, so it had to be tested on
  matches it did not come from. TEAM.md says WS1 lowers or zeroes `nearest` "once WS3 beats it
  on WS5's validation data".

**The rule, fixed before any run (16:32 UTC).**

- Adopt 0.5 if both of these hold on the five DFL matches that WS3 did not use:
    - (a) the mean spellF1 difference, 0.5 minus 1.0, is positive;
    - (b) 0.5 is strictly ahead on at least four of the five.
- Otherwise keep 1.0.
- J03WPY and J03WQQ, WS3's matches, only check that the setup reproduces WS3's numbers.
- Scale 0.0 is reported for context and is not adopted in this round: when the evidence does not
  separate two candidates, the smaller change wins.

## Result

**Setup.**

- Code: `main` at 25a5fa4, with PR #12 merged. The sources are `nearest`, `deadball`, `visible` v2
  and the `occluded` stub. The combine parameters are those of `configs/pipeline.toml`.
- Harness: WS5's functions, the same path as `python -m saha.external.score`:
    - `dfl.prepare`;
    - `degrade.degrade`, with the default degradation and seed 0;
    - `harness.build_source`, `harness.decode` and `harness.score`.
- Each source is built once per match; only the `nearest` scale changes.
- Raw DFL labels, 200-draw bootstrap.
- To reproduce, run `python -m saha.external.score --matches <id> --bootstrap 200 --config <file>`,
  where `<file>` is a copy of `configs/pipeline.toml` with the `nearest` scale changed.

**The setup check passes.** J03WPY and J03WQQ reproduce WS3's numbers to four decimals at all
three scales. The intervals and the possession counts match too.

spellF1 per match:

| match | role | 1.0 (today) | 0.5 | 0.0 | 0.5 − 1.0 | possessions at 1.0 / 0.5 / 0.0 (truth) |
|---|---|---|---|---|---|---|
| J03WPY | WS3's | 0.3698 | 0.3828 | 0.3987 | +0.0129 | 229 / 206 / 193 (421) |
| J03WQQ | WS3's | 0.3494 | 0.3502 | 0.3388 | +0.0007 | 208 / 201 / 181 (401) |
| J03WMX | held out | 0.3191 | 0.3186 | 0.3079 | −0.0005 | 211 / 192 / 175 (511) |
| J03WN1 | held out | 0.3320 | 0.3256 | 0.3343 | −0.0064 | 241 / 215 / 206 (500) |
| J03WOH | held out | 0.3561 | 0.3616 | 0.3478 | +0.0055 | 208 / 202 / 180 (402) |
| J03WOY | held out | 0.3576 | 0.3464 | 0.3494 | −0.0112 | 233 / 208 / 200 (456) |
| J03WR9 | held out | 0.4109 | 0.4057 | 0.4007 | −0.0052 | 217 / 196 / 175 (370) |
| **mean, five held out** | | **0.3551** | **0.3516** | **0.3480** | **−0.0036** | |
| mean, all seven | | 0.3564 | 0.3558 | 0.3539 | −0.0006 | |

**What the table shows.**

- **The rule fails.** On the five held-out matches, 0.5 is ahead on one (J03WOH), and the mean
  difference is −0.0036. Both conditions fail.
- **Scale 0.0 does no better.** It is ahead on one of the five, with a mean difference of −0.0071.
- **WS3's gain came mostly from one match.** At 0.5, J03WPY gave +0.013; the other six give
  between −0.011 and +0.006.
- **The differences are small.** Each match's bootstrap interval is about ±0.04 (J03WPY at 1.0:
  0.330–0.406). The paired differences are an order of magnitude smaller.
- **Lower scales do what they should.** They give fewer possessions on every match. macroF1 and
  team accuracy rise slightly on most matches (J03WMX: macroF1 0.7788, 0.7798, 0.7875). But
  spellF1 ranks the submission, and it does not rise on new matches.

Each source alone, at scale 1.0 (context for TEAM.md's condition):

| source alone | mean, five held out | mean, all seven |
|---|---|---|
| `nearest` | 0.1995 | 0.2099 |
| `visible` v2 | 0.1922 | 0.1935 |
| `deadball` | 0.1011 | 0.1040 |

`visible` v2 alone does not beat `nearest` alone on WS5's data yet. So TEAM.md's condition for
lowering `nearest` is not met either.

**Caveats.**

- **Raw labels.** Raw DFL labels change possession on every touch, more often than CHALLENGE.md's
  controlled possession. A relabelled set may favour fewer possessions, and hence a lower
  `nearest`. WS5's relabelling is not ready before the freeze.
- **One degradation seed.**
- **The `occluded` stub.** `occluded` was the stub. WS4's bridge (PR #7) writes only on frames
  without the ball, and `nearest` only on frames with it.

## Decision

Rejected, with v2: `nearest` stays at 1.0 with v2. The rule was fixed before the runs, and the
five matches WS3 did not use do not support the change. The public 5 minutes played no part
(CLAUDE.md, rule 2).

**Update, the same evening.**

- **The question was reopened.** WS3 v2.1 made WS3's evidence louder, and that reopened it. With
  v2.1, `nearest` at 0.5 is adopted: experiments/20261111_ws1_nearest-half-with-v21.md.
- **This test also changes.** On labels closer to CHALLENGE.md's controlled possession, 0.5 also
  helps v2 (+0.0115, ahead on 6 of 7). This test used raw labels only.
