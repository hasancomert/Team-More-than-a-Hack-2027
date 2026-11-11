---
date: 2026-11-11
package: ws3
name: touch-confirm-v2
author: Osman (owner, runs on the data), Claude Code (code and record)
commit:
config: configs/pipeline.toml
public_spell_f1: 0.4333
public_spell_low: 0.27
public_spell_high: 0.60
public_macro_f1: 0.8598
n_possessions: 357
mean_duration_s: 9.2
d_rate: 0.483
verdict: open
---

## What changed

`saha/visible/build.py` v2 replaces v1's control runs (20261106_ws3_control-hysteresis-v1.md):

- **Touches instead of control.** A touch is the closest team within 1.5 m of the ball, at any
  ball speed and duels included. Same-team touches at most 3 frames apart are one touch.
- **Confirmed hand-overs.** A touch by the team without the ball is a candidate. It is confirmed
  when, before the team with the ball touches it again, the touch lasts 6 frames or more, or the
  same team touches the ball a second time. Unconfirmed: a deflection, the ball stays.
- **The hand-over is placed at the new team's first touch** after the old team's last one, not at
  the start of a control run.
- Probabilities as before: 0.9 on the owner's touches, 0.8 between two of its touches, 0.6 while
  the ball goes to the other team, on the other team's unconfirmed touches and after a stretch's
  last touch.
- Tests rewritten for v2 (`tests/test_visible.py`, 9, synthetic). `python -m saha.tools.ci`:
  6 of 6 steps pass (91 tests).

## Why

Measured on WS5's DFL match J03WPY (open data, ball always tracked, labels used freely), with
aggregates only on our match:

- v1's control condition removed most of the labelled changes. Of DFL's 324 labelled H<->A
  switches, the new team is within 1.5 m of the ball at the switch in 100% (median 0.43 m), with a
  ball slower than 5 m/s in 73%, without an opponent within 1 m more in 51%, and for 6 frames in a
  row in only 23%. Changes of possession happen in duels and at a first touch.
- v1 placed the change late. Against the labelled switch, the start of the new team's control
  run is a median 19 frames late (21% within 8 frames). The new team's first touch after the old
  team's last one is a median 2 frames late (34% within 8 frames).
- On our match the same pattern shows without labels: in 106 of v1's 160 hand-overs the new team
  touched the ball before its control run, by up to 0.94 s at the 75th percentile and 2.64 s at
  the 90th.

DFL's labels change on every touch, while CHALLENGE.md asks for controlled possession, so DFL is
an upper bound on what counts as a change. The confirmation rule is what keeps a deflection or a
lost duel from being one.

## Result

Timing on DFL J03WPY (clean), v2's hand-overs against the labelled switches:

| | v1 | v2 |
|---|---|---|
| our hand-overs | 161 | 464 |
| ours within 8 / 25 frames of a labelled switch | 21% / 43% | 32% / 49% |
| labelled switches with one of ours within 25 frames | 27% | 60% |

spellF1 on DFL J03WPY, degraded like our match by WS5's harness (seed 0), fused with WS2's
`deadball` (on `main`) and the `occluded` stub, 200-draw bootstrap:

| visible | nearest scale | spellF1 | interval | macroF1 | team acc | possessions p/gt |
|---|---|---|---|---|---|---|
| v1 | 1.0 (today's config) | 0.3505 | 0.306–0.392 | 0.7882 | 0.801 | 177/421 |
| v1 | 0.5 | 0.3384 | 0.297–0.379 | 0.7893 | 0.804 | 157/421 |
| v1 | 0.0 | 0.2901 | 0.242–0.332 | 0.7731 | 0.781 | 147/421 |
| **v2** | 1.0 | 0.3698 | 0.330–0.406 | 0.7856 | 0.796 | 229/421 |
| **v2** | 0.5 | 0.3828 | 0.337–0.421 | 0.7892 | 0.803 | 206/421 |
| **v2** | **0.0** | **0.3987** | 0.358–0.437 | 0.7958 | 0.813 | 193/421 |

Variants of v2 on the same match, for the record (not chosen from: the differences are within
the noise of one match):

| v2 variant | nearest 1.0 | 0.5 | 0.0 |
|---|---|---|---|
| base (1.5 m, 6 frames, 2 touches) | 0.3698 | 0.3828 | 0.3987 |
| 3 touches to confirm | 0.3813 | 0.3845 | 0.3954 |
| 10 frames to confirm | 0.3788 | 0.3866 | 0.4039 |
| touch within 1.0 m | 0.3930 | 0.3960 | 0.4073 |
| 3 touches and 10 frames | 0.3800 | 0.3783 | 0.3854 |

- v2 is above v1 at every scale of `nearest`, by 0.02–0.11.
- With v2, `nearest` helps less the more of it there is: v2 and `nearest` read the same nearest
  player, and the fusion counts it twice. At scale 0 v2 is 0.048 above today's configuration.

Second DFL match, J03WQQ, same setup:

| visible | nearest 1.0 | 0.5 | 0.0 |
|---|---|---|---|
| v1 | 0.3318 | 0.3212 | 0.2893 |
| v2 base | 0.3494 | 0.3502 | 0.3388 |
| v2, 3 touches | 0.3445 | 0.3485 | 0.3394 |
| v2, 10 frames | 0.3446 | 0.3473 | 0.3467 |
| v2, 1.0 m | 0.3491 | 0.3464 | 0.3256 |
| v2, 3 touches and 10 frames | 0.3460 | 0.3497 | 0.3444 |

Mean of the two matches:

| visible | nearest 1.0 | 0.5 | 0.0 |
|---|---|---|---|
| v1 | 0.3412 | 0.3298 | 0.2897 |
| **v2 base** | 0.3596 | **0.3665** | **0.3688** |

- v2 beats v1 at every scale on both matches: +0.018 to +0.050 on J03WQQ.
- The best scale of `nearest` does not agree between the matches (0 on J03WPY, 0.5 on J03WQQ).
  Their two-match means differ by 0.002, far inside the noise; 0.5 varies less between matches.
- The variant that looked best on J03WPY (touch within 1.0 m) is among the worst on J03WQQ. This
  is why no variant was picked from one match.

On our match (checks only), `main` of 10 November with WS2's second classifier:

| | `main` (WS1's report 2) | + v2, nearest 1.0 | + v2, nearest 0.0 |
|---|---|---|---|
| possessions | 293 | 357 | 267 |
| mean / median s | 11.2 / 6.7 | 9.2 / 5.5 | 12.3 / 8.2 |
| under 2 s | 15 | 55 | 23 |
| D | 48.3% | 48.3% | 48.3% |
| home share | | 60.8% | 62.0% |

`python -m saha.visible.build`: 1,594 touches, 519 confirmed possessions in 134 stretches,
392 hand-overs, 117 unconfirmed touches, opinion on 96.8% of the tracked graded frames.

## Decision

Open:

- Adopt v2 in place of v1 (this PR): above v1 at every scale on two DFL matches.
- Proposal for WS1: lower `nearest` in `configs/pipeline.toml` to 0.5. Scales 0.5 and 0 tie on
  the two-match mean (0.3665 and 0.3688); 0.5 is the smaller change and varies less between
  matches. TEAM.md: "once WS3 beats it on WS5's validation data, WS1 lowers or zeroes its scale".
- Keep v2's base values: no variant is better on both matches.
- Everything here uses DFL's raw labels; WS5's relabelling may change the ranking of the variants.

The public 5 minutes do not decide (CLAUDE.md, rule 2).
