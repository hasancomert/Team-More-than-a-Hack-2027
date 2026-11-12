---
date: 2026-11-11
package: ws1
name: nearest-half-with-v21
author: Hasan (WS1), Claude Code (runs and record)
commit: c9a6a39
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

- `configs/pipeline.toml`: `nearest` goes from 1.0 to 0.5.
- WS3 v2.1 (`ws3/first-evidence` at c9a6a39) comes into `main` in the same PR. PR #12 was merged
  before v2.1's commits.

The two go in together. On controlled labels, v2.1 with `nearest` at 1.0 is slightly below v2
(see the results).

## Why

- **WS3's claim.** WS3 v2.1 makes WS3's evidence louder than `nearest` (weight 2). WS3 reports
  that, on all seven of WS5's DFL matches, `nearest` at 0.5 then beats 1.0 by 0.025, on 7 of 7
  matches (experiments/20261111_ws3_first-touch-weight-v21.md).
- **Why it needed checking.**
    - Those seven matches also chose v2.1's two settings.
    - Raw DFL labels change possession on every touch, which rewards switching. CHALLENGE.md asks
      for controlled possession.
- **The first test.** WS1's test of the same proposal with v2 rejected it on raw labels
  (experiments/20261111_ws1_nearest-scale-half.md).

**The rule, fixed before any run (21:50 UTC).** Adopt `nearest` 0.5 with v2.1 if both of these
hold, on the raw labels and on the controlled-2s labels:

- v2.1 at 0.5 has a higher mean spellF1 over the seven matches than v2.1 at 1.0;
- v2.1 at 0.5 is ahead on at least five of the seven.

Otherwise keep 1.0.

**The controlled labels.** They are a WS1 proxy for CHALLENGE.md's controlled possession, where a
touch, a deflection or a lost duel is not a change.

- Every H or A run shorter than 1 s (controlled-1s) or 2 s (controlled-2s) whose neighbours on
  both sides are the other team is merged into them: A H A becomes A A A.
- D and X runs never change.
- They are not WS5's relabelling, which is not ready before the freeze.

## Result

**Setup.** The same path as the first test:

- WS5's harness: `dfl.prepare`, then `degrade.degrade` (default degradation, seed 0), then
  `harness.build_source`, `decode` and `score`;
- the seven DFL matches;
- point estimates;
- v2 is `main` at 25a5fa4, and v2.1 is `main` merged with c9a6a39.

**The setup check passes.** On raw labels, the v2.1 means reproduce WS3's numbers exactly (0.3617
and 0.3869).

Mean spellF1 over the seven matches:

| labels | true possessions | v2 at 1.0 (`main` today) | v2.1 at 1.0 | v2.1 at 0.5 | 0.5 − 1.0 (v2.1) | 0.5 ahead |
|---|---|---|---|---|---|---|
| raw | 3,061 | 0.3564 | 0.3617 | **0.3869** | +0.0253 | 7 of 7 |
| controlled-1s | 2,715 | 0.3809 | 0.3800 | **0.4096** | +0.0296 | 7 of 7 |
| controlled-2s | 2,335 | 0.3990 | 0.3910 | **0.4204** | +0.0293 | 7 of 7 |

Per match, on controlled-2s labels:

| match | v2 at 1.0 | v2.1 at 1.0 | v2.1 at 0.5 | 0.5 − 1.0 (v2.1) |
|---|---|---|---|---|
| J03WPY | 0.4058 | 0.3973 | 0.4205 | +0.0232 |
| J03WQQ | 0.3735 | 0.3716 | 0.3897 | +0.0181 |
| J03WMX | 0.4133 | 0.3849 | 0.4143 | +0.0294 |
| J03WN1 | 0.3786 | 0.3742 | 0.4107 | +0.0366 |
| J03WOH | 0.3931 | 0.3943 | 0.4238 | +0.0295 |
| J03WOY | 0.3944 | 0.3725 | 0.4271 | +0.0547 |
| J03WR9 | 0.4343 | 0.4426 | 0.4565 | +0.0138 |

**What the tables show.**

- **The rule holds.** It holds on raw labels and on both controlled sets, on 7 of 7 matches each
  time.
- **v2.1 at 0.5 beats today's `main` on every label set** (7 of 7 each): +0.031 on raw labels,
  +0.029 on controlled-1s and +0.021 on controlled-2s. The gain shrinks as the labels get stricter,
  as WS3 expected, but it stays.
- **v2.1 needs 0.5.** v2.1 at 1.0 is below v2 at 1.0 on both controlled sets: −0.001 on
  controlled-1s and −0.008 on controlled-2s. So v2.1 and `nearest` at 0.5 belong together.
- **The predictions do not overshoot.** v2.1 at 0.5 predicts 1,650 possessions over the seven
  matches. That is fewer than the truth on every definition, including on every match under
  controlled-2s. v2 at 1.0 predicts 1,547, and v2.1 at 1.0 predicts 1,785.
- **The first test also changes.** With v2, 0.5 also helps on controlled-2s labels: +0.0115,
  ahead on 6 of 7. Only the raw labels the first test used made it look like a loss.

**On our match (checks only).** These are WS3's aggregates, from Osman's run:

- 376 possessions, with a mean of 8.7 s and a median of 4.9 s;
- 76 possessions under 2 s;
- D on 48.3% of frames.

Short possessions are the risk WS3 wrote down. The controlled-2s result is the best available
check of that risk before the freeze.

**Caveats.**

- **No untouched set.** The seven matches also chose v2.1's settings, and none is left for a
  clean test. The 7-of-7 result on every label set limits the concern.
- **The controlled labels are WS1's proxy.** They are not WS5's relabelling.
- **One degradation seed.**
- **The `occluded` stub.** `occluded` was the stub. WS4 (PR #7) writes only where the ball is
  missing, and `nearest` only where it is tracked.
- **Where the weights live.** v2.1 writes weight 2 into its own evidence, and the config's
  `visible` scale stays 1.0. With `nearest` at 0.5, WS3's evidence now counts four times as much
  as `nearest`'s. FINAL_CHOICE.md must say so.

## Decision

Adopted: `nearest` at 0.5 in `configs/pipeline.toml`, together with WS3 v2.1 in the same PR. The
rule was fixed before the runs, and it holds on raw labels and on both controlled sets, on 7 of 7
matches each time. The public 5 minutes played no part (CLAUDE.md, rule 2). To revisit on WS5's
relabelled DFL data, if it arrives: v2.1's weight, `nearest`'s scale, and WS1's switching costs.
