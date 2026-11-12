---
date: 2026-11-12
package: ws4
name: bridge-fixed-tags-v2
author: Baki (owner, runs on the data), Claude Code (code and record)
commit: a19955889380
config: configs/pipeline.toml
public_spell_f1: 0.4778
public_spell_low: 0.311
public_spell_high: 0.633
public_macro_f1: 0.8827
n_possessions: 365
mean_duration_s: 9.0
d_rate: 0.483
verdict: adopted
---

## What changed

WS4 v2, PR #7 (`ws4/first-evidence`), following WS1's review of 5 and 11 November. It replaces
[v1](20261105_ws4_bridge-tags-v1.md), which never ran on the data.

1. **Each tag's meaning comes from its official definition.** `tag_reliability` and `TAG_PRIOR`
   are gone. A fixed table, `TAG_SIDE`, takes each code's side from OVERVIEW.md, where the tag
   Team is "the team performing the event":
    - with the ball, P(tagged team has the ball) = `TAG_P` = 0.7: BUILD UP, PROGRESSION,
      CREATING CHANCES, ATTACKING TRANSITION, BALL IN FINAL THIRD, BALL IN THE BOX, PLAYERS IN
      THE BOX, LONG BALL;
    - without the ball, P = 0.3: DEFENDING IN ATTACKING / MIDDLE / DEFENSIVE THIRD, DEFENSIVE
      TRANSITION;
    - no opinion: SET PIECES, GOALS, team N/A, any other code (counted in the report).

   `TAG_SCALE` stays 0.5, and overlapping tags are still averaged in log-odds. So the tags give
   at most about 0.42 nats per frame.
2. **`build(table)` runs on WS5's external tables.**
    - An external table (`"external"` in `table.meta`) gets no tags, never this match's.
    - `bridge_gaps()` and `opinion()` cut gaps and anchors at the runs of the table's own `half`
      array, not at this match's fixed graded halves. On our match `table.half` is
      `alignment.half_of_frames()`, so the bridge's output is unchanged.
3. **Wording.** The self-check is a check that never chooses a value. The doc says how each
   number is set. The 0.9 clip has a new reason, and the next steps no longer fit anything on
   this match.

No other value changed: `CONTROL_M`, `CONTROL_SPEED`, `ANCHOR_*`, `MEAN_HOLD_S`, `P_MIN`, `P_MAX`
and `WEIGHT` are as in v1. Details: `docs/tech/ws4_occluded.md`.

## Why

- **D017.** v1 measured each code's meaning on this match's touches, on every run. That is a
  value an algorithm estimates from this match, which the organizers' rules forbid. The official
  definitions already say which side each code names. `TAG_P` and `TAG_SCALE` are set by
  reasoning: no external dataset has Smart Tagging, so neither can be tuned (D016).
- **The harness.** `python -m saha.external.score` calls `build(table)` on degraded DFL matches.
  v1 either skipped WS4 there (no match files) or would have put this match's tags on a DFL
  match's frames, and gave no opinion on any gap outside this match's half boundaries.

## Result

### WS5's external harness (DFL, open data, CC BY 4.0)

`python -m saha.external.score --matches J03WPY J03WQQ --bootstrap 200`, then the same with the
occluded scale at 0. The default degradation hides the ball on 41.3% and 41.7% of scored frames.
DFL has no Smart Tagging, so this scores the bridge alone. "evidence occluded ok" on both
matches. The "all" row:

| match | WS4 | spellF1 | interval | macroF1 | team accuracy | possessions pred / true |
|---|---|---|---|---|---|---|
| J03WPY | with | **0.3840** | 0.35–0.42 | 0.7950 | 0.810 | 228 / 421 |
| J03WPY | without | 0.3698 | 0.33–0.41 | 0.7856 | 0.796 | 229 / 421 |
| J03WQQ | with | **0.3597** | 0.32–0.40 | 0.7680 | 0.754 | 214 / 401 |
| J03WQQ | without | 0.3494 | 0.31–0.39 | 0.7638 | 0.747 | 208 / 401 |
| mean | with | **0.3719** | 0.335–0.41 | 0.7815 | 0.782 | 221 / 411 |
| mean | without | 0.3596 | 0.32–0.40 | 0.7747 | 0.772 | 218.5 / 411 |

- With WS4, spellF1 is higher on both matches: +0.014 and +0.010, mean +0.012. macroF1 and team
  accuracy are higher on both too.
- The gain is inside the bootstrap intervals, so it is not significant. But WS4 is nowhere below
  the run without it, which is the adoption condition.
- The mean row's interval is the mean of the two bounds, a guide only.
- The occluded source alone scores 0.013 and 0.020. That is expected: alone it has no opinion
  on the tracked frames (59% of them).

### WS1's seven-match check (12 November)

WS1 reviewed PR #7 and scored WS4 on all seven of WS5's DFL matches, with the occluded scale at
1 and at 0, seed 0. Mean spellF1:

| context | labels | with WS4 | without | difference | WS4 ahead |
|---|---|---|---|---|---|
| main + PR #7 | raw | 0.3649 | 0.3564 | +0.0085 | 6 of 7 |
| main + PR #7 | controlled-1s | 0.3899 | 0.3809 | +0.0090 | 6 of 7 |
| main + PR #7 | controlled-2s | 0.4075 | 0.3990 | +0.0085 | 6 of 7 |
| main + PR #19 + PR #7 | raw | 0.3937 | 0.3869 | +0.0068 | 6 of 7 |
| main + PR #19 + PR #7 | controlled-1s | 0.4169 | 0.4096 | +0.0073 | 6 of 7 |
| main + PR #19 + PR #7 | controlled-2s | 0.4273 | 0.4204 | +0.0069 | 5 of 7 |

- J03WPY and J03WQQ reproduce the numbers above.
- The only clear loss is J03WN1, between −0.001 and −0.004.
- WS4 adds only 10 predicted possessions over the seven matches.
- "controlled" labels merge every possession shorter than 1 s (or 2 s) that sits between two
  possessions of the other team into them. They are WS1's proxy for controlled possession,
  described in WS1's record `20261111_ws1_nearest-half-with-v21` (PR #19).
- Source: WS1's review comment on PR #7.

### On the match (checks only)

`python -m saha.occluded.build`:

- touches on 27,430 of 94,214 tracked graded frames (29.1%);
- 150 missing-ball gaps, length p50 10.7 s, p90 33.8 s, max 221.5 s; both sides with a team in
  57 (same team 37, a hand-over inside 20), neither side in 27;
- 64,643 missing graded frames: an opinion on 93.2%, a tag on 52.1%, |p − 0.5| ≥ 0.2 on 18.7%;
- 571 tag instances used, 67 left out (SET PIECES 59, GOALS 4, N/A 4, unknown 0).

`python -m saha.occluded.selfcheck` (target: the touch team on hidden tracked frames, not the
truth):

| hidden | stretches | bridge: coverage, accuracy | bridge + tags: coverage, accuracy |
|---|---|---|---|
| 2 s | 108 | 100.0%, 0.914 | 100.0%, 0.951 |
| 5 s | 91 | 100.0%, 0.921 | 100.0%, 0.952 |
| 10 s | 74 | 99.0%, 0.792 | 99.0%, 0.810 |
| 20 s | 55 | 95.1%, 0.763 | 95.1%, 0.771 |

- The tags are not inverted: bridge + tags is at or above the bridge alone at every length.
- Calibration: the bands are mostly as right as their mean confidence or more (under-confident).
  The exceptions are 2 s bridge 0.5–0.6 (158 frames, right 0.456), 10 s bridge + tags 0.8–0.9
  (mean 0.843, right 0.823) and 20 s bridge + tags 0.5–0.6 (mean 0.548, right 0.511).

`python -m saha.pipeline`, with and without WS4 (the same config with the occluded scale at 0):

| | without WS4 | with WS4 |
|---|---|---|
| possessions | 357 | 365 |
| mean duration | 9.2 s (median 5.5 s) | 9.0 s (median 5.3 s) |
| shorter than 2 s | 55 | 57 |
| D rate | 48.3% | 48.3% |
| home share | 60.8% | 61.0% |
| public spellF1 (check only) | 0.4333 [0.27–0.60] | 0.4778 [0.31–0.63] |
| public macroF1 (check only) | 0.8598 | 0.8827 |
| SAHA.csv SHA256 | `733f4277…` | `37d64827…` |

- Every evidence file validates (`python -m saha.common.evidence`).
- The two public intervals overlap almost entirely: the public 5 minutes say nothing here.
- The SHA256 with WS4 is
  `37d648272f25cfbcb318e799727296db47d3edf94d5bb38a72f342b30c3a7c05`, from commit `a199558`.

## Decision

Adopted on 12 November. All four conditions below hold, and WS1's seven-match run confirms that
WS4 is not below the run without it: it is ahead on 5 or 6 of 7 matches on every label set. The
public 5 minutes did not decide (CLAUDE.md, rule 2). The conditions:

- the evidence validates and the pipeline runs;
- the self-check is calibrated;
- the match-wide statistics stay plausible;
- on WS5's matches, spellF1 with WS4 is not below the run without it beyond the bootstrap noise.

Where each condition stands on 12 November:

- validates and runs: yes, on our match and on both DFL matches;
- self-check: the tags are not inverted, and the bands are mostly as right as their confidence
  or more, with the three exceptions listed above;
- match-wide statistics: 365 possessions (357 without WS4), mean 9.0 s, D rate unchanged;
- WS5's matches: spellF1 is +0.014 and +0.010 with WS4, inside the noise but not below it. On
  all seven matches (WS1's check), the mean is +0.007 to +0.009.

WS1 reviewed PR #7 and found it ready to merge. No value was changed after any of these runs.
