---
date: 2026-11-06
package: ws5
name: dfl-first-harness
author: Claude Code (WS5), run by planckestein
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

First WS5 delivery, `saha/external/`:

- `dfl.py`: downloads DFL/Sportec open tracking (IDSSE, CC BY 4.0) to `cache/external/dfl/`,
  streams the XML into arrays, and maps the ball's `BallStatus` / `BallPossession` to H/A/D.
- `degrade.py`: makes it look like our match (ball hidden on 41% of frames in long stretches,
  no ball height or speed channel, fresh slots per half).
- `harness.py`, `score.py`: run every package's `build(table)` on the degraded table, decode
  with `saha.combine`, score with the official metric; `--sweep` over combine parameters.
- `build.py`: the definition check, external label statistics next to the public 5 minutes.

No change to the submission. The public-window fields above are empty because this machine
does not have the match data yet; WS5 changes nothing that the pipeline runs.

## Why

D016/D017: every tuned value must come from external data. Without WS5 nothing can be tuned
(STATUS.md). DFL comes first because it is 25 fps and has ball status and possession on every
frame.

## Result

External match DFL J03WPY, seed 0, `configs/pipeline.toml` at the time (only `nearest` has an
opinion; the other three packages are stubs on `main`):

| run | spellF1 | macroF1 | alive/dead F1 | team acc | possessions pred / gt |
|---|---|---|---|---|---|
| degraded, cost_switch 15 | 0.2300 | 0.4876 | 0.633 | 0.827 | 419 / 421 |
| degraded, cost_switch 35 (current) | 0.2335 | 0.4946 | 0.633 | 0.844 | 259 / 421 |
| degraded, cost_switch 60 | 0.2038 | 0.4945 | 0.633 | 0.845 | 203 / 421 |
| clean (ball always visible), cost_switch 35 | 0.2852 | 0.3941 | 0.716 | 0.848 | 297 / 421 |

The nearest-player baseline scores 0.2335 here, close to the official baseline on our match
(0.2433). That is a plausibility check of the harness, not a target.

Definition check (docs/tech/ws5_external.md): DFL possessions are much shorter than ours (median
3.8 s vs 7.6 s on the public 5 minutes; 29.5% vs 11.8% under 2 s) and its D rate is much higher
(44.3% vs 19.6%).

## Decision

Open. The harness is ready for every package to use. No parameter should be chosen from these
numbers yet:

- only one match is converted;
- DFL's possession definition differs from ours.

Next: convert all seven matches, then propose a reasoned relabelling of DFL possession, and
compare it with the raw labels.
