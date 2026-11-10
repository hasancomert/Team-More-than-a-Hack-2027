---
date: 2026-11-06
package: ws3
name: control-hysteresis-v1
author: Osman (owner, runs on the data), Claude Code (code and record)
commit: e5b33ea
config: configs/pipeline.toml
public_spell_f1: 0.4105
public_spell_low: 0.28
public_spell_high: 0.54
public_macro_f1: 0.6307
n_possessions: 335
mean_duration_s: 12.8
d_rate: 0.323
verdict: open
---

## What changed

WS3's first evidence file replaces the stub in `saha/visible/build.py`:

- `p_home` on graded frames where the ball is tracked, and nowhere else; `p_dead` stays NaN.
- Control: a player within 1.5 m of a ball slower than 5 m/s, with no opponent within 1 m more
  (no duel), for at least 6 frames. The same "at the ball" as the self-check and WS4.
- Hysteresis between control runs: the ball stays with the last team in control until the
  opponent takes control (0.8 between runs of the same team, 0.6 before a hand-over, 0.9 on a
  run). The state is forgotten across ball gaps longer than 0.5 s and at each half.
- Tests: `tests/test_visible.py`, synthetic only.

Details and parameters: `docs/tech/ws3_visible.md`.

## Why

- `nearest` gives the ball to the nearest team on every tracked frame, so a pass that flies past
  an opponent, a deflection or a duel each look like a hand-over. CHALLENGE.md says none of these
  is a change of possession: possession stays with the last team until the opponent controls it.
- Under spellF1 every such flicker that survives the decoder is a false possession
  (docs/METRIC_NOTES.md). WS3's evidence pushes back against them where `nearest` pushes for
  them, and agrees with `nearest` where a team really controls the ball.

## Result

Tests: `tests/test_visible.py` (8, synthetic) and the full suite pass (73).

On the match (Osman's machine, 10 November, branch at e5b33ea, so `main` as of 6 November:
`nearest` and `visible` have opinions, `deadball` and `occluded` are stubs).

`python -m saha.visible.build`:

| | |
|---|---|
| graded frames with the ball tracked | 94,214 |
| control frames | 25,095 |
| stretches | 134 |
| control runs | 881 |
| hand-overs inside a stretch | 160 |
| frames with an opinion | 83,902 (0.891 of tracked frames) |
| home share of the opinions | 52.8% |

`python -m saha.pipeline`, next to WS1's nearest-only run on the same `main`:

| | nearest only | nearest + visible |
|---|---|---|
| possessions | 368 | 335 |
| mean duration | 11.7 s | 12.8 s (median 8.5 s) |
| shorter than 2 s | | 7 |
| D rate | 32.3% | 32.3% |
| home share | 57.5% | 59.1% |
| public spellF1 (check only) | 0.4000 | 0.4105 [0.28–0.54] |

The decoder makes fewer, longer possessions, as intended; the D rate is unchanged because WS3
writes no `p_dead`. The public difference is far inside the bootstrap interval and decides nothing.

`python -m saha.external.score --matches J03WPY --bootstrap 200` (WS5's DFL harness, degraded to
our conditions, seed 0, raw DFL labels; deadball and occluded are stubs):

| sources | spellF1 | interval | macroF1 | alive/dead F1 | team acc | possessions pred / gt |
|---|---|---|---|---|---|---|
| nearest + visible | 0.2284 | 0.20–0.26 | 0.4997 | 0.633 | 0.856 | 241 / 421 |
| nearest alone | 0.2335 | 0.20–0.26 | 0.4946 | 0.633 | 0.844 | 259 / 421 |
| visible alone | 0.1328 | 0.10–0.17 | 0.3618 | 0.716 | 0.780 | 97 / 421 |

- Team accuracy goes up (0.844 → 0.856), spellF1 does not move beyond the noise.
- The comparison is biased against v1: DFL's raw possession changes on every touch (421
  possessions, median 3.8 s, docs/tech/ws5_external.md), and v1 exists to not do that. Until WS5
  relabels DFL to a "controlled possession" definition, these numbers cannot show whether the
  hysteresis helps.

Still to do: the other six DFL matches; `nearest` at scale 0.5; the same runs on relabelled DFL.

## Decision

Open, until the runs on the data and on WS5's matches. Adopt if all of these hold:

- the evidence validates and the pipeline runs;
- on WS5's external data, spellF1 with `visible` is above `nearest` alone, by more than the
  bootstrap noise;
- the match-wide statistics stay plausible: possession count, mean duration, D rate.

The public 5 minutes do not decide (CLAUDE.md, rule 2).
