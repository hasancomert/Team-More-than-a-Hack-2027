---
date: 2026-11-06
package: ws3
name: control-hysteresis-v1
author: Osman (owner, runs on the data), Claude Code (code and record)
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

Synthetic tests only so far. On the match: TO DO (Osman). Run these, then fill in the front matter
from the pipeline's aggregate output:

- `python -m saha.visible.build`: control frames, control runs, hand-overs, coverage;
- `python -m saha.pipeline`;
- `python -m saha.tools.selfcheck`;
- `python -m saha.external.score`, on WS5's DFL matches: with and without `visible`, and with
  `nearest` at scale 1.0 and 0.5 (the two read the same nearest player, so fusing both at full
  scale counts it twice).

## Decision

Open, until the runs on the data and on WS5's matches. Adopt if all of these hold:

- the evidence validates and the pipeline runs;
- on WS5's external data, spellF1 with `visible` is above `nearest` alone, by more than the
  bootstrap noise;
- the match-wide statistics stay plausible: possession count, mean duration, D rate.

The public 5 minutes do not decide (CLAUDE.md, rule 2).
