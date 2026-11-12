---
date: 2026-11-05
package: ws4
name: bridge-tags-v1
author: Baki (owner, runs on the data), Claude Code (code and record)
commit:
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

WS4's first evidence file replaces the stub in `saha/occluded/build.py`:

- `p_home` on graded frames where the ball is not tracked, and nowhere else;
- the bridge over each gap: the touch team on each side, then a forward–backward two-state chain
  across the gap;
- Smart Tagging phase teams as weak evidence. SET PIECES are excluded, because a restart's team is
  WS2's question. Each code's meaning (team with the ball, or team without it) is measured on the
  touches, with no ground truth.
- New: `python -m saha.occluded.selfcheck`, a label-free accuracy and calibration check.

Details and parameters: `docs/tech/ws4_occluded.md`.

## Why

- 41% of graded frames have no ball. Without WS4, the Viterbi carries the last team through every
  gap and switches only at the next sighting: the worst place for a hand-over that happened
  inside the gap.
- The bridge puts the switch where it is most likely.
- The tags are the only direct statement about possession inside long gaps.

## Result

Synthetic data only so far. Setup:

- the full-length synthetic match;
- 40% of graded frames hidden in 3–30 s stretches;
- a dead-ball stand-in;
- the nearest-player evidence for `p_home`.

| variant | spellF1 | macroF1 | hidden alive frames right |
|---|---|---|---|
| without WS4 | 0.786 | 0.903 | 0.567 |
| bridge only | 0.807 | 0.905 | 0.574 |
| bridge + tags | 0.992 | 0.994 | 0.971 |

The tags' gain is optimistic: the synthetic tags are exactly aligned with the truth.

On the match: TO DO (Baki). Run these, then fill in the front matter from the pipeline's
aggregate output:

- `python -m saha.occluded.build`: gap counts, coverage, and the per-code tag shares;
- `python -m saha.occluded.selfcheck`;
- `python -m saha.pipeline`.

## Decision

Rejected. v1 was replaced before any run on the data, because it measured each Smart Tagging
code's meaning on this match's touches on every run: a value estimated from this match, which
the organizers' rules forbid (DECISIONS D017). The v2 record,
[20261112_ws4_bridge-fixed-tags-v2.md](20261112_ws4_bridge-fixed-tags-v2.md), replaces it: each
code's meaning comes from its official definition, with one reasoned strength.
