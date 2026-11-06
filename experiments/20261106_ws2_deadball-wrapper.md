---
date: 2026-11-06
package: ws2
name: deadball-wrapper
author: Claude Code (WS2), run by Alparslan
commit: 35683cb
config: configs/pipeline.toml
public_spell_f1: 0.4848
public_spell_low: 0.315
public_spell_high: 0.667
public_macro_f1: 0.7693
n_possessions: 305
mean_duration_s: 10.3
d_rate: 0.504
verdict: open
---

## What changed

`saha/deadball/` replaces the Phase 0 stub with the event-driven dead-ball detector of branch
`ws2/event-deadball`, ported into the shared structure (STATUS.md, WS2 list):

- reads the match only through `saha.common` (frame table, Smart Tagging kick-off windows) and
  writes `p_dead` through `write_evidence`: WS2's dead/alive decision at 0.99 / 0.01, weight 20
  on graded frames; no `p_home`;
- the dead-ball classifier, trained on SkillCorner open data (MIT) by `saha.deadball.train`, is
  committed as `saha/deadball/models/dead_skillcorner.pkl` with its manifest (D020);
- the team-shape model and every self-supervised part of the prototype are left out, and the
  whistle is off unless `--whistle` is given;
- scikit-learn 1.9.1 and its dependencies are pinned in `requirements-saha.txt`;
- `tests/conftest.py` sets `SAHA_DEADBALL_MODEL=none`, so the pipeline tests run the builder
  with no opinion, as before: on the synthetic match (four players) the classifier calls 96% of
  frames dead, and the tests check that the truth comes back; `tests/test_deadball.py` tests the
  package with a stand-in classifier;
- `docs/tech/ws2_deadball.md` holds the design, assets, training data and external results.

## Why

The organizers' rules (D016, D017): the prototype's dead-ball classifier and team-shape model
were fitted on this match, and its whistle bands were measured on this match's audio. The
classifier is now trained on twenty labelled external matches, and the decoder settings that go
with it (correlation time 204 frames, weight 32) and the hold after whistles only were chosen on
those matches, matches held out. Details and numbers: docs/tech/ws2_deadball.md.

## Result

External, SkillCorner, 20 matches in five held-out folds, official `spell_scores`, the
prototype's team side throughout (so the differences come from the dead ball):

| | spellF1 | dead-ball F1 | stoppage F1 |
|---|---|---|---|
| prototype's first design, without any hold | 0.571 | 0.833 | 0.696 |
| SkillCorner classifier, hold after whistles only, prototype's team evidence | 0.632 | 0.899 | 0.826 |
| **the same without team evidence (this package)** | **0.626** | **0.901** | **0.833** |

Dropping the team evidence costs 0.006 spellF1 [−0.008, −0.003] and is required by the rules.

Our match, checks only, against the previous adopted version (only `nearest` with an opinion):

| | previous | this |
|---|---|---|
| D rate | 32.3% | 50.4% |
| possessions (mean) | 368 (11.7 s) | 305 (10.3 s) |
| public spellF1 [bootstrap] | 0.4000 [0.26–0.54] | 0.4848 [0.32–0.67] |

The public intervals overlap: noise, and not a reason for anything (D016). Other checks:
- `deadball`'s decision and the `D` frames of `SAHA.csv` differ on 0 of 158,857 graded frames.
- With the same classifier file, the port and the prototype (no whistle, no team evidence) differ
  on 0 graded frames.
- The frame table and the prototype's own tracking reader agree on every value.
- `pipeline --fresh` (empty cache) gave the same `SAHA.csv`, SHA256 `081f57c1b588…`.
- Rebuilding the classifier gave the same file all four times, SHA256 `0506d791b9ac…`, twice from
  a freshly converted copy of the data.
- `pytest`: 77 passed.

The DFL harness (WS5) was not run with this package. DFL freezes the ball on dead frames, so a
dead-ball detector that reads the ball would look better there than it is (docs/tech/ws2_deadball.md).

## Decision

Open, for review. To decide as a team: whether the decoder stays in this package or its ideas
move to WS1's combiner (STATUS.md, WS2 item 4), and the whistle (item 8).
