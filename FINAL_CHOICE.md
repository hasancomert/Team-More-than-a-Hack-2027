# Final choice

**Template. WS1 fills it in on 12 November, at the feature freeze (TEAM.md).** After that, only
bug fixes enter `main`, and every one of them is recorded here.

Nothing in this file may be justified by the public 5 minutes (CLAUDE.md, rule 2). Each value
comes from WS5's external data, from the model's definition, or from football and physics
reasoning, and says which.

## The submission

| | |
|---|---|
| commit | `TODO` |
| command | `python -m saha.pipeline --fresh --data <path>` |
| `submissions/SAHA.csv` SHA256 | `TODO` |
| reproduced by | TODO (member, operating system, date) |
| reproduced by | TODO (member, operating system, date) |

## Configuration (`configs/pipeline.toml` at that commit)

| setting | value | source |
|---|---|---|
| `eps` | TODO | |
| `prior_dead` | TODO | the model's definition: P(dead) = 1/3, so H, A and D tie without evidence (experiments/20261105_ws1_exact-emissions.md) |
| `cost_switch` | TODO | |
| `cost_to_dead` | TODO | |
| `cost_from_dead` | TODO | |
| scale of `nearest` | TODO | |
| scale of `deadball` | TODO | |
| scale of `visible` | TODO | |
| scale of `occluded` | TODO | |

## Evidence packages

| package | commit | what it gives | external validation | experiment record |
|---|---|---|---|---|
| WS1 `nearest` | TODO | `p_home` where the ball is tracked | | |
| WS2 `deadball` | TODO | `p_dead` on every graded frame | | |
| WS3 `visible` | TODO | `p_home` where the ball is tracked | | |
| WS4 `occluded` | TODO | `p_home` where the ball is missing | | |

## Why this version

TODO: the external comparison that chose it, with WS5's numbers and their intervals.

## Checks on our match (they chose nothing)

TODO: possession count, mean and median duration, possessions under 2 s, D rate, home share,
and the public 5 minutes as a pipeline check.

## Rules

- [ ] No hand labels, and nothing tied to specific frames apart from the published half
      boundaries (CLAUDE.md, rule 1).
- [ ] Training and tuning on external data only; the public 5 minutes checked the pipeline and
      nothing else (rule 2, D016, D017).
- [ ] The match data and anything frame-level derived from it are absent from the repository
      and its history (`python -m saha.tools.guard --history`).
- [ ] AI assistants saw aggregate output only (D008, D017).
- [ ] Every external dataset is listed with its licence in the technical document.

## Fixes after the freeze

| date | commit | what | new SHA256 | reproduced by |
|---|---|---|---|---|

## Sign-off

Team leader: TODO (name, date)
