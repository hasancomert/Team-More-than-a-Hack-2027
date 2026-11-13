# Final choice

**Frozen on 12 November 2026, at the feature freeze (TEAM.md).** From now on only bug fixes enter
`main`, and every one of them is recorded under "Fixes after the freeze".

Nothing in this file may be justified by the public 5 minutes (CLAUDE.md, rule 2). Each value
comes from WS5's external data, from the model's definition, or from football and physics
reasoning, and says which.

## The submission

| | |
|---|---|
| commit | `225cae0`, `main` after PR #20 |
| command | `python -m saha.pipeline --fresh --data <path>` |
| `submissions/SAHA.csv` SHA256 | `a3348aebc2986463f24e1763cc6ff49f35f8bea7b33268ca92e2fb16d7ea8afe` |
| reproduced by | Hasan (WS1), Windows, 12 November |
| reproduced by | Baki (WS4), Windows, Python 3.11.9, 13 November, at `225cae0` |

PR #20 changed only an experiment record and a document. So `eee3162` (after PR #19) and
`225cae0` give the same file.

The two machines agree on more than the SHA256:

- **The four evidence files hash the same (first 12 characters):** `nearest` 3964488b5def,
  `deadball` 5296d534a6dd, `visible` bc8aea32bb54, `occluded` 6dfc61d85e88.
- **The match-wide statistics are the same** (next section).

## Configuration (`configs/pipeline.toml` at that commit)

| setting | value | source |
|---|---|---|
| `eps` | 0.01 | numerical floor: probabilities are clipped to [0.01, 0.99] before the logit (`saha/combine/fuse.py`) |
| `prior_dead` | −0.693 | the model's definition: P(dead) = 1/3, so H, A and D tie without evidence (experiments/20261105_ws1_exact-emissions.md) |
| `cost_switch` | 35 nats | reasoned placeholder from Phase 0, never tuned: tuning needs WS5's relabelled DFL set, which is not ready (D017) |
| `cost_to_dead` | 35 nats | as `cost_switch` |
| `cost_from_dead` | 35 nats | as `cost_switch` |
| scale of `nearest` | 0.5 | WS5's seven DFL matches, with a rule fixed before the runs; it held on raw and on controlled labels (experiments/20261111_ws1_nearest-half-with-v21.md) |
| scale of `deadball` | 1.0 | WS2's decision is final: `p_dead` 0.99 / 0.01 at weight 20, which the combiner reproduces frame for frame (D022) |
| scale of `visible` | 1.0 | WS3 v2.1 writes weight 2 itself, chosen on WS5's seven DFL matches (experiments/20261111_ws3_first-touch-weight-v21.md) |
| scale of `occluded` | 1.0 | WS4 writes weight 1, a reasoned placeholder; checked on WS5's seven DFL matches (experiments/20261112_ws4_bridge-fixed-tags-v2.md) |

With these weights, WS3's evidence counts four times as much as `nearest`'s where the ball is
tracked.

## Evidence packages

| package | commit | what it gives | external validation | experiment record |
|---|---|---|---|---|
| WS1 `nearest` | `13ec8fd` | `p_home` where the ball is tracked; `p_dead`, which WS2's decision overrides | alone: 0.2099 mean spellF1 on WS5's seven DFL matches | experiments/20261111_ws1_nearest-half-with-v21.md (scale) |
| WS2 `deadball` | `e47341c` | `p_dead` on every graded frame: event-gated decoder, classifier trained on 20 SkillCorner matches (MIT) | held-out SkillCorner: spellF1 0.624 and dead-ball F1 0.899 as delivered; 0.583 and 0.835 under WS5's hiding | experiments/20261107_ws2_hidden-training.md |
| WS3 `visible` | `5e105e7` | `p_home` where the ball is tracked: touches confirmed by the team that takes the ball, weight 2 | WS5's seven DFL matches, below | experiments/20261111_ws3_first-touch-weight-v21.md |
| WS4 `occluded` | `a199558` | `p_home` where the ball is missing: a bridge over each gap, and Smart Tagging phases with sides fixed by their official definitions | WS5's seven DFL matches, below | experiments/20261112_ws4_bridge-fixed-tags-v2.md |

The whistle stays off (D025). The tracking's ball speed channel is not used (D015).

## Why this version

**Setup.** WS5's harness on its seven DFL matches (open data, CC BY 4.0):

- each match degraded like ours (seed 0): the ball is hidden on about 41% of scored frames;
- the official scorer, unmodified;
- mean spellF1, as point estimates. Each match's bootstrap interval is about ±0.04, so the paired
  per-match counts are the stronger evidence.

**The labels.** The "controlled-2s" labels are WS1's proxy for CHALLENGE.md's controlled
possession: every possession shorter than 2 s between two possessions of the other team is
merged into them. Raw DFL labels change possession on every touch. WS5's relabelling is not ready.

| version | raw labels | controlled-2s | against the row above |
|---|---|---|---|
| `nearest` and WS2, no WS3, no WS4 | 0.3186 | | (WS3's run) |
| `main` of 11 November: `nearest` 1.0, WS2, WS3 v2, WS4 stub | 0.3564 | 0.3990 | ahead on 7 of 7, raw (WS3's run) |
| + WS3 v2.1 and `nearest` 0.5 | 0.3869 | 0.4204 | ahead on 7 of 7, both label sets |
| **+ WS4 v2: the frozen version** | **0.3937** | **0.4273** | ahead on 6 of 7 raw, 5 of 7 controlled-2s |

Every step was chosen on these external matches, with its rule fixed before the run, and is
written up in its experiment record.

## Checks on our match (they chose nothing)

These come from the frozen version's runs, aggregate output only. They are identical on both
machines.

- **Possessions:** 383, with a mean of 8.6 s and a median of 4.8 s; 78 are shorter than 2 s.
- **D rate:** 48.3% of graded frames.
- **Home share:** 60.5%.
- **Evidence coverage of graded frames:**
    - `nearest`: `p_home` on 59.3%;
    - `deadball`: `p_dead` on 100%, at weight 20;
    - `visible`: `p_home` on 57.9%, at weight 2;
    - `occluded`: `p_home` on 37.9%, at weight 1.
- **The public 5 minutes, a pipeline check:** spellF1 0.4923 [0.36–0.63], macroF1 0.8742, and
  22 predicted possessions against 17 true ones. The interval is too wide to say more.

## Rules

- [x] No hand labels, and nothing tied to specific frames apart from the published half
      boundaries (CLAUDE.md, rule 1).
- [x] Training and tuning on external data only. The public 5 minutes checked the pipeline and
      nothing else (rule 2, D016, D017, D024).
- [ ] The match data and anything frame-level derived from it are absent from the repository
      and its history (`python -m saha.tools.guard --history`).
    - `main` passes: `python -m saha.tools.guard --range <official root>..main` is clean on all 92
      commits since the official repository's root.
    - `--history` also scans the unmerged branches. It refuses `possession/models/dead_skillcorner.pkl`
      (3.5 MB) on WS2's prototype branch `ws2/event-deadball`. That file is a model trained on
      SkillCorner, not match data, but it lies outside D020's path.
    - **Before the repository goes public:** delete that branch, then tick this box once
      `--history` passes.
- [x] AI assistants saw aggregate output only (D008, D017).
- [ ] Every external dataset is listed with its licence in the technical document: DFL/IDSSE
      (CC BY 4.0) and SkillCorner open data (MIT). To do on 13–15 November.

## Fixes after the freeze

| date | commit | what | new SHA256 | reproduced by |
|---|---|---|---|---|

## Sign-off

Team leader: TODO (Hasan, once the second reproduction matches)
