# Final choice

**Frozen on 12 November 2026, at the feature freeze (TEAM.md).** Three improvements were added
after the freeze by the team leader's decisions D026, D027 and D028, each with a rule fixed before
its runs, and the result was reproduced on two machines on 14 November. The frozen version stays
recorded below as the fallback.

Nothing in this file may be justified by the public 5 minutes (CLAUDE.md, rule 2). Each value
comes from WS5's external data, from the model's definition, or from football and physics
reasoning, and says which.

## The submission

| | |
|---|---|
| commit | `d276615`, in `main` since PR #21 (merge `860ae46`) |
| command | `python -m saha.pipeline --fresh --data <path>` |
| `submissions/SAHA.csv` SHA256 | `af770330dbc3e5c414f8618192dcc683e79ff4092692952957ba59b6fbd0d426` |
| reproduced by | Hasan (WS1), Windows |
| reproduced by | Baki (WS4), Python 3.11.9, scikit-learn 1.9.1, 14 November |

Later commits change only documents (this file and the records), so they give the same file.

The two machines agree on more than the SHA256:

- **The seven evidence files hash the same (first 12 characters):** `nearest` 3964488b5def,
  `deadball` 5296d534a6dd, `hiddendead` f87e5b410bce, `visible` bc8aea32bb54, `occluded`
  6dfc61d85e88, `shape` 9c5845d42312, `onball` ab4bdea06f84. The first four are those of the
  frozen version.
- **The match-wide statistics are the same** ("Checks on our match" below).

## The fallback: the frozen version

| | |
|---|---|
| commit | `225cae0`, `main` after PR #20 |
| `submissions/SAHA.csv` SHA256 | `a3348aebc2986463f24e1763cc6ff49f35f8bea7b33268ca92e2fb16d7ea8afe` |
| reproduced by | Hasan (WS1), Windows, 12 November; Baki (WS4), Windows, Python 3.11.9, 13 November |

It would have shipped had PR #21 not been merged by 14 November, 18:00 Turkey time (D026–D028).

## Configuration (`configs/pipeline.toml` at `d276615`)

| setting | value | source |
|---|---|---|
| `eps` | 0.01 | numerical floor: probabilities are clipped to [0.01, 0.99] before the logit (`saha/combine/fuse.py`) |
| `prior_dead` | −0.693 | the model's definition: P(dead) = 1/3, so H, A and D tie without evidence (experiments/20261105_ws1_exact-emissions.md) |
| `cost_switch` | 35 nats | reasoned placeholder from Phase 0, never tuned: tuning needs WS5's relabelled DFL set, which is not ready (D017) |
| `cost_to_dead` | 35 nats | as `cost_switch` |
| `cost_from_dead` | 35 nats | as `cost_switch` |
| scale of `nearest` | 0.5 | WS5's seven DFL matches, with a rule fixed before the runs; it held on raw and on controlled labels (experiments/20261111_ws1_nearest-half-with-v21.md) |
| scale of `deadball` | 1.0 | WS2's decision: `p_dead` 0.99 / 0.01 at weight 20, final everywhere except where `hiddendead` overrides it (D022, D028) |
| scale of `hiddendead` | 2.0, `overrides = "deadball"` | as the other WS1 models; on its frames WS2's `p_dead` is dropped (D028; experiments/20261113_ws1_hiddendead.md) |
| scale of `visible` | 1.0 | WS3 v2.1 writes weight 2 itself, chosen on WS5's seven DFL matches (experiments/20261111_ws3_first-touch-weight-v21.md) |
| scale of `occluded` | 1.0 | WS4 writes weight 1, a reasoned placeholder; checked on WS5's seven DFL matches (experiments/20261112_ws4_bridge-fixed-tags-v2.md) |
| scale of `shape` | 2.0 | as strong as WS3's weight 2, fixed before the runs (D026; experiments/20261113_ws1_shape-model.md) |
| scale of `onball` | 2.0 | as WS3 and `shape`, fixed before the runs (D027; experiments/20261113_ws1_onball-model.md) |

## Evidence packages

| package | commit | what it gives | external validation | experiment record |
|---|---|---|---|---|
| WS1 `nearest` | `13ec8fd` | `p_home` where the ball is tracked; `p_dead`, which WS2's decision overrides | alone: 0.2099 mean spellF1 on WS5's seven DFL matches | experiments/20261111_ws1_nearest-half-with-v21.md (scale) |
| WS2 `deadball` | `e47341c` | `p_dead` on every graded frame: event-gated decoder, classifier trained on 20 SkillCorner matches (MIT) | held-out SkillCorner: spellF1 0.624 and dead-ball F1 0.899 as delivered; 0.583 and 0.835 under WS5's hiding | experiments/20261107_ws2_hidden-training.md |
| WS1 `hiddendead` | `d276615` | a second opinion on WS2's dead calls where the ball is missing, inside play: gradient boosting trained on DFL (CC BY 4.0) | WS5's seven DFL matches, held out, below | experiments/20261113_ws1_hiddendead.md |
| WS3 `visible` | `5e105e7` | `p_home` where the ball is tracked: touches confirmed by the team that takes the ball, weight 2 | WS5's seven DFL matches, below | experiments/20261111_ws3_first-touch-weight-v21.md |
| WS4 `occluded` | `a199558` | `p_home` where the ball is missing: a bridge over each gap, and Smart Tagging phases with sides fixed by their official definitions | WS5's seven DFL matches, below | experiments/20261112_ws4_bridge-fixed-tags-v2.md |
| WS1 `shape` | `757ba6b` | `p_home` where the ball is missing, from the teams' shape: logistic model trained on DFL (CC BY 4.0) | WS5's seven DFL matches, held out, below | experiments/20261113_ws1_shape-model.md |
| WS1 `onball` | `3477ae3` | `p_home` where the ball is tracked, from the ball and the players around it: gradient boosting trained on DFL (CC BY 4.0) | WS5's seven DFL matches, held out, below | experiments/20261113_ws1_onball-model.md |

The whistle stays off (D025). The tracking's ball speed channel is not used (D015). The three WS1
models are files with their manifests in the repository (D020; `shape` a 3 KB JSON).

**Rebuilt from the committed code** (14 November, on WS5's DFL data):

- `onball` and `hiddendead`, which train on one thread, came out byte for byte the same.
- `shape` came out byte for byte the same with the default thread count of the machine that
  trained it (4). On one thread its coefficients differ slightly, because the linear algebra
  sums in another order (cosine 0.9997). That moves P(home) by 0.001 on average and by at most
  0.022 on DFL's frames, and changes 0.13% of its decisions.
- The pipeline only reads the committed files, so the submission does not depend on any of
  this.

## Why this version

**Setup.** WS5's harness on its seven DFL matches (open data, CC BY 4.0):

- each match degraded like ours: the ball is hidden on about 41% of scored frames;
- the official scorer, unmodified;
- mean spellF1, as point estimates. Each match's bootstrap interval is about ±0.04, so the paired
  per-match counts are the stronger evidence;
- each WS1 model fitted on the six other matches every time (leave one match out).

**The labels.** The "controlled-2s" labels are WS1's proxy for CHALLENGE.md's controlled
possession: every possession shorter than 2 s between two possessions of the other team is
merged into them. Raw DFL labels change possession on every touch. WS5's relabelling is not ready.

| version (seed 0) | raw labels | controlled-2s | against the row above |
|---|---|---|---|
| `nearest` and WS2, no WS3, no WS4 | 0.3186 | | (WS3's run) |
| `main` of 11 November: `nearest` 1.0, WS2, WS3 v2, WS4 stub | 0.3564 | 0.3990 | ahead on 7 of 7, raw (WS3's run) |
| + WS3 v2.1 and `nearest` 0.5 | 0.3869 | 0.4204 | ahead on 7 of 7, both label sets |
| + WS4 v2: the frozen version | 0.3937 | 0.4273 | ahead on 6 of 7 raw, 5 of 7 controlled-2s |
| + `shape` (D026) | 0.4913 | 0.5310 | ahead on 7 of 7, both label sets |
| + `onball` (D027) | 0.5308 | 0.5873 | ahead on 7 of 7, both label sets |
| **+ `hiddendead` (D028): the submission** | **0.5476** | **0.6094** | ahead on 7 of 7, both label sets |

Over the three degradation seeds (0, 1, 2), the last four rows score 0.398, 0.490, 0.531 and 0.549
on raw labels, and 0.433, 0.533, 0.587 and 0.609 on controlled-2s; each WS1 step is ahead on 21 of
21 runs on both label sets. Every step was chosen on these external matches, with its rule fixed
before the run, and is written up in its experiment record. Two more models, for the team where
the ball is missing, failed their rules (experiments/20261113_ws1_learned-bridge.md,
experiments/20261113_ws1_long-window-shape.md).

## Checks on our match (they chose nothing)

These come from the runs at `d276615`, aggregate output only. They are identical on both machines.

- **Possessions:** 361, with a mean of 9.7 s and a median of 5.1 s; 67 are shorter than 2 s.
- **D rate:** 44.8% of graded frames (48.3% in the frozen version).
- **Home share:** 59.7%.
- **Evidence coverage of graded frames:**
    - `nearest`: `p_home` on 59.3%;
    - `deadball`: `p_dead` on 100%, at weight 20; WS2 calls 76,664 graded frames dead;
    - `hiddendead`: a second opinion on 55,300 of those (ball missing, inside play), 34.8% of
      graded frames; it finds 9.7% of them rather alive (on DFL, about a quarter);
    - `visible`: `p_home` on 57.9%, at weight 2;
    - `occluded`: `p_home` on 37.9%, at weight 1;
    - `shape`: `p_home` on 39.2% (96.3% of the frames without the ball); half margins 35.2 m and
      30.0 m, no warning;
    - `onball`: `p_home` on 58.1% (97.9% of the frames with the ball).
- **The public 5 minutes, a pipeline check:** spellF1 0.6588 [0.49–0.81], macroF1 0.8950, and 17
  predicted possessions against 17 true ones. The frozen version gave 0.4923 [0.36–0.63] and
  macroF1 0.8742. The interval is too wide to say more, and the public window chose nothing.

## Rules

- [x] No hand labels, and nothing tied to specific frames apart from the published half
      boundaries (CLAUDE.md, rule 1).
- [x] Training and tuning on external data only: the three WS1 models are trained on WS5's DFL
      matches, and nothing is fitted on our match. The public 5 minutes checked the pipeline and
      nothing else (rule 2, D016, D017, D024).
- [ ] The match data and anything frame-level derived from it are absent from the repository
      and its history (`python -m saha.tools.guard --history`).
    - `main` and this branch pass: the guard checks every pushed commit.
    - `--history` also scans the unmerged branches. It refuses `possession/models/dead_skillcorner.pkl`
      (3.5 MB) on WS2's prototype branch `ws2/event-deadball`. That file is a model trained on
      SkillCorner, not match data, but it lies outside D020's path.
    - **Before the repository goes public:** delete that branch, then tick this box once
      `--history` passes.
- [x] AI assistants saw aggregate output only (D008, D017).
- [x] Every external dataset is listed with its licence in the technical document
      (`docs/tech/saha_main.md`, "Training data"): DFL/IDSSE (CC BY 4.0), SkillCorner open data
      (MIT) and the TU Dortmund whistle recordings (CC BY-SA 4.0).

## Changes after the freeze

| date | commit | what | decision | new SHA256 | reproduced by |
|---|---|---|---|---|---|
| 13 Nov | `757ba6b` | `shape`: the team where the ball is missing | D026 | (not shipped alone) | |
| 13 Nov | `3477ae3` | `onball`: the team where the ball is tracked | D027 | (not shipped alone) | |
| 13 Nov | `d276615` | `hiddendead`: second opinion on WS2's dead calls where the ball is missing | D028 | `af770330…` | Hasan, Baki |

No bug fix entered after the freeze.

## Sign-off

Team leader: Hasan Cömert (WS1), 14 November 2026. I approve `d276615` as the submission
(SHA256 `af770330dbc3e5c414f8618192dcc683e79ff4092692952957ba59b6fbd0d426`), reproduced on
Hasan's and Baki's machines, with PR #21 merged into `main`.
