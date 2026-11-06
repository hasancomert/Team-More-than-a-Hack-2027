# Status

Updated **5 November 2026** by WS1. This page says where we are and what to do next; WS1 updates
it at every milestone. The plan itself is [TEAM.md](TEAM.md), the rules are
[CLAUDE.md](CLAUDE.md).

## Read first: the organizers' rules on training and tuning (5 November)

The organizers sent the rules on what is and is not allowed, then answered our questions. They
add no detail, but they say that "anything that is not explicitly allowed is not permitted". The
full text and what it changes are in DECISIONS D016 and D017; CLAUDE.md rules 1 and 2 follow
them. In short:

- Train, fit, calibrate and tune on **external data only**. That rules out this match's
  tracking and tags too: no fitting on them, self-supervised included, and no value estimated
  from this match on every run.
- Never check our output against the video. Watching the video to understand what counts as
  possession is allowed.
- Automatic checks on this match's data are allowed. They catch errors; they never choose a
  value.
- Never upload the data to AI services (D008 already).

## Where we are: Phase 0 is done

The shared infrastructure is in the repository. It was verified on the real data, on the team
leader's machine: `python -m saha.tools.data_checks` passed **16 of 16** checks.

- **Alignment.** Our data layer reproduces the official baseline frame for frame, with 0
  differing frames in all four smoothing windows. The frame alignment is right.
- **No shortcut in the data.** The tracking has no ball-state or ball-ownership channel. It has
  positions and speeds only, for 46 track slots (24 home, 22 away) and for the ball.
- **The pipeline runs end to end** on the real data in a few minutes, and writes a valid
  `submissions/SAHA.csv`. With only the nearest-player evidence and placeholder parameters it
  reports 368 possessions (mean 11.7 s), 32% dead ball and a 57.5% home share. That is a
  pipeline check, not a result.
- **[docs/EDA.md](docs/EDA.md) has the main facts of the data.** Read it before you start. Two
  facts change the plan:
    - The ball goes missing in long stretches (median 10.7 s), not in short blips.
    - The players' speed channel is in m/s, even though the metadata says km/h (DECISIONS
      D012). The ball's speed channel does not follow the ball: take the ball's speed from
      `saha.common.kinematics` (D015).

## Now: Phase 1 has started

PR #1 was merged on 31 October, so everything above is on `main`. Phase 1 is parallel work, until
11 November. Owners and goals are in TEAM.md.

### Everyone, before writing code

1. Accept the GitHub invitation.
2. Confirm that you registered as **SAHA**, and download the data from your own link.
3. Set up the environment ([TEAM.md, Setup](TEAM.md#setup)). It must be **Python 3.11**, not a
   newer one (D014). `python -m saha.tools.setup_dev` must print only OK lines.
4. Run `python -m saha.pipeline` once. It builds your frame table and tests your installation
   end to end.
5. Read CLAUDE.md, your package in TEAM.md (with the boundaries between packages), docs/EDA.md
   and docs/METRIC_NOTES.md.

### How to work

```bash
git checkout main
git pull
git checkout -b ws2/first-evidence        # your package and a short topic
```

- Edit only your package (`saha/deadball`, `saha/visible`, `saha/occluded`, `saha/external`).
  `build(table)` in your `build.py` receives the frame table and returns three arrays,
  `(p_dead, p_home, weight)`, one value per video frame. The stub shows the pattern.
- Before you push:
    1. `python -m saha.<package>.build`, then `python -m saha.common.evidence`.
    2. `python -m saha.pipeline`
    3. `python -m pytest`
    4. `python -m saha.common.experiments new <ws> <short-name>`, and fill the record in.
- Push your branch, open a pull request into `main`, and ask another member to approve it.
  Changes to `saha/common` also need WS1.

### First deliveries, by 4 November

| package | first working evidence file |
|---|---|
| WS2 `deadball` | `p_dead` from the ball outside the pitch and the Smart Tagging restart windows |
| WS3 `visible` | `p_home` on ball-visible frames: nearest player plus a control condition |
| WS4 `occluded` | `p_home` on ball-missing frames: possession carried from the last and next sightings, plus the Smart Tagging phase team |
| WS5 `external` | one open dataset, labels mapped to H/A/D, scored with the official metric |
| WS1 | joint score report on 5 November |

## Milestones

| date | what | state |
|---|---|---|
| 28 – 31 Oct | Phase 0: shared infrastructure | done; PR #1 merged on 31 Oct |
| 31 Oct – 11 Nov | Phase 1: parallel work | started |
| 4 Nov | every package's first working evidence file | |
| 5 Nov | joint score report 1 (WS1) | |
| 9 Nov | joint score report 2 (WS1) | |
| 10 Nov | last day for everyone to be registered as SAHA | |
| 12 Nov | feature freeze: FINAL_CHOICE.md with its commit hash | |
| 13 – 15 Nov | technical document; reproduction on two machines (SHA256); public-repository preparation | reproduction already matched once, on 6 Nov |
| 14 Nov | the form link arrives by email | |
| 16 Nov | submit the form | |
| 17 Nov, 23:59 CET | deadline (Türkiye: 18 Nov, 01:59) | |

## Open items

- **Team leader:**
    - Name the remaining owner: WS3 is still open (TEAM.md).
    - Turn on branch protection for `main`, if the GitHub plan offers it.
- **WS2:** bring the prototype on branch `ws2/event-deadball` into the shared structure. It
  made real progress on 6 November; what is left is the list in the next section.
- **WS1:**
    - Done, on the data (4 November): the players' speed is in m/s (D012). The ball's speed
      channel is not, so the ball's speed now comes from its positions (D015,
      `saha.common.kinematics`).
    - Done: the self-check, with its touches found from the ball's positions
      ([sweep 2](experiments/20261104_ws1_cost-switch-sweep-2.md)).
        - It finds 942 touches, so each recovered share is good to about ±0.01. The first
          sweep had only 109.
        - Lower costs recover a few more touches, but only by inventing many short
          possessions.
        - `cost_switch` stays at 35. At 35 nats, the decoder recovers 89% of the withheld
          touches.
    - Then: let the combiner keep the short possessions that strong evidence supports. Today
      none shorter than 2 s survives.
        - Stronger evidence from WS3 matters more here than a lower cost.
        - The costs are tuned on WS5's external data only (D016).
        - The self-check stays a check: it never chooses a value (D016). Its touches also come
          from the same signal as the nearest-player evidence, so it favours costs that follow
          the evidence closely.
    - Done: exact emissions in the combiner, so that a confident `p_home` no longer pushes
      against D (found by WS4;
      [experiments/20261105_ws1_exact-emissions.md](experiments/20261105_ws1_exact-emissions.md)).
      With the nearest-player evidence alone, the match-wide statistics did not change.
    - Done (6 November): the submission reproduces byte for byte on a second machine. Seyyid's
      clean install gives the same SHA256 as the team leader's (`9fedd7df…`). The check is
      repeated on the final version (13 – 15 November).
    - Joint score report 1 on 5 November.
    - The technical document says how the team used Claude Code: aggregate output only, never
      data (D008, D017).

## WS2: bringing `ws2/event-deadball` into the shared structure

The branch holds a strong event-driven prototype: a dead-ball model, the referee's whistle,
kick-off detection and an event-gated decoder. It was written before Phase 0, as a separate
pipeline. Its ideas are worth keeping, but its shape has to change before anything from it is
merged. WS2 (with Claude Code, if you use it) works through this list.

### Done on 6 November (commits 0dd6df0 and 2e45c19)

- **Python 3.11** with the team's pins, plus scikit-learn 1.9.1 (still in its own requirements
  file; see item 5).
- **The dead-ball classifier is trained on external data:** 20 open SkillCorner matches. Its
  decoder settings are chosen on held-out external matches, and so is the choice to hold a
  stoppage after whistles only. Nothing about the dead-ball model is fitted on this match any
  more (D016, D017).
- **The public window is a sanity check only.** The earlier comparisons on the public window
  and on restart times read off the video no longer decide anything.

### Still to do, in this order

1. **Replace the whistle band.** `possession/audio.py` has `BAND_LOW = (3700, 3960)` and
   `BAND_HIGH = (4180, 4400)` Hz, measured on this match's audio. Its detection threshold was
   also read from this match's score distribution. Both are data made by hand on this match
   and a calibration on the provided video, so the rules forbid them (CLAUDE.md rule 1, D016,
   D017).
    - Take the band and the threshold from published referee-whistle measurements, or from
      external recordings, and cite the source.
    - Never check them against this match.
2. **Take the formation model out of WS2.** It is trained on pseudo-labels from this match's
   tracking. Its correlation time is estimated on this match, and its weight is chosen by a
   test on this match. All of that is fitting on this match (D017).
    - Who has the ball is WS3's and WS4's question anyway (TEAM.md, boundaries).
    - If the model is worth keeping, it moves to WS3 or WS4, trained on WS5's external data.
3. **Start from current `main`.** Open a new branch from `main` (for example
   `ws2/deadball-evidence`) and port the code there. Do not merge `main` into the old branch.
4. **One pipeline.** Read the match through `saha.common` (`frame_table.load()`,
   `smart_tagging.load()`, `kinematics`). Write the result from `saha/deadball/build.py`
   through `write_evidence`:
    - `p_dead` on every frame you can judge;
    - `p_home` only inside restart windows (TEAM.md, boundaries).

   Drop `run.py`, the second decoder and `requirements-solution.txt`. The decoder ideas go into
   WS1's combiner, in a separate PR with WS1: event gating, the whistle hold, and discounting by
   correlation time.
5. **Dependencies.** Add `scikit-learn==1.9.1` to `requirements-saha.txt` in the same PR
   (D014).
6. **The model file.** Commit it as `saha/deadball/models/dead_skillcorner.pkl` (3.3 MB), with
   its manifest `dead_skillcorner.pkl.json` next to it (D020). The manifest holds:
    - `training_script`: the script in the repository that rebuilds the model from the external
      data;
    - `data` and `licence`: the 20 SkillCorner matches and their licence;
    - `seed`, and `libraries` (the Python, scikit-learn and numpy versions);
    - `sha256`: the SHA256 of the model file.

   The guard checks the manifest on every commit and in CI.
7. **One external set, with WS5.** WS2's IDSSE (DFL) and SkillCorner loaders overlap with
   `saha/external` (Seyyid).
    - Use WS5's DFL reader.
    - Move the SkillCorner loader into `saha/external`, with its licence in
      `docs/tech/ws5_external.md`.
    - Score with WS5's harness.

   Then the whole team tunes on one external set, with one protocol. Agree on it with Seyyid.
8. **The whistle needs a team decision** (D006). It requires the tactical video (5.2 GB) and
   ffmpeg on every machine that reproduces the submission. Bring it to the team together with a
   determinism check: the same whistle times on two machines. Until then, everything else must
   work without it.
9. **Move the documentation.** `TECHNICAL.md` becomes `docs/tech/ws2_deadball.md`. Say where
   each number comes from: external data, the laws of the game, or reasoning. WS1 assembles
   `SAHA.md` from the package sections.
