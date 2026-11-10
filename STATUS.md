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

## CI runs locally (10 November)

GitHub no longer starts the CI workflow on this repository: the account's Actions billing
would have to be raised, and the team leader chose not to pay (D023). So a red "checks" with
"The job was not started" on a pull request is expected, and is not a failure of your code.

- Before you open a PR, run `python -m saha.tools.ci`. It runs every check of the workflow
  locally, with synthetic data only.
- WS1 runs it again on your PR merged into `main` before the team leader merges.

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
| 5 Nov | joint score report 1 (WS1) | skipped: no package had delivered |
| 9 Nov | joint score report 2 (WS1) | written on 8 Nov (docs/reports/) |
| 10 Nov | last day for everyone to be registered as SAHA | |
| 12 Nov | feature freeze: FINAL_CHOICE.md with its commit hash | |
| 13 – 15 Nov | technical document; reproduction on two machines (SHA256); public-repository preparation | reproduction already matched once, on 6 Nov |
| 14 Nov | the form link arrives by email | |
| 16 Nov | submit the form | |
| 17 Nov, 23:59 CET | deadline (Türkiye: 18 Nov, 01:59) | |

## Open items

- **Team leader:**
    - Turn on branch protection for `main`, if the GitHub plan offers it.
- **WS2:** PR #14 ports the prototype into `saha/deadball`. What is left is in the WS2 section
  below.
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
    - Joint score report 2:
      [docs/reports/20261109_joint_report_2.md](docs/reports/20261109_joint_report_2.md). Report 1
      was skipped, because no package had delivered by 5 November.
    - Ready for the freeze:
        - the `FINAL_CHOICE.md` template;
        - `docs/tech/saha_main.md` and WS1's own section `docs/tech/ws1_pipeline.md`;
        - `python -m saha.tools.techdoc`, which assembles SAHA.md from them.
    - The technical document says how the team used Claude Code: aggregate output only, never
      data (D008, D017).

## WS2: dead ball

PR #14 (`ws2/deadball-evidence`, 6 November) ports WS2's prototype into `saha/deadball`, following
the list that stood here. It does the following:

- **Whistle:** the band and the threshold come from outside this match (whistle makers'
  published tones, and TU Dortmund's labelled recordings). The whistle is off by default.
- **Formation model:** it was fitted on this match, so it is not ported.
- **Dead-ball classifier:** it is trained on 20 open SkillCorner matches and committed with its
  manifest (D020). `python -m saha.deadball.train` rebuilds it byte for byte.
- **Dependencies:** scikit-learn and its dependencies are pinned in `requirements-saha.txt`.
- **Documentation:** `docs/tech/ws2_deadball.md` gives the source and licence of every number.
- **Decoder:** WS2's decoder stays in `saha/deadball`, and the combiner reproduces its decision
  frame for frame (D022).
- **Reproduced on two machines:** the team leader's machine gives the same SAHA.csv SHA256 as
  WS2's. That held on 7 November with the first classifier (`081f57c1…`) and on 10 November with
  the second one, trained with the hidden ball (`3b74ae89…`).

Still open:

1. **Does the classifier transfer?** It learned from 10 fps broadcast tracking in which 29% of
   frames are dead. On this match it calls about half of the frames dead. Score WS2 on WS5's DFL
   matches, which are 25 fps and full pitch like ours, to see whether it transfers. That is a
   check, not a tuning.
2. **One external set, with WS5.** Move the SkillCorner loader from `saha/deadball/skillcorner.py`
   into `saha/external`, together with Seyyid.
3. **The whistle stays off in the submission** (D025, decided on 10 November). The detector
   and its fingerprint tool stay in the code, off by default.
