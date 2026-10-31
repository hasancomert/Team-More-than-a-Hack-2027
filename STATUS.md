# Status

Updated **31 October 2026** by WS1. This page says where we are and what to do next; WS1 updates
it at every milestone. The plan itself is [TEAM.md](TEAM.md), the rules are
[CLAUDE.md](CLAUDE.md).

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
    - The speed channel is in m/s, even though the metadata says km/h (DECISIONS D012).

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
| 13 – 15 Nov | technical document; reproduction on two machines (SHA256); public-repository preparation | |
| 14 Nov | the form link arrives by email | |
| 16 Nov | submit the form | |
| 17 Nov, 23:59 CET | deadline (Türkiye: 18 Nov, 01:59) | |

## Open items

- **Team leader:**
    - Name the remaining owners: Baki takes WS3 or WS4; the other of the two, and WS5, are
      still open (TEAM.md).
    - Turn on branch protection for `main`, if the GitHub plan offers it.
- **WS2:** fold the event-driven dead-ball prototype on branch `ws2/event-deadball` into
  `saha/deadball`. It was written before Phase 0 as a separate pipeline, so it needs to move to
  Python 3.11, the shared frame table and the evidence interface. Its referee-whistle signal
  reads the tactical video with ffmpeg, which needs a team decision first (D006).
- **WS1:**
    - Load the ball's speed channel into the frame table.
    - Confirm the speed unit from player displacements (D012).
    - Stop the combiner from suppressing every possession shorter than 2 s. The placeholder
      costs do that today, and docs/METRIC_NOTES.md shows that short possessions are where the
      score is.
