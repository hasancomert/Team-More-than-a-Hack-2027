# TEAM.md: SAHA plan

Rules for humans and Claude Code: [CLAUDE.md](CLAUDE.md). Decisions: [DECISIONS.md](DECISIONS.md).
How the metric punishes mistakes: [docs/METRIC_NOTES.md](docs/METRIC_NOTES.md).

## Before you start (every member)

- [ ] Registered for More than a Hack 2027 with the team name **SAHA** (exactly, upper case).
      Last day: **10 November**.
- [ ] Downloaded the data from **your own** enrolment link. Never share a link, a file, or a
      piece of the data, with anyone or anything. That includes chat apps, Drive and LLMs
      (CLAUDE.md, rule 3).
- [ ] Python **3.11** virtual environment, set up as below. `python -m saha.tools.setup_dev`
      prints only OK lines.
- [ ] Accepted the GitHub collaborator invitation, and have a Google account (the form needs one).
- [ ] Read CLAUDE.md. If you use Claude Code:
    - Start it from the repository root, so that `.claude/settings.json` is loaded.
    - Never let it open the data. Have it write the code, and run anything data-dependent
      yourself.

## Setup

Put the four tactical files, names unchanged, in a `tactical` folder inside a data folder of your
choice, for example `D:\mtah2027\tactical\BAR-ATH_tactical_atd.txt`. `MTAH_DATA_DIR` points at
the data folder (`D:\mtah2027`), not at `tactical`. The videos are not needed.

Windows (PowerShell). Install Python 3.11 first if `py list` does not show it: run
`py install 3.11`, or use the 3.11.9 installer from python.org.

```powershell
git clone https://github.com/hasancomert/Team-More-than-a-Hack-2027.git
cd Team-More-than-a-Hack-2027
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-starter.txt -r requirements-saha.txt
$env:MTAH_DATA_DIR = "D:\mtah2027"      # this shell
setx MTAH_DATA_DIR "D:\mtah2027"        # every future shell
python -m saha.tools.setup_dev
```

macOS / Linux. Install Python 3.11 first if `python3.11 --version` fails: use python.org, or
`brew install python@3.11`.

```bash
git clone https://github.com/hasancomert/Team-More-than-a-Hack-2027.git
cd Team-More-than-a-Hack-2027
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-starter.txt -r requirements-saha.txt
export MTAH_DATA_DIR=~/mtah2027          # add it to ~/.zshrc or ~/.bashrc too
python -m saha.tools.setup_dev
```

`python -m saha.tools.setup_dev` must print only OK lines. Then `python -m saha.pipeline` builds
the frame table, writes `submissions/SAHA.csv`, validates it and scores it, in a few minutes.
`python --version` inside the activated environment must say 3.11.

Every new shell: `cd` into the repository and activate the environment again
(`.venv\Scripts\Activate.ps1`, or `source .venv/bin/activate`).

## Work packages

| package | owner |
|---|---|
| WS1 infrastructure and integration | Hasan (team leader) |
| WS2 dead ball | Alparslan |
| WS3 possession while the ball is tracked | Baki or _TBD_ (Baki takes WS3 or WS4) |
| WS4 possession while the ball is missing | Baki or _TBD_ |
| WS5 external data and independent validation | _TBD_ |

Each package writes one evidence file (interface in CLAUDE.md) from `saha/<package>/build.py`.
The pipeline already runs every package: today they are stubs with no opinion, so your first PR
only replaces `build()`.

The first delivery is a **simple but working** evidence file, by **4 November**. "Working" means
three things:

- it validates
- the pipeline runs with it
- you wrote an experiment record

"Success" is never measured on the public 5 minutes alone (CLAUDE.md, rule 2).

**Read [docs/EDA.md](docs/EDA.md) before you start.** It was made from the real data, and it
changes some of the starting ideas below (marked "EDA").

### Boundaries between packages

Every frame gets at most one opinion per question, so that no signal is counted twice and the
pieces fit together without negotiation:

| question | frames | package |
|---|---|---|
| `p_dead`: is the ball dead? | every frame | WS2 `deadball` |
| `p_home` while the ball is tracked | frames where `ball_visible` is true | WS3 `visible` |
| `p_home` while the ball is missing | frames where `ball_visible` is false, short gaps included | WS4 `occluded` |
| `p_home` from a restart's team | inside a restart window only (optional) | WS2 `deadball` |

- **Read anything, write only on your own frames.** WS4 may use the last and next ball sightings,
  and WS3 may look at the frames around a sighting. But the opinion is written only on your own
  frames: `p_home` NaN and weight 0 everywhere else.
- Everyone splits the frames with the same mask: `frame_table.load().ball_visible`.
- `nearest` (WS1) answers WS3's question as the baseline. Once WS3 beats it on WS5's validation
  data, WS1 lowers or zeroes its scale in `configs/pipeline.toml`, with an experiment record.
- The pipeline warns when `visible` and `occluded` both give `p_home` on the same frame.
- Moving a boundary is a team decision (DECISIONS.md).

### WS1: infrastructure and integration (owner: Hasan, team leader)

- **Goal.** One reproducible pipeline that turns everyone's evidence into the best submission.
- **Owns.** The shared data layer (`saha/common`), the evidence interface and validator, the
  combiner and temporal model (`saha/combine`), the pipeline, the metric and evaluation tools,
  and the submission package.
- **Inputs.** Every evidence file, and WS5's validation sets.
- **Outputs.** `submissions/SAHA.csv` with its SHA256, joint score reports on 5 and 9 November,
  FINAL_CHOICE.md, and the assembled technical document.
- **First delivery.** Done in Phase 0: the pipeline runs end to end with the nearest-player
  evidence.
- **Success.**
    - Two members get the same SHA256 from clean installs.
    - On WS5's external data, the combiner beats the best single source.
    - Match-wide statistics look like football.
- **Starting ideas.**
    - An HMM/Viterbi (or similar) segmentation over H/A/D.
    - Fuse the evidence as log-odds.
    - Flicker costs a lot (docs/METRIC_NOTES.md), so the switching costs favour few, long
      possessions. Their values are tuned on WS5's external data (D016).
    - Use X only in mid-possession.

### WS2: dead ball (D) (owner: Alparslan)

- **Goal.** `p_dead` on every graded frame. The exact start and end of every stoppage matter
  just as much (both boundaries shape the possessions around them).
- **Inputs.**
    - the frame table: ball position and visibility, player positions and speeds
    - Smart Tagging restarts
    - optionally, the homography
- **Output.** `cache/evidence/deadball.parquet`: `p_dead`. Add `p_home` only inside a restart
  window whose team you know (see the boundaries above).
- **First delivery.** Ball outside the pitch, plus Smart Tagging restart windows, as `p_dead`.
- **Success.**
    - Good alive/dead F1 and restart timing on WS5 data that has ball status.
    - Stoppage count and durations are plausible match-wide. Every Smart Tagging restart is
      explained by a nearby stoppage.
    - No stoppages shorter than about a second.
- **Starting ideas.**
    - the ball is outside the pitch
    - the ball is unseen for a long time while the players slow down
    - set-piece formations
    - A Smart Tagging restart (Type: Throw-in, Corner Kick, Goal Kick, Free Kick, Kick Off)
      *ends* a dead-ball period.
    - After a goal, the ball is D until the kick-off.
    - EDA: the ball is outside the 0–1 pitch on 4.2% of visible graded frames, in 81 excursions
      with a median of 0.8 s. Some readings lie far off the pitch (y up to 1.34), so filter
      implausible positions before trusting them.
    - EDA: in the public window, restart tags are long windows, only loosely aligned with the
      real restart. Use a tag as the window to search in, and the tracking to find the frame.
    - EDA: players' mean speed in the public window is 1.2 m/s when the ball is dead and 2.3 m/s
      in play. That split uses the public ground truth: it helps to understand the data, but it
      is never a threshold (D016).

### WS3: possession while the ball is tracked (owner: Baki or _TBD_)

- **Goal.** `p_home` on frames where the ball is visible, and the exact frame where possession
  changes.
- **Inputs.** The frame table: ball and player positions and speeds.
- **Output.** `cache/evidence/visible.parquet`: `p_home`, only on frames where the ball is
  tracked (see the boundaries above).
- **First delivery.** The nearest player plus a control condition, as `p_home` with a weight
  that grows with confidence.
- **Success.**
    - Team accuracy and change timing on WS5 data, with the ball visible.
    - Fewer spurious changes than `nearest` at equal or better recall.
- **Starting ideas.**
    - The nearest player, plus a control condition: the ball stays near one player for several
      frames, and the ball speed is low.
    - A free or contested ball stays with the last team. Possession changes only when the
      opponent controls the ball. Model this with hysteresis.
    - The ball has no height channel, and its speed channel does not follow its positions
      (DECISIONS D015). Take the ball's speed from `kinematics.displacement_speed` and the
      nearest player from `kinematics.nearest_player`, both in `saha.common.kinematics`.
    - The ball's positions are clean, on the graded frames where the ball is tracked
      (experiments/20261104_ws1_cost-switch-sweep-2.md):
        - The median step between frames is 0.2 m, and under 1% of steps exceed 1 m.
        - Its speed from positions has a median of 5.1 m/s and a 90th percentile of 14 m/s.
        - A player is within 1.5 m of the ball on 42% of these frames.

### WS4: possession while the ball is missing (owner: Baki or _TBD_)

The ball is missing on **41% of graded frames**.

- **Goal.** `p_home` where the ball is not tracked.
- **Inputs.**
    - the frame table: team shapes, speeds, the last and next ball sightings
    - Smart Tagging phases and their teams
- **Output.** `cache/evidence/occluded.parquet`: `p_home`, only on frames where the ball is
  missing, short gaps included (see the boundaries above).
- **First delivery.** Carry possession into each gap from the last and next ball sightings: who
  had the ball just before, and who has it just after. Add the Smart Tagging phase team as weak
  evidence.
- **Success.**
    - Team accuracy on WS5 data with the ball hidden the way ours is.
    - Calibration: a `p_home` of 0.8 should be right about 80% of the time.
    - Coverage of the missing-ball frames.
- **Starting ideas.**
    - EDA: the gaps are long, not short. There are 150 missing-ball stretches on graded frames,
      with a median of 10.7 s and a maximum of 221 s. Stretches of 10 s or more hold 85% of the
      missing frames, and there are none between 1 and 3 s. The tracker seems to bridge short
      occlusions itself, so interpolating the ball buys little.
    - So the core of WS4 is the team shape: who attacks, the defensive block, pressing.
    - Use the team of each Smart Tagging phase. Treat Team = N/A (every throw-in) separately.
    - Study PathCRF (https://github.com/hyunsungkim-ds/pathcrf), which infers possession
      without the ball. Our tracking has no player identities and its slots break up. Train
      any such model on external data only (D016).

### WS5: external data and independent validation (owner: _TBD_)

- **Goal.** A labelled validation set from other matches' open data, made to look like ours.
  Parameters are chosen there, not on the 17 public possessions.
- **Inputs.** Open data only, each with a recorded licence and attribution:
    - Metrica sample data (https://github.com/metrica-sports/sample-data): the same 0–1
      coordinates. Game 3 is in the same EPTS format, so `saha.common.epts` reads it.
    - SkillCorner Open Data (https://github.com/SkillCorner/opendata): 10 fps, possession on
      every frame.
    - DFL/Sportec open data (kloppy `sportec.load_open_tracking_data`): kloppy reads
      `BallPossession` and `BallStatus`. **First** check that the open files actually fill them.
- **Output.** `cache/external/` (gitignored), a harness that scores our pipeline there with the
  official metric, and `docs/tech/ws5_external.md` (licences, attribution, protocol).
- **First delivery.** One dataset, labels mapped to H/A/D, scored with the official metric.
- **Success.** Other packages use it to choose their parameters. The possession definition is
  checked against the public 5 minutes (to see whether the definitions agree, not to tune).
- **Starting ideas.**
    - Degrade the data to our conditions: hide the ball on about 41% of frames, in occlusion-like
      stretches; remove identities; split the slots.
    - Their possession definition may differ from ours: measure the difference before trusting
      it.

### Other notes

- Ball detection from the video is allowed as an input, if the detector is not trained on this
  match's video (D016). It is heavy and needs a team decision before anyone starts.
- Labelled data is scarce: 7,500 frames and 17 possessions. Rule-based and probabilistic models
  come first. Any training or fitting, supervised or not, uses external data only (D016).

## Calendar

| when | what |
|---|---|
| 28 Oct – 1 Nov | Phase 0, shared infrastructure (WS1). Everyone else: register as SAHA, download the data, set up the environment. |
| 2 – 11 Nov | Parallel work. **4 Nov**: every package has its first working evidence file. **5 and 9 Nov**: joint score reports (WS1). |
| 10 Nov | Last day for every member to register as **SAHA**. |
| **12 Nov** | **Feature freeze.** The final configuration and its rationale go into FINAL_CHOICE.md with the commit hash. Bug fixes only after this. |
| 13 – 15 Nov | Technical document: each package writes `docs/tech/<ws>_*.md`, WS1 assembles SAHA.md. Reproduction from a clean install on two members' machines, comparing SHA256. Public-repository preparation. **14 Nov**: the form link arrives by email. |
| 16 Nov | Submit the form (one day of buffer). One response per team. |
| 17 Nov, 23:59 CET | Deadline (Türkiye: 18 Nov, 01:59). |

## How we decide

- The organizers' rules (DECISIONS D016): train, fit and tune on external data only, never on
  this match's ground truth or video.
- The public 5 minutes check correctness only. Four or five people looking at the same 17
  possessions over-fit together. No change is merged because the public score went up.
- Parameters come from:
    - WS5's external data, where they are tuned
    - football and physics reasoning, with no fitting
- Match-wide statistics (`python -m saha.pipeline` prints possession count, durations and D
  rate) and the self-check catch errors. They never choose a value.
- Every significant change gets an experiment record (experiments/README.md). Team-level
  decisions go into DECISIONS.md.
