# Decisions

A short, dated, append-only log of team-level decisions. Newest entries go at the bottom. To
change a decision, add a new entry that names the one it supersedes. Interface changes always
come here (CLAUDE.md).

## 2026-10-28 (Phase 0, WS1)

- **D001 Repository.** We use the private repository `hasancomert/Team-More-than-a-Hack-2027`,
  created from the official repository with its full history: 6 upstream commits, MIT LICENSE
  kept. It is not a fork, because a fork of a public repository cannot be private. The
  `upstream` remote is the official repository, with push disabled. WS1 checks upstream twice a
  week (`python -m saha.tools.upstream_check`) and merges changes through a PR.
- **D002 Environment.** Python 3.11. `requirements-saha.txt` pins the numeric stack: numpy
  2.3.5, pandas 2.3.3, pyarrow 21.0.0, scipy 1.16.3, kloppy 3.18.0, matplotlib 3.10.7. The
  upstream requirements files stay untouched. Changing a pin needs a new entry here and a fresh
  reproduction check.
- **D003 Evidence interface v1.**
    - Exactly the four columns described in CLAUDE.md.
    - Values are quantised to 1e-4 at write time.
    - One file per package; variants are named `<package>_<variant>`.
    - `configs/pipeline.toml` decides which evidence files are used, and with what scale.
- **D004 Exact arithmetic in the combiner.** Fixed-point log-odds (1e-10 nats per step) and an
  integer Viterbi decoder with fixed tie-breaking. The submission is then byte-identical on
  Windows/macOS/Linux and x86/ARM.
- **D005 Package layout.** `deadball` (WS2), `visible` (WS3), `occluded` (WS4), `external`
  (WS5). WS1 owns `common`, `combine`, `nearest`, `pipeline` and `tools`.
- **D006 Inputs for now.** The four tactical files only: ATD metadata and raw, Smart Tagging,
  homography. The videos and the broadcast feed are not used until a team decision says so.
- **D007 Git.**
    - No direct pushes to `main`. A pre-push hook enforces this locally. GitHub branch
      protection is added if the account's plan offers it for private repositories.
    - Every change is a PR with at least one approval. `saha/common` changes need WS1's
      approval.
    - Hooks come from `python -m saha.tools.setup_dev`.
- **D008 (accepted by the team leader, 2026-10-31) Data and LLMs.** FAQ.md and OVERVIEW.md quote a T&C
  clause: no making the competition data, or extracts of it, available to a third-party hosted
  or cloud LLM. So:
    - No match data or frame-level derivative ever reaches an LLM.
    - Claude Code sessions write code and run only commands with aggregate output.
    - `.claude/settings.json` denies reading the data paths. The deny covers the file tools and
      the common shell readers. It does not cover Python scripts or Windows PowerShell, so it is
      a guardrail; the rule itself is in CLAUDE.md.
    - The Claude Code sandbox would also cover scripts on macOS, Linux and WSL2. It is not
      enabled in the shared file, because it restricts network and writes. Members may enable
      it in `.claude/settings.local.json`.
    - Data-dependent runs happen on members' own machines.
- **D009 (accepted by the team leader, 2026-10-31) No game-specific constants.** The only frame numbers in
  the model are the published graded-half boundaries, and they are used only to write X outside
  the halves. Everything else must come from an algorithm, including the pre-kick-off D in the
  second half (110808–113741).

## 2026-10-29

- **D010 Team size.** We are five: one owner per work package, and WS1 is the team leader. The
  four-person fallback is dropped from TEAM.md.
- **D011 Package boundaries.** Each frame gets at most one opinion per question:
    - WS2 owns `p_dead`, plus an optional `p_home` inside restart windows only.
    - WS3 writes `p_home` only where the ball is tracked (`ball_visible`).
    - WS4 writes `p_home` only where the ball is missing, short gaps included.
    - Packages read anything but write only on their own frames.
    - The pipeline warns about `visible`/`occluded` overlap.
  Details: TEAM.md, "Boundaries between packages".

## 2026-10-31

- **D012 Speed is in m/s.** The metadata labels the players' speed channel `speed_kph`, with unit
  `kph`. The values are metres per second, as DATA.md says. In the public window the players'
  mean speed is 2.3 in play and 1.2 when the ball is dead (docs/EDA.md); in km/h those would be
  walking speeds. The frame table keeps the values as delivered and calls them m/s. WS1 confirms
  it with a displacement-based check in Phase 1.
- **D013 Owners, so far.** WS1 Hasan (team leader), WS2 Alparslan. Baki takes WS3 or WS4. The
  other of the two, and WS5, are still open; they are added here and in TEAM.md once decided.
- **D014 Python 3.11 for everyone.** Confirmed after a branch was written for Python 3.14. Every
  package runs on Python 3.11 with the pins in `requirements-saha.txt`, because the submission
  must come out byte-identical on every member's machine (D002). A missing dependency (for
  example scikit-learn) is added there, pinned, through a PR.

## 2026-11-04

- **D015 The ball's speed comes from its positions.** The team leader ran
  `python -m saha.tools.data_checks` on the data, on `main` at 8d7ca72:
    - The players' speed channel follows their displacements: the median ratio is 0.99 over
      996,690 samples. This confirms D012.
    - The ball's channel does not: the median ratio is 0.19, over only 6,806 samples.

  Until we know what the ball's channel measures, nothing uses it as a speed:
    - `saha.common.kinematics.displacement_speed` gives the ball's speed from its positions.
    - `kinematics.nearest_player` gives the nearest player, with the same arithmetic as the
      nearest-player evidence.
    - The frame table keeps the channel as delivered (`ball_speed`), for diagnostics only.
      `python -m saha.tools.data_checks --speed` describes it.

## 2026-11-05

- **D016 (the organizers' rules, received 5 November) Train and tune on external data only.**
  The organizers wrote:
    - Not allowed: checking the output against the video; hand labels, human-produced data or
      any other data created manually on the provided game; uploading the provided data to AI
      services; training, fitting or calibrating on the ground-truth possession data or on the
      provided video.
    - Allowed: automatic checks using only the provided data; inspecting the provided data and
      the video to better understand what counts as possession; the video, the tracking, the
      Smart Tagging and other generated data as inputs, as long as nothing was trained on the
      ground truth or on the provided video; providing files of intermediate steps; any dataset
      we have the rights to use, after checking its licence.
    - Training and tuning: any training uses external data only, and parameters that need
      tuning are tuned on external data, never on the ground truth or the provided video.

  What it changes for us (CLAUDE.md rules 1 and 2 follow it):
    - "External data only" can be read as "nothing from this match". Until the organizers
      answer our questions, we read it that way: no training or fitting on this match's tracking
      or tags either, self-supervised included.
    - The self-check and the match-wide statistics are checks. They catch errors; they never
      choose a value. Neither does the public score, nor its bootstrap interval.
    - WS2's prototype fits two models on this match: the dead-ball classifier, trained on labels
      that rules make from the tracking, and the team-shape model. Both need external training
      data, rules with reasoned constants, or the organizers' approval (STATUS.md).
    - The whistle band comes from outside this match. Estimating it from this match's audio may
      count as calibrating on the provided video.
    - Intermediate files may go to the organizers if they ask for them. The public repository
      still holds no match data (D008, CLAUDE.md rule 3).
- **D017 (the organizers' answer, 5 November) The strict reading of D016 is final.** We asked
  the organizers four questions:
    - self-supervised fitting on this match's tracking;
    - label-free checks that choose a parameter;
    - estimating the whistle from this match's audio;
    - showing aggregate statistics to an AI coding assistant.

  They gave no further detail and repeated the rules. They added: "Each team is responsible for
  making sure that its particular project, tool, platform or dataset complies with the
  competition rules. As a reference, anything that is not explicitly allowed is not permitted."
  So:
    - Nothing is trained, fitted, calibrated or tuned on any of this match's data: tracking,
      tags, video or ground truth. That includes self-supervised models, and values that an
      algorithm estimates from this match on every run (a tag's meaning, a lag, a whistle
      frequency).
    - Values come from WS5's external data, or from football and physics reasoning. A reasoned
      value is never adjusted to this match's output.
    - This match's data is used only as input, for automatic checks, and to understand what
      counts as possession.
    - The answer does not address aggregate statistics shown to an AI assistant. The team leader
      decided on 5 November to keep D008 as it is:
        - Members may show Claude Code aggregate output only: counts, percentages, percentiles
          and scores.
        - No data and no frame-level derivative ever reaches an LLM.
        - The technical document says so.
- **D018 Owners: Baki takes WS4.** Baki opened WS4's first evidence (PR #7). WS3 and WS5 are
  still open. This replaces the open part of D013.

## 2026-11-06

- **D019 Owners: Seyyid takes WS5** (GitHub planckestein, PR #9). Only WS3 is still open.
- **D020 (accepted by the team leader, 6 November) Model files trained on external data.** A
  model trained only on external open data is not match data, so it may be committed:
    - under `saha/<package>/models/`, as `.pkl` or `.joblib`, up to 5 MB;
    - with a manifest next to it, `<file>.json`. The manifest names the training script (in the
      repository), the external data and its licence, the seed, the library versions and the
      model's SHA256.

  The guard allows exactly that path and size, and checks the manifest on every commit and in
  CI. Pickles and big files stay forbidden everywhere else. A model fitted on any of this
  match's data is never allowed (D017).
- **D021 Owners: Osman takes WS3** (GitHub osmaneren151, PR #12). Every package now has an
  owner: WS1 Hasan, WS2 Alparslan, WS3 Osman, WS4 Baki, WS5 Seyyid.

## 2026-11-07

- **D022 (accepted by the team leader, 7 November) WS2's decoder stays in `saha/deadball`.**
    - WS2 decides dead or alive with its own event-gated decoder, validated on held-out
      external matches.
    - It hands the combiner that decision as `p_dead` 0.99 / 0.01 at weight 20, and the
      combiner reproduces it frame for frame.
    - The decoder's ideas do not move into WS1's combiner before the feature freeze. We revisit
      this only if WS5's external harness shows that fusing beats it.
    - The pipeline tests run WS2's builder without its classifier (`SAHA_DEADBALL_MODEL=none`).
      The four-player synthetic match is no input for a model trained on real matches, and
      `tests/test_deadball.py` tests the package itself.

## 2026-11-10

- **D023 (team leader, 10 November) GitHub Actions is off; the CI runs locally.** Since 10
  November GitHub does not start the workflow on this private repository: the account's Actions
  billing would have to be raised, and the team leader chose not to pay. So:
    - `python -m saha.tools.ci` runs every step of `.github/workflows/ci.yml` locally, with
      synthetic data and public files only.
    - Every author runs it before opening a PR. WS1 runs it again on the PR merged into
      `main`, in a cloud session, and reports the result before the team leader merges.
    - A red "checks" saying "The job was not started" is expected. It is not a failure of the
      code.
- **D024 (the team, 10 November) We keep the strict reading of D017.** RULES.md, as published
  in the repository, lists among what remains fine on the provided game: "Learning without human
  labels — unsupervised or self-supervised methods, or pseudo-labels produced by an algorithm,
  provided no person decided any of them." The organizers' clarification of 5 November says that
  any training must use external data only, and that anything not explicitly allowed is not
  permitted. The two texts disagree. WS1 had missed the RULES.md sentence when it wrote D017.
  The team keeps the stricter reading:
    - The gain from relaxing it is small. WS2's team-shape model was worth 0.006 spellF1 on
      external data, and every such model can be trained on WS5's data instead.
    - The cost of being wrong would be disqualification.

  The technical document states this choice.
- **D025 (the team, 10 November) The whistle stays off in the submission.** This closes D006.
    - Its benefit cannot be measured on external data, because no external set has audio.
    - Using it would require the tactical video (5.2 GB) and ffmpeg on every machine that
      reproduces the submission.
    - It would also put the byte-identical reproduction at risk across ffmpeg versions.

  WS2's detector stays in the code, off by default.
