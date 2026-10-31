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
