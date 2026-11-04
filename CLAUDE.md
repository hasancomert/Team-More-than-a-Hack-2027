# CLAUDE.md: team SAHA, More than a Hack 2027 (Phase 1: ball possession)

Every Claude Code session on this repository reads this first. The rules apply to humans too.
Team plan and owners: TEAM.md. Where we are now: STATUS.md. Decisions: DECISIONS.md.
Metric details: docs/METRIC_NOTES.md. The data's main facts: docs/EDA.md.
Before working on a package, read its open items in STATUS.md; they come first.

## The task

- Label every frame 0…199318 of the tactical video of FC Barcelona (**H**) vs Athletic Club
  (**A**). Each label is `H`, `A`, `D` (dead ball) or `X` (no answer).
- Output: `submissions/SAHA.csv`, header `frame,label`, exactly 199,319 rows.
- Ranked on **spellF1** (possessions matched by temporal IoU). Ties are broken by macroF1.
- Baseline to beat: spellF1 **0.2433**.
- Deadline: 17 Nov 2026, 23:59 CET. Deliverables:
    - SAHA.csv
    - this repository (made public at the end)
    - a technical document, SAHA.md or SAHA.pdf
- Official docs, in this order: CHALLENGE, DATA, EVALUATION, SUBMISSION, RULES, FAQ, OVERVIEW.
  If EVALUATION.md and `evaluation/possession_metrics.py` disagree, the code wins.

## Hard rules

1. **Labels come from an algorithm only.** These are forbidden:
    - hand labels
    - hand-corrected outputs
    - rules or constants tied to specific frames of this match

   The published graded-half boundaries are the only exception.
2. **The only ground truth for this match is the public 5 minutes.** Use it only to check that
   the pipeline is correct. Never use it to train, tune, or choose between two versions. Labelled
   open data from *other* matches is fine (WS5). It must be public, and the technical document
   must explain it.
3. **The match data never leaves members' machines.** This covers the data itself and anything
   derived from it frame by frame: tracking, tags, homography, video, frame tables, evidence
   files, caches, per-frame CSVs.
    - It never goes into git, Drive or chat apps.
    - It never goes to an LLM. The competition T&C forbid making the data, or any extract of it,
      available to a cloud-hosted LLM (FAQ.md, OVERVIEW.md). Claude is one.

   For a Claude Code session this means:
    - **Never read, print, `head`, `cat`, grep or load** anything under `data/`, `cache/`,
      `submissions/` or `$MTAH_DATA_DIR`, or any `*.parquet` or `BAR-ATH_*` file. This applies
      to the Read tool, Bash and Python alike.
    - `.claude/settings.json` denies the file tools and the common shell readers (cat, head,
      tail, sed) there. It cannot stop a Python script or Windows PowerShell, so the rest is
      on you.
    - Start Claude Code from the repository root, or the settings file is not loaded.
    - Write code that reads the data, and let the human run it. Run a command yourself only if
      its output is aggregate: scores, counts, percentages, hashes. If an output could show
      values from the data, do not run it.
    - Never write download links, file IDs or enrolment details anywhere in the repository.
4. **One command reproduces the submission to the byte** on any member's machine:
   `python -m saha.pipeline --data <path>`. This needs:
    - deterministic code and fixed seeds
    - Python 3.11
    - the pins in `requirements-saha.txt`
5. **Never modify `evaluation/` or `games.py`.** That is the official scorer. Other upstream
   files are read-only too, except `README.md` and `.gitignore` (WS1 only).
6. **Never do any of these:**
    - push to `main`
    - merge a PR
    - force-push someone else's branch
    - change repository settings, visibility or collaborators
    - submit the form

   Push only your own branch, and only when your human asks.

## The metric in one minute

Full study: docs/METRIC_NOTES.md.

- A possession is a maximal run of H or of A. D separates possessions: `H D H` is two.
- Predicted and true possessions of the same team are matched one-to-one by IoU. F1 is averaged
  over IoU 0.3–0.7. A 1 s possession counts as much as a 40 s one.
- Flicker kills the score: tiny possessions are false positives. Each of these costs about 1–2
  possessions:
    - a false split
    - a missed hand-over
    - a wrong team
    - a fake or missed stoppage
- A 1-frame offset is free. Being 8 frames late gives about 0.89 on the full match, and 1 s late
  gives about 0.65. Short possessions are where boundary precision matters.
- `X` is forward-filled with your last label in the same half. In mid-possession it is free. At
  a hand-over it costs exactly as much as being late. macroF1 counts every `X` as a miss.
- Graded frames: **11185–90651** and **110808–190197**. Frames 110808–113741 (before the
  second-half kick-off) are all D. Every other frame is ungraded, so write `X` there.
- On the full match, spellF1 gaps below ~0.04 are noise. Run `--bootstrap 1000`.
- Do not reverse-engineer private ground-truth statistics from published numbers.

## Frame alignment

The only source is `saha/common/alignment.py`. Never re-implement it.

| source | index | video frame (= submission row) |
|---|---|---|
| ATD | `frame_id` 1…199319 | `frame_id − 1` |
| Smart Tagging | seconds | `round(seconds × 25)` (Python round, like the starter) |
| homography | `frame` 0…199318 | already a video frame |

Read the match only through `saha.common`:

- `frame_table.load()` gives arrays per video frame.
- `smart_tagging.load()` and `homography.load()` load the other two sources.

## Evidence interface

How packages talk to the combiner. Each package writes `cache/evidence/<name>.parquet` with
exactly 199,319 rows and exactly these columns:

| column | type | meaning |
|---|---|---|
| `frame` | int32 | 0…199318 in order |
| `p_dead` | float32 | P(ball dead); NaN = no opinion |
| `p_home` | float32 | P(Barcelona has the ball \| ball alive); NaN = no opinion |
| `weight` | float32 | ≥ 0, how much to trust this row; 0 = no opinion |

- Write the file with `saha.common.evidence.write_evidence(...)`. It validates the arrays and
  quantises them to 1e-4.
- Check every file with `python -m saha.common.evidence`.
- The builder is `saha/<package>/build.py`, run with `python -m saha.<package>.build`.
- Boundaries (TEAM.md): WS2 owns `p_dead`. WS3 writes `p_home` only where the ball is tracked,
  and WS4 only where it is missing. Read anything, but write only on your own frames.
- The combiner (`saha/combine`) fuses log-odds in integers and runs a Viterbi pass over H/A/D.
- Changing the interface is a team decision, recorded in DECISIONS.md.

## Who owns what

| path | owner |
|---|---|
| `saha/common/`, `saha/combine/`, `saha/nearest/`, `saha/pipeline.py`, `saha/tools/`, `configs/` | WS1 |
| `saha/deadball/` (D detection) | WS2 |
| `saha/visible/` (possession while the ball is tracked) | WS3 |
| `saha/occluded/` (possession while the ball is missing) | WS4 |
| `saha/external/` (external data, independent validation) | WS5 |
| `docs/tech/<ws>_*.md` | each WS, its own section |
| `experiments/` | everyone, own files only |
| `tests/` | the owner of the code under test |
| `evaluation/`, `games.py`, `starter/`, upstream `*.md` | upstream, read-only |

## Git flow

- No direct pushes to `main`. Everything enters through a pull request.
- Use one branch per change: `ws2/<topic>`, `ws3/<topic>`, and so on.
- Keep PRs small. Each needs at least one approval from another member.
- Changes to `saha/common/` also need WS1's approval.
- Before opening a PR:
    1. `python -m pytest`
    2. `python -m saha.pipeline`, if you have the data
    3. an experiment record
- The hooks block the following (`python -m saha.tools.setup_dev` installs them, CI re-checks):
    - `data/`, `cache/`, `submissions/`
    - data-dump formats
    - files over 500 KB
    - frame-level tables
    - edits to `evaluation/` or `games.py`
    - pushes to `main`

## Experiment records

After every significant change:

1. Run `python -m saha.pipeline`. It validates the file and prints the public spellF1 with a
   1000-draw bootstrap, plus match-wide possession count, mean duration and D rate.
2. Run `python -m saha.common.experiments new <ws> <short-name>`.
3. Fill in what changed, why, the result and the decision. Commit the record with the change.

Never merge a change only because the public score went up. Choose parameters from:

- football and physics reasoning
- match-wide consistency (possession count, durations, D rate)
- bootstrap intervals
- WS5's external validation set

## Determinism

- Fix every seed.
- Nothing may depend on the wall clock or on directory-listing order (sort file lists).
- No parallel reductions that change summation order.
- Evidence is quantised to 1e-4 and the combiner works in exact integers. Keep it that way.

## Commands

All of them run from the repository root.

```bash
python -m saha.tools.setup_dev              # once: hooks, upstream remote, environment check
python -m saha.pipeline --data <path>       # the submission, validated and scored
python -m saha.common.frame_table           # build/check the shared frame table cache
python -m saha.<package>.build              # one evidence file
python -m saha.common.evidence              # validate every evidence file
python -m saha.common.experiments           # every experiment in one table
python -m saha.tools.selfcheck              # combiner settings, label-free (after a pipeline run)
python -m pytest                            # tests (synthetic data only)
python evaluation/validate_submission.py submissions/SAHA.csv
python evaluation/run_evaluation.py --bootstrap 1000
```
