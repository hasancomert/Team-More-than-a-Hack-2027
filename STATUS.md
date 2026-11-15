# Status

Updated **15 November 2026** by WS1. This page says where we are and what is left; WS1 updates
it at every milestone. The plan itself is [TEAM.md](TEAM.md), the rules are
[CLAUDE.md](CLAUDE.md), the decisions [DECISIONS.md](DECISIONS.md).

## The rules on training and tuning (5 November)

The organizers' rules and answers are in DECISIONS D016 and D017; CLAUDE.md rules 1 and 2 follow
them. They say that "anything that is not explicitly allowed is not permitted". In short:

- Train, fit, calibrate and tune on **external data only**. Nothing is fitted on this match,
  its tracking and tags included.
- Never check our output against the video.
- Automatic checks on this match's data are allowed. They catch errors; they never choose a
  value. The public 5 minutes check the pipeline and choose nothing.
- Never upload the data to AI services (D008).

## CI runs locally (10 November)

GitHub does not start the CI workflow on this repository (D023). A red "checks" with "The job was
not started" on a pull request is expected. Before a PR, run `python -m saha.tools.ci`; WS1 runs
it again on the PR merged into `main` before the team leader merges.

## Where we are

**The submission is final.** `d276615` gives `submissions/SAHA.csv` with SHA256 `af770330…` on
Hasan's and Baki's machines, with the same seven evidence files. PR #21 put it in `main` on
14 November and the team leader signed FINAL_CHOICE.md. The technical document is
[SAHA.md](SAHA.md): the main part and one short appendix per package.

**Three models were added after the freeze**, each with a rule fixed before its runs (PR #21):

| change | decision | where it writes | DFL spellF1, raw / controlled (7 matches × 3 seeds, held out) |
|---|---|---|---|
| the frozen version `225cae0` | | | 0.398 / 0.433 |
| + `saha/shape`: the team from the two teams' shape | D026 | graded frames without the ball | 0.490 / 0.533 |
| + `saha/onball`: the team from the ball and the players around it | D027 | graded frames with the ball | 0.531 / 0.587 |
| + `saha/hiddendead`: a second opinion on WS2's dead calls where the ball is missing | D028 | WS2's dead calls with the ball missing, inside play | 0.549 / 0.609 |

All three are trained on WS5's seven DFL matches only (CC BY 4.0). Records:
`experiments/20261113_ws1_shape-model.md`, `experiments/20261113_ws1_onball-model.md`,
`experiments/20261113_ws1_hiddendead.md`. Two more models for the team where the ball is missing
failed their rules (`experiments/20261113_ws1_learned-bridge.md`,
`experiments/20261113_ws1_long-window-shape.md`).

No more model changes.

## Milestones

| date | what | state |
|---|---|---|
| 28 – 31 Oct | Phase 0: shared infrastructure | done |
| 31 Oct – 11 Nov | Phase 1: the packages | done: WS2 #14, WS3 #12 (v2.1), WS4 #7 (v2), WS5's harness |
| 12 Nov | feature freeze: FINAL_CHOICE.md with its commit hash | done: `225cae0`, reproduced on two machines |
| 13 Nov | shape model (D026), on-ball model (D027), dead-ball second opinion (D028) | done, in PR #21 |
| 14 Nov | the submitted version reproduced on two machines | done: `d276615`, `af770330…` |
| 14 Nov, 18:00 TR | PR #21 merged | done, signed in FINAL_CHOICE.md |
| 13 – 15 Nov | technical document; public-repository preparation | SAHA.md done (PR #23); the repository is not public yet |
| 14 Nov | the form link arrives by email | |
| 16 Nov | submit the form (team leader only) | |
| 17 Nov, 23:59 CET | deadline (Türkiye: 18 Nov, 01:59) | |

## Open items

- **Team leader (Hasan):**
    - Make the repository public before the form, once `ws2/event-deadball` is deleted and
      `python -m saha.tools.guard --history` passes.
    - Submit the form on 16 November: team SAHA, the members' full names, `submissions/SAHA.csv`
      (SHA256 `af770330…`), the repository's URL and SAHA.md.
- **Alparslan (WS2):**
    - D028 gives a second opinion on WS2's dead calls where the ball is missing; WS2's code is
      unchanged. Read it, and say if anything in it is wrong for WS2's decoder.
    - Delete the branch `ws2/event-deadball`, or ask WS1 to; the guard's `--history` check scans
      every branch that will be public.
- **Everyone:** read your own short appendix in SAHA.md (written by WS1 from your section in
  `docs/tech/`) and say if anything in it misstates your work.

## Reproducing the submission

```bash
git checkout <commit>
python -m saha.pipeline --fresh --data <path to the data bundle>
```

Python 3.11 with the pins in `requirements-saha.txt` (TEAM.md, Setup). The pipeline prints the
SHA256 of `submissions/SAHA.csv`; two machines must print the same one.
