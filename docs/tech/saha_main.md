# SAHA: ball possession from tactical tracking

Team SAHA, More than a Hack 2027. Code: TODO (public repository URL). Submission: `SAHA.csv`.

<!-- This file is the cross-package part of the technical document. WS1 writes it on 13-15
November; `python -m saha.tools.techdoc` appends each package's section (docs/tech/ws*_*.md) as an
appendix and writes SAHA.md. SUBMISSION.md asks for the sections below; keep it to a few pages. -->

## Summary

TODO (WS1): one paragraph. What the system does, the final configuration (FINAL_CHOICE.md) and
the external results it rests on.

## The approach

TODO (WS1): the pipeline in one picture, and why it is split this way.

- WS2 (Appendix B) decides whether the ball is dead.
- WS3 (Appendix C) decides who has the ball while it is tracked.
- WS4 (Appendix D) decides who has the ball while it is missing.
- WS1's combiner (Appendix A) fuses the three and turns them into few, long possessions.

## The possession rules we implemented

TODO: where the line is drawn on each of the following, and which package draws it.

- **Loose balls and challenges:** WS3's hysteresis. The ball stays with the last team until the
  opponent controls it.
- **Restarts and stoppages:** WS2.
- **The missing ball:** WS4.
- **How long a possession must be to count:** WS1's switching costs.

## Assets: used and dropped

TODO: one table covering:

- the tracking: positions, the players' speed in m/s, and the ball's speed channel, dropped
  (D015);
- Smart Tagging;
- the homography;
- the video and the whistle;
- each external dataset, with its licence.

## Missing and noisy tracking

TODO:

- The ball is missing on 41% of graded frames, mostly in long stretches (docs/EDA.md).
- WS4's bridge covers those stretches.
- WS2's classifier is trained with the ball hidden as well.
- The tracking slots carry no identities.
- All speeds are computed from positions.

## Training data

TODO: every labelled dataset, with its source, who labelled it, how much, and what it trained or
tuned. None of it comes from the provided game.

## Results

TODO: the external results (WS5's DFL matches, WS2's SkillCorner folds), then the checks on our
match. The public 5 minutes appear only as a pipeline check.

## Limitations

TODO.

## Reproduction

```bash
python -m saha.pipeline --fresh --data <path>
```

TODO: Python 3.11 and `requirements-saha.txt`. Give the SHA256 and the two machines that
reproduced it (FINAL_CHOICE.md).

## How we kept to the rules

TODO:

- The strict reading of the organizers' rules (DECISIONS D016, D017).
- No hand labels; the public 5 minutes checked the pipeline and nothing else.
- The match data never left members' machines.
- AI assistants wrote code and saw aggregate output only (D008, D017).

## Appendices

Each package's section, as its owner wrote it.
