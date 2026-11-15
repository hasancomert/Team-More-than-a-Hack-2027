## Appendices

Each package in brief, written by WS1 from its owner's section. The full sections, as their owners
wrote them, are in the repository under
[`docs/tech/`](https://github.com/hasancomert/Team-More-than-a-Hack-2027/tree/main/docs/tech),
and every experiment, adopted or rejected, has a record under
[`experiments/`](https://github.com/hasancomert/Team-More-than-a-Hack-2027/tree/main/experiments).

### Appendix A: WS1, data layer, combiner and two learned models (Hasan Cömert)

- **Data layer** (`saha/common`): one row per video frame. An ATD `frame_id` is frame
  `frame_id − 1` and a Smart Tagging time is frame `round(seconds × 25)`; fed into the official
  baseline, the parser reproduces its four published files with 0 differing frames. Every speed
  comes from positions, because the ball's speed channel does not follow the ball (D015).
- **Combiner** (`saha/combine`): every source's log-odds added in exact integers, then a
  three-state Viterbi over H, A and D with switching costs of 35 nats. The file is the same on
  every machine.
- **`nearest`:** the official nearest-player baseline as evidence, at scale 0.5 (chosen on DFL).
- **Two models, trained on DFL only, each adopted by a rule fixed before its runs:**
    - `onball` (D027), the team where the ball is tracked, from the ball and the players around
      it: 0.908 right, held out, against 0.780 for the nearest player;
    - `hiddendead` (D028), a second opinion on WS2's dead calls where the ball is missing: WS2's
      false dead calls on those live frames fall from 21.6% to 5.6%.
- **Integration** of WS5's `shape` model (D026) into the pipeline, next to WS4 (Appendix E).

Full section: [`docs/tech/ws1_pipeline.md`](https://github.com/hasancomert/Team-More-than-a-Hack-2027/blob/main/docs/tech/ws1_pipeline.md).

### Appendix B: WS2, the dead ball (Alparslan Tuğra Şahin)

- **Three parts** decide dead or alive on every graded frame:
    - a classifier trained on 20 SkillCorner matches (MIT), with copies where WS5's hiding
      removes the ball;
    - stop and restart events from the laws of the game and from physics: the ball crossing a
      line, vanishing while everyone slows down, struck after lying still, a kick-off;
    - an event-gated decoder, in which a stoppage opens only near a stop event and closes only at
      a confirmed restart.
- **Held out on SkillCorner:** spellF1 0.624 and dead-ball F1 0.899; under WS5's hiding, 0.583
  and 0.835. Training with the hidden copies gained +0.158 there.
- **The referee's whistle** is detected from the video's audio, with a threshold set on the TU
  Dortmund recordings, but it stays off in the submission (D025).
- **Limitations:** stoppage starts; only 66–89% of true restarts are detected, slow free kicks
  least of all; a different league and camera.

Full section: [`docs/tech/ws2_deadball.md`](https://github.com/hasancomert/Team-More-than-a-Hack-2027/blob/main/docs/tech/ws2_deadball.md).

### Appendix C: WS3, possession while the ball is tracked (Osman Eren Başaran)

- **The line it draws** is CHALLENGE.md's: possession changes only when the opponent gains
  control. A touch without control, a deflection or a lost duel does not count.
- **Touches:** a team touches the ball when its nearest player is the closest one and within
  1.5 m.
- **Confirmed hand-overs:** the other team's touch counts if it lasts 6 frames or more, or if the
  same team touches the ball again first. The change is placed at that first touch.
- **Evidence:** P = 0.9 on the team's own touches, 0.8 between two of them and 0.6 while the ball
  is loose, at weight 2. The state is forgotten across a gap in the ball's tracking longer than
  0.5 s.
- **On WS5's seven DFL matches:** spellF1 0.319 without WS3 and 0.387 with v2.1 and `nearest` at
  0.5, ahead on 7 of 7 matches.
- **Limitations:** no ball height, so a high ball over a player counts as a touch; identity
  switches near the ball can fake one.

Full section: [`docs/tech/ws3_visible.md`](https://github.com/hasancomert/Team-More-than-a-Hack-2027/blob/main/docs/tech/ws3_visible.md).

### Appendix D: WS4, possession while the ball is missing (Abdülbaki Bayır)

- **A bridge over each gap:** the team at the ball on each side of a gap (touches within 2 s,
  recent ones counting more). Inside the gap, a two-state Markov chain changing team once per
  12 s on average. Near a side the answer follows that side; in a long gap it fades to no
  opinion; when the sides disagree it crosses over once, where a hand-over is most likely.
- **Smart Tagging phases** add weak evidence (at most 0.42 nats per frame). Each code's side is
  fixed by its official definition: BUILD UP is the team with the ball, DEFENDING IN … the team
  without it. Nothing is measured on this match.
- **On WS5's seven DFL matches:** +0.0085 spellF1, ahead on 6 of 7. DFL has no tags, so this is
  the bridge alone.
- **Limitations:** a possession entirely inside a gap is invisible to the bridge; WS1's shape
  model now covers part of it. The tags' timing is as delivered.

Full section: [`docs/tech/ws4_occluded.md`](https://github.com/hasancomert/Team-More-than-a-Hack-2027/blob/main/docs/tech/ws4_occluded.md).

### Appendix E: WS5, external data and independent validation (Seyyid Abdülbaki Bozkurt)

- **The data:** seven DFL/IDSSE matches (CC BY 4.0; Bassek, Weber, Rein and Memmert, 2025),
  with DFL's own ball status and possession on every frame at 25 fps, mapped to D, H and A.
- **The degradation**, which makes them look like ours:
    - the ball hidden on 41% of each half in log-normal stretches (median 10.7 s, 3–221 s);
    - no ball height and no ball speed channel;
    - a fresh track slot per person and half.
- **The harness** runs each package's own builder on the degraded matches, fuses them as the
  pipeline does and scores them with the official scorer unmodified. `nearest`'s scale, WS3's
  settings and the three learned models were chosen there; WS2's on SkillCorner.
- **Caveat:** DFL's possession changes on every touch and its dead ball lasts longer, so its
  labels mean less than ours. WS1 therefore also scored everything on a stricter
  "controlled-2s" reading.
- **`shape`: the team without the ball (D026).** Seyyid's oracle runs showed that the team, not
  the dead ball, was the bottleneck of the frozen version. From the literature on team shape
  (PathCRF; Kim et al.) he built a ball-free team model, gradient boosting on 52 team features,
  that raised DFL's spellF1 from 0.394 to 0.470 (raw) and from 0.427 to 0.529 (controlled-2s),
  ahead on 7 of 7 matches.
    - The submitted `shape` is a logistic model of the same idea on 24 team features, which
      scored higher (0.491) and gets the team right on 0.88 of the frames without the ball,
      held out.
    - Seyyid reviewed it and reproduced its result with his own harness: 0.394 → 0.491 raw and
      0.427 → 0.531 controlled-2s, seed 0, ahead on 7 of 7 matches.

Full section: [`docs/tech/ws5_external.md`](https://github.com/hasancomert/Team-More-than-a-Hack-2027/blob/main/docs/tech/ws5_external.md).
