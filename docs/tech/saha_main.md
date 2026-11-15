# SAHA: ball possession from tactical tracking

Team SAHA, More than a Hack 2027 (FC Barcelona – Athletic Club). Code:
https://github.com/hasancomert/Team-More-than-a-Hack-2027. Submission: `SAHA.csv`.

**Team SAHA:** Hasan Cömert (WS1, team leader), Alparslan Tuğra Şahin (WS2), Osman Eren Başaran
(WS3), Abdülbaki Bayır (WS4), Seyyid Abdülbaki Bozkurt (WS5). WS stands for work package: each
member owned one part of the system ("owner" below).

<!-- Assembled into SAHA.md by `python -m saha.tools.techdoc`: this file, then one short appendix
per package (docs/tech/appendix_summaries.md), each linking to its owner's full section
(docs/tech/ws*_*.md); `--full` appends the full sections instead. SUBMISSION.md asks for: the
approach, the assets used and dropped, the training data, the possession rules, the limitations
and the reproduction. -->

## Summary

We label each frame from seven sources of evidence and one decoder. Each source answers one of
two questions on its own frames: is the ball dead, and, if it is alive, which team has it. An
exact integer combiner adds their log-odds, and a Viterbi pass over H, A and D turns the result
into few, long possessions.

- **External data only.** Everything that is trained or tuned uses public data that we have the
  right to use: seven DFL/IDSSE matches (CC BY 4.0) and twenty SkillCorner matches (MIT).
  Nothing is fitted on the provided game.
- **The external benchmark.** We degrade the seven DFL matches to our conditions, with the ball
  hidden on 41% of frames in long stretches. On them, the submitted version scores a mean
  spellF1 of **0.55** with DFL's own labels and **0.61** with a stricter "controlled possession"
  reading of them. The nearest-player baseline with our dead ball scores 0.32 on the same
  matches. The official baseline scores 0.2433 on the provided game.
- **Reproduction.** `python -m saha.pipeline --fresh --data <path>` rebuilds the file byte for
  byte. Two team members did so on their own machines.

## The approach

```
tracking ──► frame table ──► evidence builders ──────────────────────────► combiner ──► SAHA.csv
(positions)   (one row per    nearest   p_home where the ball is tracked     log-odds,
               video frame)   deadball  p_dead on every graded frame         integer,
                              visible   p_home where the ball is tracked     Viterbi
                              occluded  p_home where the ball is missing     over H/A/D
                              shape     p_home where the ball is missing
                              onball    p_home where the ball is tracked
                              hiddendead p_dead where WS2 calls dead and the ball is missing
```

| source | owner | what it says | how it is set |
|---|---|---|---|
| `nearest` | WS1 | the team of the player nearest a tracked ball: the official baseline, as evidence | scale 0.5, chosen on DFL |
| `deadball` | WS2 | dead or alive, from a classifier trained on SkillCorner and an event-gated decoder (stop events, confirmed restarts, kick-offs) | trained and tuned on SkillCorner; final (D022) |
| `visible` | WS3 | the team in possession while the ball is tracked: touches, with a hand-over only when the new team confirms it | reasoned values; two settings chosen on DFL |
| `occluded` | WS4 | a bridge across each gap in the ball's tracking, and Smart Tagging phases whose side comes from their official definitions | reasoned values; checked on DFL |
| `shape` | WS5 | the team in possession while the ball is missing, from the two teams' shape and motion | logistic model trained on DFL (D026) |
| `onball` | WS1 | the team in possession while the ball is tracked, from the ball and the players around it, with the teams' shape as context | gradient boosting trained on DFL (D027) |
| `hiddendead` | WS1 | a second opinion on WS2's dead calls where the ball is missing, from the players' movement and the edges of the gap; it replaces WS2's call there and can only soften it | gradient boosting trained on DFL (D028) |

**Why this design.**

- **The metric matches whole possessions.** A 1 s possession counts as much as a 40 s one, so
  flicker is fatal. Per-frame evidence plus switching costs (35 nats, about a second of strong
  evidence) changes possession only where the evidence for it adds up.
- **Five people could work in parallel.** Each package answers one question on its own frames,
  in one interface (`p_dead`, `p_home`, weight).
- **Integer arithmetic.** The combiner works in exact integers (fixed-point log-odds, 1e-10 nats
  per step), so the submission is byte-identical on every machine.

**Where the score is lost.** We measured it on the external matches by giving the system the
truth for one part at a time, and built a model for the largest gap each time. WS5 (Seyyid)
reached the first conclusion independently, with his own oracle runs: the team, not the dead
ball, was the bottleneck.

| that part perfect | before the shape model | before the on-ball model | before the dead-ball second opinion |
|---|---|---|---|
| nothing | 0.39 | 0.49 | 0.53 |
| the dead ball | 0.45 | 0.57 | 0.61 |
| the team where the ball is missing | **0.63** | 0.62 | 0.66 |
| the team where the ball is tracked | 0.66 | **0.74** | 0.74 |

spellF1 on DFL's raw labels; the first column is seed 0, the others 3 seeds. After the on-ball
model, two more models for the team where the ball is missing (a learned bridge, longer
windows) gained too little to pass their rules. Most of the dead-ball loss came from WS2's
false dead calls where the ball is missing (21.7% of the live frames there); correcting only
those, perfectly, would have given 0.57. The second opinion on them was adopted (D028).

## The possession rules we implemented

- **Loose balls and challenges (WS3).** Possession stays with the last team until the opponent
  confirms a touch. A confirmed touch lasts 6 frames (0.24 s) or more, or is followed by a second
  touch of the same team.
    - A deflection or a lost duel does not count.
    - The hand-over is placed at the new team's first touch.
    - After a restart or a gap in the tracking, the first touch starts a possession.
- **Who has a tracked ball (WS1).** Next to WS3's touch rules, a model trained on DFL reads the
  ball and the players around it: how near each team's players are, how fast they close in,
  where the ball is going, and the two teams' shape. Its inputs are chosen for the moments where
  the nearest player misleads: passes in flight, duels and loose balls.
- **Restarts and stoppages (WS2).** A stoppage opens near a stop event: the ball leaving the
  pitch, or the ball vanishing while everyone slows down. It closes at a confirmed restart:
    - a kick-off or a throw-in;
    - a ball played from a goal area or a corner;
    - a ball entering the pitch fast;
    - the players speeding up.

  Each half starts dead until its kick-off, which is found where the formation breaks inside the
  Smart Tagging kick-off window. A dead ball always splits two possessions.
- **A second opinion where the ball is missing (WS1).** Where WS2 calls the ball dead while it is
  missing, inside the played part of a half, a model trained on DFL weighs instead whether play
  goes on: whether the players keep running, how long the gap and the dead spell are, and where
  and how fast the ball was at the edges of the gap. It can turn WS2's call into live play, never
  the reverse, and every frame before a kick-off stays as WS2 decided.
- **The missing ball (WS4 and WS1).** Inside a gap in the ball's tracking:
    - WS4's bridge carries each side's team inward, and fades in long gaps.
    - Smart Tagging phases add weak evidence, with a side fixed by their official definitions:
      BUILD UP is the team with the ball, DEFENDING IN … the team without it.
    - The shape model reads which team is attacking from where both teams stand and how they
      move.
- **How long a possession must be (WS1).** There is no minimum length. A possession survives
  when its evidence pays for two switches of 35 nats each.

## Assets: used and dropped

| asset | use |
|---|---|
| tracking: player and ball positions | used by every package |
| tracking: the players' speed channel | used, in m/s (D012) |
| tracking: the ball's speed channel | **dropped**: it does not follow the ball (D015); every speed comes from positions |
| Smart Tagging | used: kick-off windows (WS2); phase sides, fixed from their definitions (WS4) |
| homography, broadcast feed | not used: the tracking is already in pitch coordinates |
| tactical video and audio | **dropped**: WS2's whistle detector works, but it stays off (D025); its benefit cannot be measured on external data, and it needs the 5 GB video |
| DFL/IDSSE open data | trains the shape, on-ball and dead-ball second-opinion models; benchmark for every package (WS5's harness) |
| SkillCorner open data | trains WS2's dead-ball classifier and sets its decoder |
| TU Dortmund whistle recordings | set the whistle threshold; dropped with the whistle |

**Tried and dropped:**

- **A team-shape model and Smart Tagging meanings fitted on this match.** Both were dropped
  under the organizers' rules (D017).
- **Holding a stoppage after every stop event.** It cost spellF1 on SkillCorner.
- **Lower switching costs and post-filters on short possessions.** No consistent gain on DFL.
- **Two settings.** `nearest` at 0.5 without WS3 v2.1 lost on held-out matches (adopted with
  v2.1). A gradient-boosting shape model was within noise of the logistic one, and the logistic
  one stayed.
- **A logistic on-ball model.** Held out, it got the team right on 0.887 of the frames with the
  ball, against 0.908 for gradient boosting, which was kept.
- **Two more models for the team where the ball is missing.** A learned bridge (the formation
  plus the edges of the gap) and the shape model with 10 s and 30 s windows gained +0.003 and
  −0.002 on DFL and failed their rules: the formation alone tells the team on about 88% of those
  frames, and the rest needs the ball.

## Missing and noisy tracking

- **The ball is missing on 41% of graded frames.** It goes missing in 150 stretches with a median
  of 10.7 s, a 90th percentile of 33.8 s and a maximum of 221 s (docs/EDA.md). The tracker
  bridges short gaps itself, so the gaps that remain are long.
- **Players are tracked well but not completely.** About 20.5 of 22 are tracked per graded
  frame. The 46 track slots carry no identities, so no feature depends on a player's identity.
- **How each package handles the missing ball:**
    - WS2's classifier was trained with WS5's hidden-ball stretches added to SkillCorner, so a
      missing ball alone is read as a stoppage less often. On DFL it still happens on 21% of the
      live hidden-ball frames (Limitations).
    - WS4's bridge and the shape model cover the missing stretches.
    - The shape model is trained with players removed as well. On DFL with players removed to our
      level, it keeps 0.876 accuracy where the ball is hidden, against 0.882 with every player
      present.
    - The on-ball model keeps 0.897 accuracy where the ball is tracked with players removed to
      our level, against 0.908 with every player present. Training it with players removed
      gained too little to pass its rule.

## Training data

None of it comes from the provided game.

| dataset | source and licence | labels, and who made them | amount | used for |
|---|---|---|---|---|
| DFL/IDSSE open data | Bassek, Weber, Rein and Memmert (2025), *Scientific Data*; CC BY 4.0 | DFL's official match data (Sportec Solutions): ball possession and ball status on every frame | 7 matches, Bundesliga and 2. Bundesliga 2022/23, 25 fps | training the shape and on-ball models (110,571 frames each) and the dead-ball second opinion (frames where WS2 calls dead with the ball hidden, at three degradation seeds); benchmark for every package; WS2's pace constants (checked) |
| SkillCorner open data | github.com/SkillCorner/opendata; MIT, © SkillCorner | SkillCorner's phases of play | 20 A-League 2024/25 matches, 10 fps | training WS2's dead-ball classifier; its decoder settings |
| TU Dortmund Whistle Detection Dataset | Kleingarn and Brämer (2024), TUDOdata; CC BY-SA 4.0 | its authors: whistle blasts in RoboCup recordings | test split, 249 blasts | one number, the whistle threshold; not used by the submission (whistle off) |

**DFL's possession labels change on every touch.** CHALLENGE.md counts only controlled
possession, so we also scored everything, and trained the two team models, on a stricter
reading: a possession under 2 s between two possessions of the other team is merged into them.
Choices are reported on both readings.

## Results

**External benchmark.** WS5's seven DFL matches, degraded like our match, scored with the
official scorer. "Raw" uses DFL's labels; "controlled" uses the stricter reading.

| version | raw | controlled |
|---|---|---|
| `nearest` + WS2 | 0.319 | |
| + WS3 v2 | 0.356 | 0.399 |
| + WS3 v2.1 and `nearest` at 0.5 | 0.387 | 0.420 |
| + WS4 v2: the version frozen on 12 November | 0.394 | 0.427 |
| + shape | 0.491 | 0.531 |
| + on-ball | 0.531 | 0.587 |
| **+ dead-ball second opinion: the submitted version** | **0.548** | **0.609** |

- The rows are seed 0.
- **The three learned models were evaluated leaving one match out.** They were fitted on the six
  other matches every time.
- **Over three degradation seeds:** the frozen version scores 0.398 raw and 0.433 controlled;
  with the shape model, 0.490 and 0.533, ahead on 21 of 21 runs; with the on-ball model too,
  0.531 and 0.587, ahead of the shape version on 21 of 21 runs; with the dead-ball second
  opinion too, 0.549 and 0.609, ahead of the on-ball version on 21 of 21 runs.
- **Every step was chosen with a rule fixed before its runs,** and is written up in an
  experiment record.

**WS2 on held-out SkillCorner matches:** spellF1 0.624 and dead-ball F1 0.899 as delivered;
0.583 and 0.835 under WS5's hiding.

**Checks on our match** (they chose nothing; identical on two machines): 361 possessions, with a
mean of 9.7 s and a median of 5.1 s; D on 44.8% of graded frames; home share 59.7%. The second
opinion covers 55,300 of WS2's 76,664 dead calls and finds 9.7% of them rather alive (about a
quarter on DFL). On the public 5 minutes, spellF1 is 0.659 [0.49–0.81] and macroF1 0.895, with
17 predicted possessions against 17 true ones; the frozen version had 0.492. The public window
has too few possessions to say more.

## Limitations

- **Two foreign leagues.** Every learned part comes from German and Australian matches, and
  none has seen Barcelona's or Athletic's style. The external benchmark is our best estimate,
  not a promise.
- **Labels that differ from the reference.** DFL's possession changes on every touch. Our
  stricter reading is a proxy, and the organizers' own definition is known only from 5 public
  minutes.
- **Untuned switching costs.** They are reasoned placeholders. Their best value depends on which
  label reading is right (35 on raw labels, 45–60 on controlled ones).
- **Unvalidated Smart Tagging evidence.** No external dataset has Smart Tagging, so WS4's tag
  evidence is weak by design and was never validated.
- **Dead ball where the ball is missing.** On DFL, WS2 alone calls 21.6% of the live frames dead
  where the ball is missing; with the second opinion, 5.6% (frame by frame, before decoding).
  The test hides the ball at random. In our match it may vanish more often when it goes out of
  play, where the second opinion may be too ready to call play live.
- **Missing players.** At our tracking level, the shape model (trained with players removed too)
  still loses about half a point of accuracy, and the on-ball model about one point. Both lose
  more if our tracking misses more players, or players nearer the ball, than we simulated.
  Both need the direction of attack, which is found per half; a half without a clear margin
  gets no opinion from either.
- **No abstention.** We never answer X inside a possession, although an X there would cost
  nothing.
- **With more time, we would:**
    - build our own ball detector on the tactical video (41% of the ball is missing);
    - relabel the external data to the reference definition, and tune the costs on it;
    - try a sequence model of team shape;
    - train the dead-ball classifier on 25 fps full-pitch tracking.

## Reproduction

```bash
python -m saha.pipeline --fresh --data <path to the data bundle>
```

- **Environment:** Python 3.11, with the pins in `requirements.txt`, `requirements-starter.txt`
  and `requirements-saha.txt` (TEAM.md, Setup).
- **The submitted file:** SHA256
  `af770330dbc3e5c414f8618192dcc683e79ff4092692952957ba59b6fbd0d426` (FINAL_CHOICE.md),
  reproduced by two team members on their own machines, with the same seven evidence files.
- **The combiner** works in integers and every evidence file is quantised to 1e-4, so the file
  does not depend on the machine.
- **The models' arithmetic** (matrix products, `exp`, tree thresholds) can differ in the last
  bit between processors. Such a difference reaches the file only if a value falls on a 1e-4
  rounding boundary, and the second opinion's inputs are rounded to float32 first. Two
  machines reproducing the same SHA256 is the check.
- **The shape model's coefficients** are in `saha/shape/model.json`. `python -m saha.shape.train`
  rebuilds them from the DFL matches: byte for byte with four threads, as they were trained, and
  within 0.022 in P(home) with another thread count.
- **The on-ball and second-opinion models** are pickles with their manifests in
  `saha/onball/models/` and `saha/hiddendead/models/`, trained on one thread with a fixed seed.
  `python -m saha.onball.train` and `python -m saha.hiddendead.train` rebuild them from the DFL
  matches byte for byte.

## How we kept to the rules

- **The strict reading.** We followed the organizers' clarification of 5 November: training,
  fitting, calibration and tuning on external data only (DECISIONS D016, D017).
    - Nothing was fitted on the provided game, its tracking and tags included. That rules out
      self-supervised fitting and values estimated from it on each run.
    - RULES.md allows "learning without human labels" on the provided game. We kept the stricter
      reading anyway, because the clarification came later and was stricter (D024).
- **No hand labels and no hand corrections.** The only frame numbers in the code are the
  published half boundaries.
- **The public 5 minutes checked the pipeline and chose nothing.** The packages' values come from
  football reasoning or from comparisons on external data. The choices made after 11 November
  (`nearest`'s scale, the shape model and its training, the on-ball model, the dead-ball second
  opinion) were made with a rule written down before their runs; so were the two models that
  were tested and rejected.
- **The match data never left team members' machines.** Not in git, cloud storage or chat;
  a git guard enforces it, and `main`'s history passes it.
- **AI assistants (Claude Code)** wrote code and saw aggregate output only: counts, percentages
  and scores. They never saw data or frame-level values (D008, D017).

## Appendices

Each package's section, as its owner wrote it.
