# WS2: dead ball

## Approach

`saha/deadball` decides, on every graded frame, whether the ball is dead, and hands that decision
to the combiner as `p_dead` (0.99 or 0.01, weight 20). At that weight the combiner keeps every
stoppage and every stretch of play exactly as WS2 decided them at its current costs: on our
match, 0 of 158,857 graded frames differ between `deadball`'s decision and the `D` frames of
`submissions/SAHA.csv`. WS2 writes no `p_home`.

```bash
python -m saha.deadball.build      # the evidence file (~15 s)
python -m saha.deadball.train      # only to rebuild the committed classifier from SkillCorner
```

The decision comes from three parts:

- **A dead-ball classifier** (`dead.py`) judges how each frame looks: the players' pace, the
  ball's visibility, position, speed and distance from the lines, and how far each side stands
  from it. It is trained on twenty labelled external matches (training data, below).
- **Stop and restart events** (`events.py`) mark the moments a stoppage opens and closes. They
  come from the laws of the game and from physics, not from fitting: the ball crossing a line,
  the ball vanishing while everyone slows down, the ball struck after lying still or after being
  out of sight, a kick-off, and (when enabled) a whistle blown while the match is moving.
- **An event-gated decoder** (`segment.py`, `decode.py`): an HMM whose transitions into dead ball
  are likely only near a stop event, and out of it only near a confirmed restart. The memory
  that "a stoppage opened and has not been closed" lives here, not in the classifier.

Why a decoder inside the package: frame by frame, set-piece preparation looks like slow
build-up, and the first seconds after a foul look like play. What separates them is what happened
before and what happens after, so the decision needs the event structure. STATUS.md asks for the
decoder ideas to go into WS1's combiner instead; until WS1 decides that, they stay here and the
combiner receives the result.

The prototype on branch `ws2/event-deadball` also labelled H/A with a team-shape model fitted on
this match. That model is not ported: the organizers' rules allow no fitting on this match, and
WS2 gives no team. Without any team evidence the decoder's H and A states are the same alive
state twice.

## Assets

| asset | used | how |
|---|---|---|
| tactical tracking (frame table) | yes | every feature and event |
| Smart Tagging | yes | the kick-off windows only (where to look for the kick-off) |
| tactical video, audio | off by default | the referee's whistle (`--whistle`; limitations) |
| homography, broadcast feed | no | the tracking is already in pitch coordinates |
| SkillCorner open data (external) | yes | trains the classifier and sets the decoder's dead-ball settings |
| IDSSE / DFL open data (external) | checks only | the hand-set player-pace constants |

**SkillCorner open data**: 20 A-League 2024/25 matches of broadcast tracking at 10 fps, MIT
licence, © SkillCorner, github.com/SkillCorner/opendata (commit `4340d27`). Downloaded by
`saha/deadball/skillcorner.py`; each tracking file is checked against the SHA256 that its Git LFS
pointer publishes.

**IDSSE**: 7 Bundesliga / 2. Bundesliga 2022/23 matches, CC BY 4.0, © DFL / Sportec Solutions
(Bassek et al., *Scientific Data*, 2025), as in docs/tech/ws5_external.md. That provider freezes
the ball whenever play is dead (on every dead frame, a median 0.48 m past the line at ball-outs),
so a dead-ball method that looks at the ball scores near-perfectly there for the wrong reason.
WS2 uses IDSSE only for the players.

## Training data

No data of this match trains or tunes anything.

- **Labels**: SkillCorner's own phases of play. They are defined only while the ball is in play,
  so a frame inside a phase is alive and a frame between phases is dead. Frames with fewer than
  ten tracked players are never used: they are the broadcast's replays and close-ups (56% of dead
  frames), and their absence would give the dead ball away.
- **Frames**: every 5th tracked frame of the 20 matches (neighbours are near copies), 439,447 in
  all, resampled from 10 to 25 fps so that every frame-based constant keeps its meaning.
- **Features** (79): 24 that describe a few seconds around each frame, and 55 that describe the
  change across it (eleven signals averaged over the second before, the second after and the two
  seconds after that, with the after-minus-before differences). A stop or a restart is a change;
  centred averages alone smear it over several seconds.
- **Classifier**: `HistGradientBoostingClassifier` (300 iterations, learning rate 0.05, 31
  leaves, at least 200 frames per leaf, L2 1.0), one per seed (0, 1, 2), logits averaged, the
  training prior removed so that the output acts as a likelihood ratio.

The model is committed as `saha/deadball/models/dead_skillcorner.pkl` (3.6 MB), with its manifest
`dead_skillcorner.pkl.json` (DECISIONS D020): the training script, the 20 matches and the
repository commit, the licence, the seeds, the library versions and the SHA256, `0506d791…`.
`python -m saha.deadball.train` rebuilds it from the external data, and the same SHA256 came out
on every rebuild, including one from a freshly converted copy of the data. Training runs on one
thread. The trees are the same on 1, 4 and 12 threads, but scikit-learn stores the thread count
in the pickle, so pinning it keeps the file identical.

## Possession rules (dead ball) and how each setting was chosen

- **A stoppage opens near a stop event and closes near a confirmed restart.** Near an event, the
  decoder switches with probability 1/25 per frame (about once a second). Away from any event,
  the rate is once per two minutes of play, so a stoppage with an undetected cause still gets
  through when the classifier is sure. The windows are: whistle −1 s to +2 s; ball exit −0.5 s
  to +1 s; ball vanishing −3 s to +0.5 s. They come from football reasoning about how far each
  event lies from the moment play stops.
- **Restarts count only when confirmed**, because players touch the ball while a set piece is
  prepared. A candidate counts if any one of these holds:
  - it is a kick-off or a touchline throw-in;
  - it is played from a ball that sat in a goal area or at a corner;
  - it comes into the pitch at 8 m/s or more;
  - the players speed up afterwards: their mean speed 1–4 s after is at least 1.2 times the
    3 s before.

  On IDSSE, 81–100% of real restarts pass the speed-up test (96% of free kicks), against 8% of
  random moments inside stoppages.
- **A set-piece restart implies the stoppage before it.** From each confirmed goal kick, corner
  or throw-in, the decoder walks back while the ball stays near the restart spot. It stops at the
  last fast ball, which is the shot or pass that ended play.
- **Kick-offs are enforced.** Before a kick-off every player stands in his own half. Inside each
  Smart Tagging kick-off window, the kick-off is where that formation breaks. The set-up before it
  is dead ball, and so is everything from the start of a half up to its first kick-off.
- **After a whistle, the stoppage holds until a confirmed restart** (60 s at most). This follows
  the laws: play stopped by the referee resumes only with a restart. Other stops are not held.
  On the 20 SkillCorner matches, a hold after a ball exit, a vanishing ball or an implied stop
  cost more than it saved, on every match (dead-ball F1 −0.037 [−0.046, −0.029]). The prototype
  first chose holding after every stop by comparing versions on the public window and on restart
  times read off the video. Neither is allowed any more, and the external matches reversed that
  choice.
- **Evidence weight**: the classifier's log-odds are divided by 204/32 frames. 204 is the median
  residual correlation time of its logit on held-out SkillCorner matches, and 32 the weight with
  the best held-out spellF1 among 8–256. Nothing is estimated from this match.
- **Player-pace constants** (1.3 and 1.6 m/s for a stoppage pace and a match in motion, 1.5 m/s
  at a whistle) are football reasoning. Checked on IDSSE: the pace that best separates dead from
  alive there is 1.55 m/s, and 83% of stoppages follow a match moving above 1.5 m/s.

## Results

**External matches, held out.** The 20 SkillCorner matches were split into five folds of four,
and every number is on matches the classifier never saw. spellF1 is the official
`possession_metrics.spell_scores` against H/A/D truth from SkillCorner's phases (the team in
possession inside each phase, dead ball between phases). There is no whistle (no audio). The
team side is the prototype's own throughout, so the differences come from the dead ball.

| | spellF1 | dead-ball F1 | stoppage F1 |
|---|---|---|---|
| prototype's first design (classifier fitted on the match itself, hold after every stop) | — | 0.792 | 0.671 |
| the same without any hold | 0.571 | 0.833 | 0.696 |
| SkillCorner classifier, weight 32, hold after whistles only | 0.632 | 0.899 | 0.826 |
| **the same without team evidence in the decoder (this package)** | **0.626** | **0.901** | **0.833** |
| perfect dead ball, same team side (the ceiling for dead-ball work) | 0.697 | 1.000 | 1.000 |

Removing the team evidence makes the dead-ball frames slightly better (F1 +0.003, stoppage F1
+0.007) and costs 0.006 spellF1 [−0.008, −0.003] (worse on 15 of 20 matches). It is removed
anyway: the team model was fitted on the match it labels. An externally trained team model is
the way to win the 0.006 back.

Where the rest goes (one kind of error put right from the truth at a time): stoppage starts
+0.028 spellF1, two stoppages merged into one +0.022, ends +0.010, missed stoppages +0.008,
false ones +0.007.

Also tested on the external matches and not adopted: learned stop and restart rates in four uses
(+0.002 at best, down to −0.063), short holds after every stop, a learned restart confirmation
(precision 0.94, recall 0.97, but no gain in the pipeline because the candidates themselves cover
only 66–89% of true restarts), a minimum stoppage length, and a rule turning short bursts between
stoppages back into play (half of its frames were dead). The scripts and their results are on
branch `ws2/event-deadball`, in `external/`.

**Our match, checks only** (no setting was chosen from these). The submission has the dead ball
on 50.4% of graded frames, and 5 of 6 tagged kick-offs are confirmed by the formation.

## Limitations

- **No whistle by default.** Its frequency bands were measured on this match's audio. That is
  calibrating on the provided video, which the rules do not allow. It also needs the 5.2 GB video
  and ffmpeg on every machine that reproduces the submission (D006). A band from published
  referee-whistle measurements would fix the first problem; the team decides the second. Fouls
  and offsides then open a stoppage only through the classifier, the players slowing down or the
  ball vanishing.
- **Restart detection.** A free kick played short and slowly does not make the players speed up,
  and tracking cannot tell it from the touches while the set piece is prepared. Only 66–89% of
  true restarts produce a candidate on the external matches.
- **Stoppage starts and short play between stoppages** are the largest remaining losses.
- **Another league, another camera.** The classifier learned from broadcast tracking of A-League
  matches; ours is a full-pitch tactical recording. Nothing measures that transfer directly.
- **Kick-offs are found about 0.8 s late** (the formation breaks once players cross the halfway
  line). The same lag appears on IDSSE, so it belongs to the method, not to this match.

## Reproduction

Python 3.11 with `requirements-saha.txt` (scikit-learn 1.9.1 and its dependencies are pinned
there). The build reads the committed classifier and takes about 15 s; no external data is
needed. From an empty cache, the whole pipeline gives the same `SAHA.csv` as a run with
everything cached (SHA256 `081f57c1…`).

- Rebuilding the classifier (`python -m saha.deadball.train`) downloads SkillCorner to
  `cache/external/skillcorner/` (1.8 GB) and converts it, then trains in about 2 minutes. Set
  `SAHA_SKILLCORNER_DIR` to use a copy already on disk.
- `SAHA_DEADBALL_MODEL` names another classifier file, or `none` for no classifier and no
  opinion, as the Phase 0 stub had. The pipeline tests run with `none`, because the synthetic
  match's four players look dead to a classifier that learned from twenty-two (it calls 96% of
  that match dead). `tests/test_deadball.py` tests the package with a stand-in classifier and
  checks that the committed one loads.
- `python -m saha.deadball.build --whistle` adds the whistle (needs ffmpeg and the video).
