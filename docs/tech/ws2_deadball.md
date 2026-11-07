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
| tactical video, audio | off by default | the referee's whistle (`--whistle`; the whistle, below) |
| homography, broadcast feed | no | the tracking is already in pitch coordinates |
| SkillCorner open data (external) | yes | trains the classifier and sets the decoder's dead-ball settings |
| IDSSE / DFL open data (external) | checks only | the hand-set player-pace constants |
| TU Dortmund Whistle Detection Dataset (external) | yes | the whistle detector's threshold |
| referee-whistle makers' specifications | yes | the whistle band |

**SkillCorner open data**: 20 A-League 2024/25 matches of broadcast tracking at 10 fps, MIT
licence, © SkillCorner, github.com/SkillCorner/opendata (commit `4340d27`). Downloaded by
`saha/deadball/skillcorner.py`; each tracking file is checked against the SHA256 that its Git LFS
pointer publishes.

**IDSSE**: 7 Bundesliga / 2. Bundesliga 2022/23 matches, CC BY 4.0, © DFL / Sportec Solutions
(Bassek et al., *Scientific Data*, 2025), as in docs/tech/ws5_external.md. That provider freezes
the ball whenever play is dead: its position stops changing on every dead frame (a median
0.48 m past the line at ball-outs), and on under 1% of live ones. That is a cue no real tracker
gives. The prototype's own pipeline, with the ball always visible, scored 0.987 dead-ball F1 there
for that reason. Under WS5's degradation it is worth little to this package: removing every
frozen position changes dead-ball F1 by 0.005 (results, below). WS2 used IDSSE for the
player-pace constants and scores this package with WS5's harness.

**Whistle Detection Dataset**: Kleingarn, D. and Brämer, D. (2024), TUDOdata,
doi:10.17877/tudodata-2024-m0f6bmi1, CC BY-SA 4.0 (related paper: *Neural Network and Prior
Knowledge Ensemble for Whistle Recognition*, RoboCup 2023, LNCS 14140). Referee whistles recorded
by NAO robots at RoboCup events and labelled by its authors: we use its test split (147,440
spectra of 1024-sample windows at 44.1 kHz, 249 whistle blasts, 0.45 h without whistles) to set
one number, the detection threshold. No part of the dataset is redistributed.

## Training data

No data of this match trains or tunes anything.

- **Labels**: SkillCorner's own phases of play. They are defined only while the ball is in play,
  so a frame inside a phase is alive and a frame between phases is dead. Frames with fewer than
  ten tracked players are never used: they are the broadcast's replays and close-ups (56% of dead
  frames), and their absence would give the dead ball away.
- **Frames**: every 5th tracked frame of the 20 matches (neighbours are near copies), resampled
  from 10 to 25 fps so that every frame-based constant keeps its meaning. Every match is used
  twice: as delivered, and with the ball hidden the way WS5's degradation hides it
  (`saha.external.degrade.hide_mask`, its default settings, seed 0: 41% of frames, in long
  stretches placed at random). 878,894 frames in all.
- **Why twice.** On SkillCorner the ball is missing on 5.7% of live frames and 55% of dead ones.
  A classifier trained on the matches as delivered learns that a ball missing for long means a
  stoppage, and on WS5's DFL matches it called 92% of live frames dead under a hidden stretch
  (results, below). Our match misses the ball on 41% of graded frames, against SkillCorner's
  20%, so live play without a ball is likely there too. Trained on both copies, the classifier
  has to judge the players when the ball is gone.
- **Features** (79): 24 that describe a few seconds around each frame, and 55 that describe the
  change across it (eleven signals averaged over the second before, the second after and the two
  seconds after that, with the after-minus-before differences). A stop or a restart is a change;
  centred averages alone smear it over several seconds.
- **Classifier**: `HistGradientBoostingClassifier` (300 iterations, learning rate 0.05, 31
  leaves, at least 200 frames per leaf, L2 1.0), one per seed (0, 1, 2), logits averaged, the
  training prior removed so that the output acts as a likelihood ratio.

The model is committed as `saha/deadball/models/dead_skillcorner.pkl` (3.6 MB), with its manifest
`dead_skillcorner.pkl.json` (DECISIONS D020): the training script, the 20 matches and the
repository commit, the hiding, the licence, the seeds, the library versions and the SHA256,
`e25874c8…`. `python -m saha.deadball.train` rebuilds it from the external data. Two things keep
the file identical between machines. The training runs on one thread: scikit-learn stores the
thread count in the pickle, though the trees came out the same on 1, 4 and all cores. And the
numpy version must be the pinned one: numpy 2.5.4 computes a percentile of the players' speeds
one float32 step away from the pinned 2.3.5, and that alone changes most of the trees.

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
  choice. A simulation on the external matches later found that even the hold after whistles
  costs spellF1 (results, below). It only matters with `--whistle`, which stays off.
- **Evidence weight**: the classifier's log-odds are divided by 204/32 frames. 204 is the median
  residual correlation time of its logit on held-out SkillCorner matches, and 32 the weight with
  the best held-out spellF1 among 8–256. Nothing is estimated from this match.
- **Player-pace constants** (1.3 and 1.6 m/s for a stoppage pace and a match in motion, 1.5 m/s
  at a whistle) are football reasoning. Checked on IDSSE: the pace that best separates dead from
  alive there is 1.55 m/s, and 83% of stoppages follow a match moving above 1.5 m/s.

### The whistle (`audio.py`, off by default)

Nothing in the detector comes from this match's audio, and it was never checked against it.

- **Score.** A whistle is one steady tone, while a crowd spreads its sound over every frequency.
  So a frame's score is how far the strongest frequency between 3 and 6 kHz stands above that
  band's median level, in dB. It needs no assumption about which whistle the referee used.
- **Band, 3–6 kHz.** It holds the tones that whistle makers publish for referee whistles
  (Dolfin PRO 3.67 and 4.15 kHz, Acme Tornado 3.74 and 4.10 kHz, Fox 40 about 3.8 kHz on
  average), and the whistles of the external recordings below, which peak at 4.7–5.3 kHz. A
  score against side bands (1.5–2.7 and 6.5–8 kHz) did slightly better on those recordings, but
  its upper side band holds the second harmonic of a 3.7–4.2 kHz football whistle, so it would
  mark down exactly the whistles we need. Those recordings cannot show that, because their
  whistles' harmonics lie near 10 kHz.
- **Analysis.** 44.1 kHz audio, one 1024-sample Hamming window per video frame, centred on it:
  the analysis of the external recordings, so that a threshold chosen on them means the same
  here.
- **Threshold, 38 dB.** It is chosen on TU Dortmund's labelled test recordings, read the way
  `audio.py` reads ours (one window per 1/25 s, a blast is at least 3 frames over the threshold,
  gaps of up to 3 frames bridged). The rule was fixed before the numbers were seen: the lowest
  threshold with no false blast. The reason is the cost of each error: a false whistle in live
  play holds a stoppage until a confirmed restart, while a missed one leaves the classifier and
  the other events.

  | threshold | blasts found (of 249) | false blasts per hour of background |
  |---|---|---|
  | 30 dB | 86.3% | 30.9 |
  | 34 dB | 83.9% | 6.6 |
  | 36 dB | 81.9% | 4.4 |
  | **38 dB** | **73.1%** | **0** (95% upper bound 6.6) |
  | 40 dB | 53.4% | 2.2 |

  With 0.45 h of background, "no false blast" still allows up to 6.6 per hour.

The prototype's bands (3.70–3.96 and 4.18–4.40 kHz, both tones required) and its 20 dB threshold
were measured on this match's audio and are gone.

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
| the same without team evidence in the decoder | 0.626 | 0.901 | 0.833 |
| **the same, classifier trained with WS5's hidden ball as well (this package)** | **0.624** | **0.899** | **0.831** |
| perfect dead ball, same team side (the ceiling for dead-ball work) | 0.697 | 1.000 | 1.000 |

The same held-out matches with WS5's hiding applied (stop and restart events recomputed from the
hidden ball), the conditions WS5 uses to stand for ours:

| classifier trained on | spellF1 | dead-ball F1 | stoppage F1 | dead share (true 29%) | live frames called dead under a hidden stretch |
|---|---|---|---|---|---|
| the matches as delivered | 0.425 | 0.630 | 0.558 | 59.7% | 93.3% |
| the hidden copies only | 0.585 | 0.841 | 0.736 | 32.3% | 13.4% |
| **both (this package)** | **0.583** | **0.835** | **0.730** | **33.2%** | **15.1%** |

Training on both copies gains +0.158 spellF1 [+0.146, +0.171] under the hiding (better on all 20
matches) and costs −0.003 [−0.006, +0.001] on the matches as delivered. The hidden copies alone
cost −0.007 [−0.010, −0.003] there. The rule was fixed before the run: adopt if the gain under
the hiding is above zero, the matches as delivered lose no more than 0.005, and DFL's dead-ball
F1 rises. The both-copies classifier passed all three; the hidden-only one failed the second.
(`external/hidden_training.py`; the shipped file is retrained under the team's pins, a sibling
of the experiment's models, below.)

Removing the team evidence makes the dead-ball frames slightly better (F1 +0.003, stoppage F1
+0.007) and costs 0.006 spellF1 [−0.008, −0.003] (worse on 15 of 20 matches). It is removed
anyway: the team model was fitted on the match it labels. An externally trained team model is
the way to win the 0.006 back.

Where the rest goes, for this package's decoder (one kind of error put right from the truth at
a time; spellF1 0.626, ceiling 0.698): stoppage starts +0.031 (+0.013 where ours begin early,
+0.017 where they begin late), two stoppages merged into one +0.020, ends +0.010, missed
stoppages +0.007, false ones +0.006. Part of the start error is SkillCorner's convention: its
stoppages begin about 0.5 s before the ball crosses the line, while the competition's begin when
it crosses. So start offsets are not tuned on SkillCorner.

Also tested on the external matches and not adopted: learned stop and restart rates in four uses
(+0.002 at best, down to −0.063), short holds after every stop, a learned restart confirmation
(precision 0.94, recall 0.97, but no gain in the pipeline because the candidates themselves cover
only 66–89% of true restarts), a minimum stoppage length, and a rule turning short bursts between
stoppages back into play (half of its frames were dead).

Re-checked for this package's decoder (no team evidence), with the choice made inside the folds:

- **The decoder's settings**: the evidence weight (16–48) and the event-gating rates (near an
  event 0.5–2 s; with no event, a stop once per 60–300 s of play and a restart once per
  30–300 s of dead ball), 144 combinations. The best of them is +0.0004 spellF1 on all twenty
  matches (optimistic). Chosen on sixteen matches and scored on the other four, the choice gives
  −0.0013 [−0.0036, +0.0003]. The settings stay as they were (weight 32, 1 s, 120 s).
- **Splitting merged stoppages with the learned restart classifier**: inside a predicted
  stoppage, a confident restart candidate followed by a stop event within 5 or 10 s is held in
  play. Stoppage F1 rises (0.833 to 0.840), but spellF1 falls: −0.003 to −0.004 with 10 s
  (95% interval below zero at every threshold from 0.5 to 0.95), and −0.0001 to −0.0007 with
  5 s. The bursts' edges come from the candidate and the stop event, a second or more off, and
  the short possessions they make miss the true ones.
- **Set-piece tags as restart evidence.** Tag-like windows were built from SkillCorner's own
  set-piece events in four shapes (3 s before to 1 s after; 8/3 s; 15/8 s; 15 s with the set piece
  anywhere inside), with all tags or 20% dropped. Inside each window the decoder trusted every
  restart candidate, or the one the learned restart classifier rated highest, or that one with the
  tag's type, so that the existing set-piece rule could find the stoppage before it. All 24
  variants change spellF1 by −0.0014 to +0.0002. Confirming restarts is not what holds the
  decoder back, so our match's Smart Tagging set pieces are not used beyond the kick-offs.
- **What the whistle would be worth**, simulated (the external matches have no audio). A whistle
  goes at the start of every true stoppage that ends in a free kick or a kick-off (fouls,
  offsides, goals: 14% of stoppages) and goes through the pipeline like a real one.

  | whistles | with the hold after a whistle | without it |
  |---|---|---|
  | perfect | −0.004 | +0.004 [+0.002, +0.005] |
  | like `audio.py` (73% found, up to 0.5 s late) | −0.003 | +0.003 [+0.002, +0.004] |
  | the same, plus false blasts at 6.6 per hour | −0.014 | −0.005 |

  The hold costs because free kicks are the restarts the detector confirms least, so a held
  stoppage runs on into play. Even without it, the whistle is worth a few thousandths. It stays
  off.
- **A player-only model where the ball is missing.** It uses only the players' pace, the fastest
  players, team spread and their changes (20 columns), so it can also learn from WS5's DFL
  matches without DFL's frozen ball reaching it. Trained on SkillCorner and DFL, it separates dead
  from alive slightly better on DFL frames with the ball hidden (AUC 0.956 to 0.964).
  - **Replacing** the classifier's evidence there costs −0.011 spellF1 on SkillCorner as
    delivered and −0.002 under WS5's hiding.
  - **Averaging** the two gives −0.002 and +0.002 [−0.0004, +0.005].
  - On DFL, with one match held out at a time, the dead-ball F1 rises (0.917 to 0.926) but the
    fused spellF1 moves by +0.001 to +0.002.

  The rule fixed before the run (a gain under the hiding with its 95% interval above zero, at
  most −0.002 as delivered) is not met, so it is not used.

The scripts and their results are on branch `ws2/event-deadball`, in `external/`
(`decoder_sweep.py`, `learned_bursts.py`, `error_costs.py shipped`, `setpiece_windows.py`,
`whistle_sim.py`, `player_model.py` and `player_model_dfl.py`; the sweep's per-match results as a
summary, `skillcorner_decoder_sweep.txt`).

**External, DFL, through WS5's harness** (`python -m saha.external.score`, and
`python -m saha.deadball.external_check` for the dead ball alone): the seven DFL matches, 25 fps
and full pitch like ours, degraded by WS5 (the ball hidden on 41% of frames in long stretches
placed at random). Nothing was chosen on these numbers.

| classifier | spellF1, fused with `nearest` | `nearest` alone |
|---|---|---|
| trained on the matches as delivered | 0.311 (0.264–0.339) | 0.210 (0.172–0.239) |
| **trained on both copies (this package)** | **0.319** (0.274–0.367) | 0.210 |

| dead ball | precision | recall | F1 | stoppage F1 | dead, DFL | dead, ours | live called dead, ball hidden / in view |
|---|---|---|---|---|---|---|---|
| trained as delivered | 0.655 | 0.997 | 0.790 | 0.570 | 44.9% | 68.3% | 92.5% / 5.2% |
| trained as delivered, frozen ball removed | 0.649 | 0.995 | 0.785 | 0.566 | 44.9% | 68.7% | 92.3% / 6.7% |
| **trained on both copies** | **0.877** | **0.961** | **0.917** | **0.783** | 44.9% | **49.2%** | **20.3% / 4.5%** |
| trained on both copies, frozen ball removed | 0.882 | 0.929 | 0.905 | 0.781 | 44.9% | 47.4% | 17.7% / 4.9% |

Trained on the matches as delivered, the classifier missed almost no stoppage but called a third
of live play dead: 93% of the wrongly dead frames lay under the harness's hidden-ball stretches.
It had learned on SkillCorner that a ball unseen for a long time means a stoppage, which holds
there but not where the ball goes missing in play. Trained on both copies, the dead-ball F1 rises
from 0.790 to 0.917. DFL's frozen ball is now worth 0.012 of it.

**Our match, checks only** (no setting was chosen from these). The submission has the dead ball
on 48.3% of graded frames (50.4% with the classifier trained on the matches as delivered), 293
possessions (305), and 5 of 6 tagged kick-offs are confirmed by the formation.

## Limitations

- **No whistle by default.** It needs the 5.2 GB video and ffmpeg on every machine that
  reproduces the submission, which the team decides (D006), with a check that two machines find
  the same whistle times. Without it, fouls and offsides open a stoppage only through the
  classifier, the players slowing down or the ball vanishing.
- **The whistle detector's evidence is thin.** Its threshold comes from robot microphones in
  RoboCup halls, not a stadium broadcast, from 0.45 h of background and from whistles at a
  higher pitch than football's. It finds 73% of those blasts. A crowd whistling at the referee
  is a tone in the same band, and nothing has measured how often that passes 38 dB.
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
everything cached (SHA256 `3b74ae89…`).

- Rebuilding the classifier (`python -m saha.deadball.train`) downloads SkillCorner to
  `cache/external/skillcorner/` (1.8 GB) and converts it, then trains in about 7 minutes. Set
  `SAHA_SKILLCORNER_DIR` to use a copy already on disk.
- `SAHA_DEADBALL_MODEL` names another classifier file, or `none` for no classifier and no
  opinion, as the Phase 0 stub had. The pipeline tests run with `none`, because the synthetic
  match's four players look dead to a classifier that learned from twenty-two (it calls 96% of
  that match dead). `tests/test_deadball.py` tests the package with a stand-in classifier and
  checks that the committed one loads.
- `python -m saha.deadball.build --whistle` adds the whistle (needs ffmpeg and the video).
- `python -m saha.deadball.audio` prints the number of whistle blasts, a SHA256 of their frames
  and the ffmpeg version, for the two-machine check of item 8 (`--refresh` decodes the audio
  again). On the WS2 machine (ffmpeg 9.0.2), a cached and a fresh decode give the same SHA256.
