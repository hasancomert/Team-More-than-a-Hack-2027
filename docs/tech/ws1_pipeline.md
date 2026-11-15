# WS1: data layer, evidence interface, combiner and reproduction

Owner: Hasan (team leader). Code: `saha/common/`, `saha/combine/`, `saha/nearest/`,
`saha/shape/`, `saha/onball/`, `saha/hiddendead/`, `saha/pipeline.py`, `saha/tools/`,
`configs/`. Draft of 8 November, with the shape, on-ball and dead-ball second-opinion models
added on 13 November; the final numbers come from FINAL_CHOICE.md.

## Approach

Each frame raises two questions: is the ball dead, and, if it is alive, which team has it. Each
package answers one of them on its own frames, as evidence. WS1 reads the match for everyone,
fuses the evidence, and segments the match into possessions.

**The data layer** (`saha/common`) gives one row per video frame.

- **Alignment.** An ATD `frame_id` is video frame `frame_id − 1`. A Smart Tagging time is frame
  `round(seconds × 25)`. Homography rows are already video frames.
- **Parsing.** The parser follows the EPTS layout that the metadata declares. Fed into the
  official baseline's smoothing, it reproduces the four published baseline files with 0
  differing frames.
- **Speeds.** The players' speed channel is in m/s, as the displacements confirm (D012). The
  ball's channel has a value on only 7.8% of the frames where the ball is tracked, and does not
  follow the ball (D015). So every package takes the ball's speed from its positions
  (`saha.common.kinematics`).

**The evidence interface.** Every package writes, per frame:

- `p_dead`, the probability that the ball is dead;
- `p_home`, the probability that Barcelona has the ball, given that it is alive;
- `weight`, how far to trust that row; 0 means no opinion.

Values are quantised to 1e-4. Packages write only on their own frames (TEAM.md, boundaries).

**Fusion.** For each frame, the combiner adds every source's log-odds, multiplied by its weight
and its scale in `configs/pipeline.toml`: naive Bayes, with the weights as tempering exponents.
It works in exact integers (1e-10 nats per step, with logits tabulated in milli-nats), so the
result is the same on every machine.

**Segmentation.** A three-state Viterbi over H, A and D.

- The per-frame scores are the exact doubled log-probabilities of the three states, with the
  normaliser n = 2 log(2 cosh(home/2)) on H and A, read from a milli-nat table. Without it, a
  confident `p_home` also pushed the alive states against D (found by WS4).
- The prior P(dead) = 1/3 makes the three states tie on a frame without evidence.
- Changing state costs `cost_switch` (H ↔ A), `cost_to_dead` or `cost_from_dead`. A possession
  changes only where the evidence for the change, summed over the frames it covers, outweighs
  its cost.
- Ties are broken in a fixed order.

**Why this design.**

- The metric matches whole possessions and punishes flicker. A per-frame classifier flickers;
  switching costs turn evidence into few, long possessions.
- Evidence files let five people work in parallel and be combined without reconciling five
  pipelines.
- Exact integer arithmetic makes the submission byte-identical across machines.

**`nearest`** is the official nearest-player baseline, re-expressed as evidence: `p_home` 0.8
for the nearest player's team where the ball is tracked, and `p_dead` from whether a ball has been
seen within 2 s. It is the reference and the fallback.

**`shape`** (D026) says who has the ball while it is missing. It reads the two teams' shape and
motion, never the ball, and writes `p_home` on the graded frames without a tracked ball, next to
WS4. Seyyid (WS5) reached the same idea independently: his gradient-boosting model on 52 team
features raised DFL's spellF1 from 0.394 to 0.470. He then reviewed this logistic version and
reproduced its result with his own harness.

- **Why.** On WS5's DFL matches, the true team on the frames without the ball alone would raise
  the frozen version's spellF1 by 0.24. No other part of the system loses as much. WS4's bridge
  got the team right on about 65% of those frames.
- **Orientation.** In each half, take each team's player nearest the low-x goal, with the median
  over the half.
    - For the team that defends that goal, this is its goalkeeper or its last defender.
    - For the other team, it is a forward, who stays further out to be onside.
    - So the team with the lower median defends the low-x goal and attacks towards high x.
    - On WS5's DFL matches the two medians are 21–27 m apart. Below 10 m the direction is unclear,
      and that half gets no opinion.
- **Features.** For each team, in its own attacking frame:
    - the centroid, the deep and the advanced players (15th and 85th percentiles), the length and
      the width;
    - the share of players in the opponent's half;
    - the mean velocity towards the opponent's goal and the mean speed.

  Each is smoothed over 1 s and over 4 s, and its change over the next 2 s is added. A player
  velocity above 12 m/s is a tracking jump and is ignored.
- **Model.** A logistic regression on home's features minus away's, with no intercept and every
  example also added with the teams swapped. So it cannot learn a home or away side: swapping the
  teams turns p into 1 − p.
    - It is trained on all seven DFL matches, on alive frames, every 5th frame.
    - Each match is also added once with players removed in runs (10% of player-frames, 5 s
      runs on average). DFL tracks all 22 players; our tracking about 20.5. Held out at our
      level, this raised the accuracy where the ball is hidden from 0.856 to 0.876, and cost
      0.002 with every player present.
    - The labels are DFL's possession, with every possession under 2 s between two of the other
      team merged into them.
    - The 24 coefficients are in `saha/shape/model.json`.
- **Output.** The probability is clipped to [0.1, 0.9] at weight 1, and the config gives it
  scale 2.

**`onball`** (D027) says who has the ball while it is tracked. It reads the ball and the players
around it, with the two teams' shape as context, and writes `p_home` on the graded frames with a
tracked ball, next to WS3 and `nearest`.

- **Why.** With the shape model in, the true team on the frames with the ball alone would raise
  spellF1 on WS5's DFL matches from 0.49 to 0.74 (raw labels): the largest loss left. The
  nearest player is right on 78% of those frames, held out, and WS3 and `nearest` both start
  from it.
- **Features** (88), for each team in its own attacking frame (the orientation of `shape`):
    - the distance from the ball to its nearest and second-nearest player, capped at 30 m;
    - its players within 5 m and within 10 m of the ball;
    - the nearest player's speed towards the ball;
    - the ball's x, and the ball's velocity towards the team's goal.

  Each is averaged over a centred 0.5 s and a centred 2 s window, the second before and the
  second after; the averages skip frames without the ball. Added to them: the ball's speed,
  acceleration and position, raw and over 1 s, and `shape`'s 24 features, home minus away.
- **Model.** scikit-learn's histogram gradient boosting (300 trees of up to 31 leaves, learning
  rate 0.05, at least 200 frames per leaf). Every example is also added with the teams swapped,
  and the answer is the mean of the model's answers in both orders, so swapping the teams turns
  p into 1 − p.
    - It is trained on all seven DFL matches, on alive frames, every 5th frame, with the same
      labels as `shape`.
    - Training runs on one thread with a fixed seed and no early stopping. The pickle and its
      manifest are in `saha/onball/models/` (D020).
    - Held out, it gets the team right on 0.908 of the alive frames with the ball (seed 0), where
      the nearest player gets 0.780.
    - With 7% of player-frames removed (our level), it keeps 0.897. Training with missing
      players too would have gained only 0.005 there, below the 0.010 that the rule fixed
      before the run required, so it is trained on complete matches.
- **Output.** The probability is clipped to [0.1, 0.9] at weight 1, and the config gives it
  scale 2. The model is silent on the few frames at each edge of a tracking gap, where the ball's
  speed is undefined; WS3 and `nearest` still cover them.

**`hiddendead`** (D028) is a second opinion on WS2's dead calls where the ball is missing.

- **Why.** With the on-ball model in, WS2 calls 21.7% of the live frames dead where the ball is
  missing, against 4.6% where it is tracked (DFL, 7 matches × 3 seeds). Each false stoppage
  splits a possession; correcting only those calls, perfectly, would raise spellF1 from 0.531
  to 0.569 (raw).
- **Where.** Graded frames where the ball is missing, WS2 calls the ball dead, and the frame lies
  between WS2's first and last alive frame of its half; every frame before a kick-off stays as
  WS2 decided. There the pipeline drops WS2's `p_dead` (`overrides = "deadball"`,
  `saha.pipeline.apply_overrides`) and the decoder weighs this source's P(dead), clipped to
  [0.1, 0.9], at scale 2. It can soften WS2's dead calls, never add one.
- **Inputs** (27, none of them depends on which team is which): where the frame is in its gap
  and in WS2's dead spell; the ball's speed, position and nearest player at the edges of the
  gap; the players' mean speed, the slower team's speed, forward speed, the teams' length, width
  and advancement, over 1 s and 4 s, and the change of the mean speed. They are rounded to
  float32, as in training.
- **Model.** The on-ball model's gradient boosting, trained by `python -m saha.hiddendead.train`
  on the seven DFL matches degraded at seeds 100–102 (never the evaluation seeds), on the frames
  above, with DFL's ball status as the target. One thread, fixed seed; the pickle and its
  manifest are in `saha/hiddendead/models/`.
- **Held out** (7 matches × 3 seeds): spellF1 0.531 → 0.549 raw and 0.587 → 0.609
  controlled-2s, ahead on 21 of 21 runs on both. Of the live frames where the ball is missing,
  WS2 alone calls 21.6% dead; with the second opinion, 5.6%.

## Assets

| asset | use |
|---|---|
| ATD tracking | positions of the players and the ball; the players' speed channel (m/s) |
| the ATD ball speed channel | dropped: mostly empty, and it does not follow the ball (D015) |
| Smart Tagging, homography, video | not used by WS1 (see the package appendices) |

## Training data

WS1 trains three models, `shape`, `onball` and `hiddendead`, on external data only.

- **The shape model.** It is trained on WS5's seven DFL/IDSSE matches (Bassek, Weber, Rein and
  Memmert, 2025; CC BY 4.0), with DFL's own possession labels, made stricter as described above.
  `python -m saha.shape.train` rebuilds `saha/shape/model.json` from them. Nothing of our match is
  fitted.
- **The on-ball model.** The same matches and labels. `python -m saha.onball.train` rebuilds
  `saha/onball/models/onball.pkl` and its manifest, which records the data, the licence, the
  library versions and the file's SHA256.
- **The second opinion on dead calls.** The same matches, degraded by WS5's harness at seeds
  100–102, with DFL's ball status as the target. `python -m saha.hiddendead.train` rebuilds
  `saha/hiddendead/models/hiddendead.pkl` and its manifest.
- **Prior.** The prior comes from the model's definition.
- **Costs.** The switching costs are reasoned placeholders (35 nats). They are to be tuned on
  WS5's external matches once those mean what ours mean (D017).
- **The label-free self-check** (`python -m saha.tools.selfcheck`) hides ball touches and checks
  that the decoder recovers them. It is a check only: it never chose a value.

## Possession rules

- **Hand-overs.** The combiner implements none of its own: H and A change where the evidence
  outweighs `cost_switch`.
- **Dead ball.** D splits possessions, and WS2's decision is reproduced frame for frame (D022).
- **Short possessions.** They survive only where the evidence is strong enough to pay for two
  changes.
- **Outside the graded halves.** The answer there is X. Inside them, the decoder always answers
  H, A or D.

## Reproduction

```bash
python -m saha.pipeline --fresh --data <path>
```

- **Environment.** Python 3.11 with the pins in `requirements-saha.txt`.
- **Determinism.**
    - Evidence is quantised.
    - The combiner works in integers.
    - Inputs are read in sorted order.
    - Every seed is fixed.
- **Checked on two machines.** The SAHA.csv SHA256 matched on two members' machines on 6 and on
  7 November, for the frozen version on 12 and 13 November, and for the submitted version
  (`d276615`, `af770330…`) on Hasan's and Baki's machines on 14 November, with all seven
  evidence files the same.
- **CI.** CI runs the tests, the guard and the official evaluation on every pull request.

## How the team worked with the data

- **Data stays local.** The match data and anything derived from it frame by frame never
  entered git, cloud storage or chat. A git guard and CI enforce it.
- **The public 5 minutes** checked that the pipeline was correct and chose nothing.
- **AI assistants (Claude Code).** They wrote code and saw aggregate output only: counts,
  percentages, percentiles and scores. They never saw data or frame-level values (D008, D017).

## Limitations

- **Placeholder costs.** The switching costs stay reasoned placeholders until WS5's relabelled
  DFL set exists.
- **Independent sources.** The fusion assumes them, but `nearest` and WS3 read the same nearest
  player. So `nearest` runs at 0.5 next to WS3 v2.1, which writes weight 2. That choice was made
  on WS5's seven DFL matches, with a rule fixed before the runs, and it held on raw and on
  controlled labels (experiments/20261111_ws1_nearest-half-with-v21.md).
- **The shape model transfers from seven German matches.** It has never seen Barcelona's or
  Athletic's style. Missing players cost it accuracy: 0.876 at our tracking level, against 0.882
  with every player present (held out, after training with missing players).
- **The orientation rule assumes goalkeepers are tracked.** A half without a clear 10 m margin
  gets no opinion from `shape`, rather than a wrong one.
- **`onball` reads the same frames as WS3 and `nearest`.** The three are not independent. Its
  scale was fixed by reasoning, as for WS3 and `shape`, and checked on DFL with a rule fixed
  before the runs (experiments/20261113_ws1_onball-model.md). It has seen only DFL's tracking,
  where the ball is tracked on every frame; ours may be noisier.
- **No abstention.** The decoder never answers X inside a possession, where X would be free.
- **WS2's decision is final, with one exception.** It overrides every other `p_dead` (D022),
  except where `hiddendead` gives a second opinion on a dead call with the ball missing
  (D028). That model has seen DFL's tracking, where the ball goes missing at random; in our
  match a missing ball may mean a stoppage more often.
