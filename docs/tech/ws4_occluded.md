# WS4: possession while the ball is missing

Owner: Baki. Code: `saha/occluded/`. Evidence: `cache/evidence/occluded.parquet`.

The ball is not tracked on 41% of graded frames, mostly in long stretches (median 10.7 s,
docs/EDA.md). WS4 gives `p_home` on exactly those frames: graded frames where `ball_visible` is
false. It never writes `p_dead` (WS2) and never writes on a frame with a tracked ball (WS3).

## Approach (v1)

Two sources, fused in log-odds per frame.

**1. The bridge over each gap.** Every missing-ball stretch has a last sighting before it and a
first sighting after it.

- *The team at the ball.* A tracked frame has a team at the ball when a player is within 1.5 m
  of a ball moving under 5 m/s. The ball's speed comes from its positions (D015). This is the
  same "touch" as WS1's `saha.tools.selfcheck`, so both read the match the same way.
- *Each side of the gap.* The touches within 2 s of the gap's edge vote, and a touch 0.5 s
  further away counts half as much. The side's `P(home)` is
  `0.5 + 0.4 · (home − away) / (home + away + 1)`. One touch right at the edge gives about 0.7,
  many give up to 0.9, and no touch gives 0.5 (no opinion).
- *Inside the gap.* Possession is a two-state Markov chain that changes team at a rate of one
  change per 12 s. The forward–backward posterior between the two sides gives `P(home)` on every
  missing frame:
    - Near a side, the result follows that side.
    - In the middle of a long gap, it fades to 0.5. Beyond about 30 s from both sides little
      is left.
    - When the two sides disagree, it crosses 0.5 once, where a hand-over is most likely. It
      never flickers.
- A gap never reaches across a half. The side beyond the half's edge has no opinion.

**2. Smart Tagging phases (weak evidence).** Every tag with a team says something about who has
the ball, except SET PIECES: a restart's team is WS2's question (TEAM.md, boundaries).

- We do not assume what the tag's team means. In BUILD UP or PROGRESSION it is presumably the
  team with the ball; in DEFENDING IN ... it is presumably the team without it. The algorithm
  measures it instead.
- For each code it takes the instances whose touches have a majority team, and counts how often
  that team is the tagged one. Five pseudo-instances on each side shrink the share towards 0.5.
  The result is `P(tagged team has the ball)`, and it is below 0.5 for a code that names the
  defending side.
- No ground truth is used for this. The share is measured on the tracked frames of this match by
  the algorithm, on every run. Nobody chooses it by hand.
- On a missing frame, the overlapping tags are averaged in log-odds and added to the bridge at
  half strength.

The result is clipped to [0.1, 0.9], with weight 1.

## Why this, and not something else

- **Interpolating the ball buys little.** The tracker bridges short occlusions itself: there are
  no gaps between 1 and 3 s, and 85% of the missing frames are in gaps of 10 s or more
  (docs/EDA.md).
- **The forward–backward chain** gives the switch position that minimises the expected error
  when nothing else is known. It is calibrated by construction under its own assumptions. It is
  also honest about long gaps: they fade to "no opinion", so they do not force a team.
- **Measuring the tag's meaning** avoids a hand-written polarity table that could be wrong for
  some code. It is also not a constant read off this match: the algorithm computes it on every run.
- **The 0.9 clip.** The combiner adds `|home|` to the alive states (`saha/combine/viterbi.py`),
  so a very confident `p_home` would also push towards "alive" inside a stoppage. Capping our
  log-odds at about 2.2 nats per frame keeps WS2's `p_dead` in charge of that question.

## Parameters and where they come from

| parameter | value | reason |
|---|---|---|
| `CONTROL_M`, `CONTROL_SPEED` | 1.5 m, 5 m/s | the touch of `saha.tools.selfcheck`: at a player's feet, not passing |
| `ANCHOR_HORIZON` | 2 s | the last touches before the ball disappears; older ones belong to another phase |
| `ANCHOR_HALF_LIFE` | 0.5 s | a hand-over just before the gap must win over a long spell before it |
| `ANCHOR_MAX` | 0.9 | the touch team is wrong sometimes (a contested ball, a defender passing close by) |
| `MEAN_HOLD_S` | 12 s | team possessions last 10–20 s in top-flight football; short contested ones count here too |
| `TAG_PRIOR` | 5 | a code needs a few dozen informative instances before its share moves far from 0.5 |
| `TAG_SCALE` | 0.5 | tags are long windows only loosely aligned with play (docs/EDA.md): weak evidence |
| `P_MIN`, `P_MAX` | 0.1, 0.9 | see the 0.9 clip above |

None of these was chosen on the public 5 minutes. They are checked with:

- `python -m saha.occluded.selfcheck` (label-free, below);
- WS5's external data, once it exists.

## Checks

- **Self-check, no labels:** `python -m saha.occluded.selfcheck`.
    - It hides 2, 5, 10 and 20 s of tracked ball in the middle of long tracked runs, and recomputes
      the opinion there.
    - It compares the result with the touch team on the hidden frames: coverage, accuracy, and
      calibration per confidence band.
    - It runs the bridge alone and the bridge with tags.
    - Caveat: a hidden tracked ball is easier than a real gap, and the target is the touch team,
      not the truth.
- **Synthetic A/B** (synthetic data only). Setup:
    - 40% of graded frames hidden in 3–30 s stretches;
    - a dead-ball stand-in from the synthetic script;
    - the nearest-player evidence for `p_home`.

  Results:
    - Without WS4: spellF1 0.786; 57% of the hidden alive frames right.
    - With the bridge alone: 0.807 and 57%. Synthetic possessions (3–25 s) often fit entirely
      inside a gap, so the edges alone cannot find them.
    - With the bridge and the tags: 0.992 and 97%. This is optimistic: the synthetic tags are
      exactly aligned with the truth, and the real ones are not.

## Limitations and next steps

- **Possessions entirely inside a gap are invisible to the bridge.** Only the tags (and later the
  team shape) can find them. Next: a team-shape model trained without labels on the tracked
  frames, with the touch team as the target. Its features would be:
    - team centroids and their spread;
    - the height of each defensive line;
    - who is in the opponent's half;
    - pressing speed near the last ball position.

  The model is then applied to the missing frames. It needs no ground truth and no hand labels.
- **The tags' timing.** Restart tags lag the real restart by seconds (docs/EDA.md), and phase tags
  probably do too. Shifting each tag window by a lag measured on the touches is the next
  refinement.
- **Dead ball inside gaps.** `p_home` is conditional on the ball being alive. Inside stoppages,
  WS2 decides; the 0.9 clip keeps us from fighting it.
- **Weight.** The per-frame opinions of one gap all come from the same two sides, so they are not
  independent. Weight 1 is a placeholder: WS1 sets the scale in `configs/pipeline.toml` on WS5's
  data.

## Assets

- The tactical ATD, read through the frame table: ball and player positions, ball visibility.
- Smart Tagging: codes, teams and windows.
- Not used: the homography, the videos.

## Training data

None. Every number is either a stated parameter or measured by the algorithm on this match's
tracking, without labels.

## Reproduction

```bash
python -m saha.occluded.build        # writes cache/evidence/occluded.parquet, prints aggregates
python -m saha.occluded.selfcheck    # label-free accuracy and calibration
python -m pytest tests/test_occluded.py
```
