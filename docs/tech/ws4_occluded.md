# WS4: possession while the ball is missing

Owner: Baki. Code: `saha/occluded/`. Evidence: `cache/evidence/occluded.parquet`.

The ball is not tracked on 41% of graded frames, mostly in long stretches (median 10.7 s,
docs/EDA.md). WS4 gives `p_home` on exactly those frames: graded frames where `ball_visible` is
false. It never writes `p_dead` (WS2) and never writes on a frame with a tracked ball (WS3).

## Approach (v2)

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
- The halves are the table's own `half` array (1 or 2 inside a graded half, 0 elsewhere). On our
  match it is `alignment.half_of_frames()`, so the output is the same as cutting at the published
  graded halves. On an external table from WS5's harness it is that match's halves.

**2. Smart Tagging phases (weak evidence).** Each code's meaning comes from its official
definition and is fixed. OVERVIEW.md defines the tag *Team* as "the team performing the event",
and each code's definition says whether that team has the ball. Nothing is measured on this
match (DECISIONS D017).

| side | codes | from the definition |
|---|---|---|
| with the ball, `P = TAG_P = 0.7` | BUILD UP, PROGRESSION, CREATING CHANCES | "offensive actions" |
| | ATTACKING TRANSITION | "immediately after recovering the ball" |
| | BALL IN FINAL THIRD | "the team maintains or plays the ball" |
| | BALL IN THE BOX | the ball "enters or is played inside the rival box" |
| | PLAYERS IN THE BOX | "attacking players ... during an offensive action" |
| | LONG BALL | the passer's team: a loose ball stays with the last team that had it (CHALLENGE.md); the bridge's far side says who got it |
| without the ball, `P = 1 − TAG_P = 0.3` | DEFENDING IN ATTACKING / MIDDLE / DEFENSIVE THIRD | "defensive actions" |
| | DEFENSIVE TRANSITION | "immediately after losing possession" |
| no opinion | SET PIECES | a restart's team is WS2's question (TEAM.md, boundaries) |
| | GOALS | a stoppage and the other team's kick-off follow |
| | team N/A, any other code | counted in the build's report, never an error |

- `P` is P(the tagged team has the ball). `TAG_P = 0.7` says "more likely than not, but far from
  sure": Smart Tagging is automatic, with false positives and negatives, and its windows are
  phases, not possessions (DATA.md, OVERVIEW.md).
- On a missing frame, the overlapping tags are averaged in log-odds, not summed, and added to the
  bridge at `TAG_SCALE = 0.5`. So the tags give at most 0.5 × logit(0.7) ≈ 0.42 nats per frame.
- No external dataset has Smart Tagging, so neither value can be tuned (D016). Both stay weak on
  purpose.
- An external table (WS5's harness, `"external"` in `table.meta`, the same test as
  `saha/deadball/build.py`) gets no tags: never this match's.

The result is clipped to [0.1, 0.9], with weight 1.

## Why this, and not something else

- **Interpolating the ball buys little.** The tracker bridges short occlusions itself: there are
  no gaps between 1 and 3 s, and 85% of the missing frames are in gaps of 10 s or more
  (docs/EDA.md).
- **The forward–backward chain** gives the switch position that minimises the expected error
  when nothing else is known. It is calibrated by construction under its own assumptions. It is
  also honest about long gaps: they fade to "no opinion", so they do not force a team.
- **Each tag's meaning comes from its official definition.** Measuring it on this match's touches
  (v1) would be a value that an algorithm estimates from this match on every run, which the
  organizers' rules forbid (DECISIONS D017). The official definitions say which side each code
  names, so no measurement is needed. The strength, `TAG_P`, is one modest value chosen by
  reasoning, the same for every code.
- **The 0.9 clip.** Since WS1's exact emissions, a confident `p_home` no longer pushes against D
  ([experiments/20261105_ws1_exact-emissions.md](../../experiments/20261105_ws1_exact-emissions.md)).
  The clip stays for another reason: the opinions inside one gap all come from the same two
  sides, so they are not independent, and no single frame should decide alone.

## Parameters and how each is set

| parameter | value | how it is set |
|---|---|---|
| `CONTROL_M`, `CONTROL_SPEED` | 1.5 m, 5 m/s | reasoning: the touch of `saha.tools.selfcheck`, at a player's feet, not passing |
| `ANCHOR_HORIZON` | 2 s | reasoning: the last touches before the ball disappears; older ones belong to another phase |
| `ANCHOR_HALF_LIFE` | 0.5 s | reasoning: a hand-over just before the gap must win over a long spell before it |
| `ANCHOR_MAX` | 0.9 | reasoning: the touch team is wrong sometimes (a contested ball, a defender passing close by) |
| `MEAN_HOLD_S` | 12 s | football knowledge: team possessions last 10–20 s in top-flight football; short contested ones count here too |
| code table (`TAG_SIDE`) | with / without / none | the official definitions (OVERVIEW.md), table above |
| `TAG_P` | 0.7 | reasoning: more likely than not, far from sure (automatic tags, phase windows) |
| `TAG_SCALE` | 0.5 | reasoning: tags are long windows only loosely aligned with play (docs/EDA.md): weak evidence |
| `P_MIN`, `P_MAX` | 0.1, 0.9 | reasoning: see the 0.9 clip above |

Every value is set by football and physics reasoning or by the official definitions. None was
fitted, and none was chosen on this match, its tracking, its tags or the public 5 minutes. The
self-check (below) and WS5's harness check them; they never choose them (D016, D017).

## Checks

These checks catch errors. They never choose a value (D016, D017).

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
      exactly aligned with the truth, and the real ones are not. (Measured with v1; the
      synthetic tags are honest, so v2's fixed table reads them the same way.)
- **WS5's external harness:** `python -m saha.external.score` runs the same `build(table)` on
  degraded DFL matches. It scores the bridge alone, because DFL has no Smart Tagging.

## Results (v2, 12 November)

Record: [experiments/20261112_ws4_bridge-fixed-tags-v2.md](../../experiments/20261112_ws4_bridge-fixed-tags-v2.md).
None of these numbers chose a value.

**WS5's external harness** (DFL J03WPY and J03WQQ, open data, CC BY 4.0, the ball hidden on
about 41% of scored frames; the bridge alone, because DFL has no Smart Tagging). The "all" row,
with and without WS4:

| match | spellF1 with | spellF1 without | macroF1 with / without | team accuracy with / without |
|---|---|---|---|---|
| J03WPY | 0.3840 [0.35–0.42] | 0.3698 [0.33–0.41] | 0.7950 / 0.7856 | 0.810 / 0.796 |
| J03WQQ | 0.3597 [0.32–0.40] | 0.3494 [0.31–0.39] | 0.7680 / 0.7638 | 0.754 / 0.747 |
| mean | 0.3719 | 0.3596 | 0.7815 / 0.7747 | 0.782 / 0.772 |

WS4 helps on both matches (+0.012 spellF1 on average). The gain is inside the bootstrap noise,
and it is never below the run without WS4.

**WS1's seven-match check (12 November).** On all seven of WS5's DFL matches, mean spellF1 with
WS4 is 0.3649 against 0.3564 without (+0.0085, raw labels), and WS4 is ahead on 6 of 7 matches.
With WS1's controlled-possession labels and with PR #19 the gain is +0.007 to +0.009, ahead on 5
or 6 of 7. The only clear loss is J03WN1, by 0.001 to 0.004. WS4 v2 is adopted.

**On our match (checks only):**

- The self-check's bridge + tags accuracy is 0.951, 0.952, 0.810 and 0.771 for 2, 5, 10 and
  20 s hidden; the bridge alone gives 0.914, 0.921, 0.792 and 0.763. The tags are not inverted.
- Match-wide, with / without WS4:
    - possessions 365 / 357;
    - mean duration 9.0 / 9.2 s;
    - D rate 48.3% / 48.3%;
    - home share 61.0% / 60.8%.
- Public spellF1, a pipeline check only: 0.4778 / 0.4333, with overlapping intervals.

## Limitations and next steps

- **Possessions entirely inside a gap are invisible to the bridge.** Only the tags can find them
  today. A team-shape model (team centroids and spread, the height of each defensive line, who is
  in the opponent's half, pressing near the last ball position) could too, but only if it is
  trained on WS5's external data (D017, D024). Never on this match's tracking. Not before the
  feature freeze.
- **The tags' timing stays as delivered.** Restart tags lag the real restart by seconds
  (docs/EDA.md), and phase tags may too. No external set has Smart Tagging to measure a lag on,
  and it must not be measured on this match (D017), so no shift is applied.
- **Dead ball inside gaps.** `p_home` is conditional on the ball being alive. Inside stoppages,
  WS2 decides.
- **Weight.** The per-frame opinions of one gap all come from the same two sides, so they are not
  independent. Weight 1 is a placeholder: WS1 sets the scale in `configs/pipeline.toml` on WS5's
  data.

## Assets

- The tactical ATD, read through the frame table: ball and player positions, ball visibility.
- Smart Tagging: codes, teams and windows.
- OVERVIEW.md's Smart Tagging definitions: the source of each code's side.
- Not used: the homography, the videos.

## Training data

None. Every number is set by football reasoning or by the official definitions, and nothing is
measured or fitted on this match.

## Reproduction

```bash
python -m saha.occluded.build        # writes cache/evidence/occluded.parquet, prints aggregates
python -m saha.occluded.selfcheck    # label-free accuracy and calibration
python -m pytest tests/test_occluded.py
```
