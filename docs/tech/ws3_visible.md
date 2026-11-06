# WS3: possession while the ball is tracked

Owner: Osman. Code: `saha/visible/build.py`. Tests: `tests/test_visible.py`.
Experiments: `experiments/*_ws3_*.md`.

## Approach

The ball is tracked on 59% of graded frames, and its positions are clean there (docs/EDA.md). The
official baseline, and our `nearest` evidence, give the ball to the team of the nearest tracked
player on every one of those frames. That is right while a player has the ball at his feet, and
wrong while it travels: a pass flying past an opponent, a deflection or a duel all look like a
hand-over, and under spellF1 every hand-over that survives the decoder is a false possession.

CHALLENGE.md draws the line differently: possession changes only when the opponent gains
**controlled** possession; a touch without control, a deflection and a won aerial duel do not count;
a loose or contested ball stays with the last team. v1 implements exactly that, in two steps.

1. **Control.** A frame is a control frame for a team when
    - its nearest player is within `CONTROL_M` = 1.5 m of the ball,
    - the ball is slower than `CONTROL_SPEED` = 5 m/s (speed from its positions, D015),
    - the nearest opponent is at least `CONTEST_M` = 1.0 m further away (otherwise it is a duel).

   A control run is at least `MIN_CONTROL` = 6 consecutive control frames (0.24 s) of one team.
   Shorter runs are touches in passing and are ignored.

2. **Hysteresis.** Between two control runs the ball belongs to the team of the first one. On each
   tracked graded frame `p_home` is

   | situation | P(team has the ball) |
   |---|---|
   | on a control run | `P_CONTROL` = 0.9 for its team |
   | between two runs of the same team | `P_HOLD` = 0.8 for that team |
   | between runs of different teams | `P_LOOSE` = 0.6 for the first team |
   | after the last run of a stretch | `P_LOOSE` = 0.6 for its team |
   | before the first run of a stretch | no opinion |

   The hand-over is therefore placed at the first frame of the opponent's control run. The state
   is forgotten at the start of each graded half and across a ball gap longer than `RESET_GAP` =
   0.5 s: what happened while the ball was missing is WS4's question.

The combiner's Viterbi pass already adds switching costs. WS3's job is to give it evidence that
agrees with the possession rules, so that a hand-over is pushed where an opponent takes control and
not where the ball passes near one.

## Where the values come from

No value is fitted on this match (D016, D017).

- 1.5 m and 5 m/s are the touch of `saha.tools.selfcheck` and WS4, so that the packages agree on
  what "at the ball" means. 1.5 m is a stride plus the ball's reach; a ball faster than 5 m/s is
  being passed, shot or cleared.
- 1.0 m for a duel: two players that close to each other at the ball are both challenging for it.
- 6 frames: receiving and keeping the ball takes longer than a deflection (1–3 frames at 25 fps).
- 0.9, 0.8, 0.6: confidence falls from a player on the ball, to a ball travelling between
  teammates, to a ball that is about to change hands. None of them is 1, because the tracker swaps
  and merges players.
- 0.5 s reset: the tracker bridges short occlusions itself (docs/EDA.md), so a longer gap is a
  real loss of the ball.

They are checked and tuned on WS5's external data (`python -m saha.external.score`).

## Assets

- Used: the frame table (ball and player positions, team of each slot, graded halves).
- Not used: Smart Tagging (WS4 and WS2 use it), homography, video, the ATD's ball speed channel
  (D015).

## Training data

None. v1 has no learned parameters.

## Limitations

- `nearest` and `visible` read the same nearest player, so fusing both at full scale counts it
  twice. Once WS3 beats `nearest` on WS5's data, WS1 lowers `nearest`'s scale (TEAM.md).
- No ball height: a high ball over a player within 1.5 m and slower than 5 m/s in the plane can
  look like control.
- Identity switches between a home and an away slot near the ball can fake a short control run.
  `MIN_CONTROL` and `CONTEST_M` limit this, they do not remove it.
- A hand-over is placed at the start of the opponent's run. If the ground truth switches at the
  first touch instead, our boundary is a few frames late.

## Reproduction

```bash
python -m saha.visible.build          # cache/evidence/visible.parquet, aggregate counts printed
python -m saha.pipeline               # the whole submission
python -m pytest tests/test_visible.py
```
