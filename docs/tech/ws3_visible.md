# WS3: possession while the ball is tracked

Owner: Osman. Code: `saha/visible/build.py`. Tests: `tests/test_visible.py`.
Experiments: `experiments/20261106_ws3_control-hysteresis-v1.md` (v1),
`experiments/20261111_ws3_touch-confirm-v2.md` (v2),
`experiments/20261111_ws3_first-touch-weight-v21.md` (v2.1, the submitted version).

## Approach

The ball is tracked on 59% of graded frames, and its positions are clean there (docs/EDA.md). The
official baseline, and our `nearest` evidence, give the ball to the team of the nearest tracked
player on every one of those frames: 17.6 changes of team per graded minute on our match. Under
spellF1 every change that survives the decoder and is not real is a false possession.

CHALLENGE.md draws the line: possession changes only when the opponent gains control; a touch
without control, a deflection and a won aerial duel do not count; a loose or contested ball stays
with the last team. v2.1 implements that in three steps.

1. **Touches.** A frame is a touch frame for a team when that team's nearest player is the closest
   to the ball and within `TOUCH_M` = 1.5 m, at any ball speed and duels included. Runs of touch
   frames of one team at most `JOIN` = 3 frames apart are one touch.

2. **Confirmed hand-overs.** A touch by the team without the ball is a candidate. It is confirmed
   when, before the team with the ball touches it again,
    - the touch lasts `CONFIRM_FRAMES` = 6 frames or more: the player keeps the ball; or
    - the same team touches it a second time (`CONFIRM_TOUCHES` = 2): a completed pass or a carry.

   An unconfirmed candidate is a deflection or a lost duel, and the ball stays with the last team.
   A confirmed hand-over is placed at the candidate's first frame: the new team's first touch after
   the old team's last one.

   A stretch's first touch starts a possession without confirmation (`FIRST_TOUCH_OWNS`): a
   restart is taken by the team with the ball, and after a ball gap nobody else has touched it.

3. **Probabilities.** On each tracked graded frame after a stretch's first possession, for the team
   with the ball, at weight `WEIGHT` = 2 (twice `nearest`'s):

   | situation | P(team has the ball) |
   |---|---|
   | its own touches | `P_TOUCH` = 0.9 |
   | the ball between two of its touches | `P_HOLD` = 0.8 |
   | the ball on its way to the other team, the other team's unconfirmed touches, after a stretch's last touch | `P_LOOSE` = 0.6 |
   | before a stretch's first touch | no opinion |

   The state is forgotten at the start of each graded half and across a ball gap longer than
   `RESET_GAP` = 0.5 s: what happened while the ball was missing is WS4's question.

The combiner's Viterbi pass adds switching costs on top. WS3's job is to hand it evidence that
already follows the possession rules, so that it changes team where an opponent takes the ball,
not where the ball passes near one.

### Why v2 replaced v1

v1 required control: a ball slower than 5 m/s, no opponent within 1 m more, for 6 frames, and it
placed the change at the start of that control. On WS5's DFL match J03WPY (open data, labelled on
every frame), that removed most changes and placed the rest late:

- at the 324 labelled changes the new team is within 1.5 m of the ball in 100% of cases, but the
  full control condition holds in only 23%;
- the start of the new team's control is a median 19 frames late; its first touch after the old
  team's last one is 2 frames late.

### Results on WS5's seven DFL matches

Degraded like our match, fused with WS2's dead ball, official metric, mean of seven matches:

| configuration | spellF1 | above `main` |
|---|---|---|
| `main`: `nearest` 1.0 + WS2, no WS3 | 0.3186 | |
| v1, `nearest` 1.0 | 0.3216 | 4/7 |
| v2, `nearest` 1.0 | 0.3564 | 7/7 |
| v2.1, `nearest` 1.0 | 0.3617 | 7/7 |
| **v2.1, `nearest` 0.5** | **0.3869** | 7/7 |

## Where the values come from

No value is fitted on this match (D016, D017).

- 1.5 m: a stride plus the ball's reach; the touch of `saha.tools.selfcheck` and WS4, so that the
  packages agree on what "at the ball" means.
- 3 frames to join: 0.12 s, the tracker's jitter around the line; no pass comes back that fast.
- 6 frames to confirm by keeping: longer than a deflection (1–3 frames at 25 fps).
- 2 touches to confirm by playing: CHALLENGE.md's proxies for control are keeping the ball and
  completing passes.
- 0.9, 0.8, 0.6: confidence falls from a player on the ball, to a ball between teammates, to a ball
  about to change hands. None is 1, because the tracker swaps and merges players.
- 0.5 s reset: the tracker bridges short occlusions itself (docs/EDA.md).

- Weight 2 and the first touch: tuned on WS5's seven DFL matches, where together they beat v2 on
  7 of 7 (experiment v2.1). Weight 3 was no better, so the smaller value stays.

Variants of the touch rule (3 touches, 10 frames, 1.0 m) were scored on DFL too. None is better on
every match, so the reasoned values stay (experiment v2).

## Assets

- Used: the frame table (ball and player positions, team of each slot, graded halves).
- Not used: Smart Tagging (WS2 and WS4 use it), homography, video, the ATD's ball speed channel
  (D015).

## Training data

None. WS3 has no learned parameters. WS5's DFL matches (open data, CC BY 4.0,
docs/tech/ws5_external.md) were used to compare versions and variants.

## Limitations

- `nearest` and `visible` read the same nearest player, so fusing both counts it twice. On seven
  DFL matches, v2.1 scores best with `nearest` at 0.5; the scale is WS1's setting (TEAM.md).
- Short possessions. v2.1 makes more of them on our match (76 under 2 s, 20% of possessions, at
  `nearest` 0.5). DFL's labels change on every touch and reward that; ours ask for control. If our
  ground truth has fewer short possessions, part of the DFL gain does not transfer. A DFL set
  relabelled to a controlled definition (WS5) would settle it.
- No ball height: a high ball over a player within 1.5 m counts as a touch. On DFL, a quarter of
  the labelled changes happen with the ball above 1.9 m (headers), so height would help where it
  exists.
- Identity switches between a home and an away slot near the ball can fake a touch. The
  confirmation rule limits this, it does not remove it.
- DFL's labels change on every touch, while ours require control. Until WS5 relabels DFL, the
  variants are ranked against a looser definition than ours.

## Reproduction

```bash
python -m saha.visible.build          # cache/evidence/visible.parquet, aggregate counts printed
python -m saha.pipeline               # the whole submission
python -m pytest tests/test_visible.py
python -m saha.external.score         # WS5's DFL matches
```
