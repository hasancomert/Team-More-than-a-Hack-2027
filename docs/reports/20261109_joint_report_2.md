# Joint score report 2 (9 November 2026)

WS1 wrote this report on 8 November, from what the packages have pushed and recorded. Joint report
1 (5 November) was not written: no package had delivered evidence by then. This report covers both.

How to read the numbers:

- **External numbers** are about other matches: WS5's seven DFL matches and WS2's twenty
  SkillCorner matches. They are the only place where versions are compared and values chosen
  (DECISIONS D016, D017).
- **Numbers about our match** are aggregate checks. They catch errors and never choose anything.
  The public 5 minutes are a pipeline check only (CLAUDE.md, rule 2).

## Where each package stands

| package | owner | state | evidence on our match |
|---|---|---|---|
| WS1 combiner, `nearest` | Hasan | on `main` | `p_home` where the ball is tracked (the official baseline, as evidence) |
| WS2 dead ball | Alparslan | PR #14: draft, approved, reproduced on two machines; waits for "Ready for review" | `p_dead` on every graded frame |
| WS3 ball tracked | Osman | PR #12: still to run on the data and on WS5's matches; experiment record to fill in | `p_home` where the ball is tracked |
| WS4 ball missing | Baki | PR #7: draft, no activity since 5 November; the Smart Tagging part must change (D017) | `p_home` where the ball is missing |
| WS5 external data | Seyyid | on `main` (PR #9) | DFL validation harness; the relabelling is next |

## External results

### DFL: seven matches, degraded like ours by WS5's harness (seed 0)

| configuration | mean spellF1 |
|---|---|
| `nearest` alone, `cost_switch` 35 (the current setting) | 0.210 |
| `nearest` + WS2, first classifier | 0.311 |
| `nearest` + WS2, classifier trained with the hidden ball too | 0.319 |

- **WS2's dead ball on DFL.** With the second classifier, dead-ball F1 goes from 0.790 to 0.917
  and stoppage F1 from 0.570 to 0.783. The first classifier called 92% of the live frames dead
  wherever the harness hides the ball. The second one fixes that.
- **The `cost_switch` sweep is not used.** With `nearest` alone, `cost_switch` 15, 35 and 60 give
  0.223, 0.210 and 0.174. DFL's possession labels change on every touch, so low costs win by
  construction (experiments/20261106_ws5_dfl-first-harness.md). The costs wait for WS5's
  relabelling.

### SkillCorner: twenty matches, five folds held out (WS2's own tool)

| WS2 version | spellF1 |
|---|---|
| prototype's first design, no stoppage hold | 0.571 |
| SkillCorner classifier, hold after whistles only, prototype's team evidence | 0.632 |
| PR #14, no team evidence (the rules require it) | 0.626 |
| PR #14, classifier trained with the hidden ball too: as delivered | 0.624 |
| the same, with the ball hidden as WS5 hides it | 0.583 (first classifier: 0.425) |

## Our match: checks only

| version | SHA256 | possessions | mean / median s | under 2 s | D | public spellF1 (check) |
|---|---|---|---|---|---|---|
| `main` (`nearest` only) | `9fedd7df…` | 368 | 11.7 / 8.2 | 0 | 32.3% | 0.40 [0.26–0.54] |
| + WS2, first classifier | `081f57c1…` | 305 | 10.3 / 6.3 | 24 | 50.4% | 0.48 [0.32–0.67] |
| + WS2, second classifier | `3b74ae89…` | 293 | 11.2 / 6.7 | 15 | 48.3% | 0.48 [0.31–0.65] |

The public intervals overlap, so the public window says nothing about which version is better.

Reproduction, byte for byte:

- `9fedd7df…`: the team leader and Seyyid.
- `081f57c1…`: Alparslan and the team leader.
- `3b74ae89…`: Alparslan and the team leader (10 November).

## Risks before the freeze (12 November)

1. **WS3 and WS4 are not merged.** WS4 has been silent since 5 November. If it stays silent,
   the fallback is the bridge part alone, without the Smart Tagging part.
2. **The combiner's costs are still reasoned placeholders** (35 nats). They can be tuned only on
   a DFL set whose possessions mean what ours mean, so WS5's relabelling is on the critical path.
3. **Double counting.** `nearest` and WS3's `visible` read the same nearest player. The scale of
   `nearest` has to be set on WS5's data once WS3 merges.
4. **Dead ball under a missing ball.** On our match, no allowed check can tell whether the long
   missing-ball stretches are live or dead play. WS2's second classifier is robust to both on
   external data, which is the best evidence we can have.
5. **The whistle** stays off until the team decides (D006).

## Next steps for WS1

- **When WS3 merges:** score `nearest` + WS2 + WS3 together on WS5's DFL matches, and set the
  scale of `nearest` there.
- **When WS5's relabelling lands:** tune `cost_switch`, `cost_to_dead` and `cost_from_dead` on
  DFL.
- **12 November:** FINAL_CHOICE.md.
- **13–15 November:** SAHA.md, assembled with `python -m saha.tools.techdoc`.
