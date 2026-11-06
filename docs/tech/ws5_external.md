# WS5: external data and independent validation

## Approach

Parameters must be tuned on external data only (DECISIONS D016, D017). WS5 builds labelled
external matches, degrades them to look like ours, and scores the whole pipeline on them with the
official metric, so that every package and WS1's combiner can choose values there.

```bash
python -m saha.external.build                    # download + convert the default DFL match
python -m saha.external.build --all              # all seven (about 5 GB of XML)
python -m saha.external.score                    # score configs/pipeline.toml on them
python -m saha.external.score --sweep cost_switch=15,25,35,50 cost_to_dead=20,35
python -m saha.external.score --no-degrade       # the same on clean data, for comparison
```

`score` runs each package's own `build(table)` on the degraded external table, fuses and
decodes with `saha.combine` exactly as the pipeline does, and scores with
`evaluation/possession_metrics.py` unmodified. It also scores each source alone, to check that
the combiner beats the best single source (WS1's success criterion). A builder that cannot run
on external data (for example one that needs our Smart Tagging) is reported as skipped.

Code: `saha/external/` (`dfl.py` reader, `degrade.py`, `harness.py`, `build.py`, `score.py`).
Downloads and converted arrays live in `cache/external/` (gitignored). External data is not our
match, so its numbers and statistics may be shared freely.

## Assets

| dataset | what we use | licence | status |
|---|---|---|---|
| DFL/Sportec open data (IDSSE): 7 Bundesliga / 2. Bundesliga matches, 2022/23 | position files: players, ball, and the ball's `BallStatus` and `BallPossession` on every frame, at 25 fps | CC BY 4.0 | in use |
| Metrica sample data | (planned) game 3, same EPTS format as ours; labels would have to come from events | to check | not yet |
| SkillCorner Open Data | (planned) 10 fps, possession per frame | to check | not yet |

Attribution: Bassek, M., Weber, H., Rein, R., & Memmert, D. (2025). *An integrated dataset of
spatiotemporal and event data in elite soccer.* Scientific Data. Data on figshare (Springer
Nature), downloaded through the figshare API with the file ids that kloppy 3.18 lists.

kloppy reads the same files (`sportec.load_open_tracking_data`), but the default figshare
download link answers automated clients with an empty `202`, and kloppy builds one Python object
per frame and player. `saha.external.dfl` streams the XML with the standard library instead:
about one minute per match, once.

**Check requested by TEAM.md, done:** the open files do fill `BallStatus` and `BallPossession`.
In match J03WPY, 50 of 146,261 frames are unlabelled (our 50-frame separator between halves);
every frame inside a half has a label.

## Training data

Labels come from DFL's own per-frame fields, produced by the data provider, not by us. Mapping:

| DFL | ours |
|---|---|
| `BallStatus = 0` | D |
| `BallStatus = 1`, `BallPossession = 1` (home) | H |
| `BallStatus = 1`, `BallPossession = 2` (guest) | A |
| anything else | X (not scored) |

None of it comes from our match.

## Degradation protocol (`saha/external/degrade.py`)

| ours (docs/EDA.md) | external, after degradation |
|---|---|
| ball missing on ~41% of graded frames | hidden on 41% of each half's scored frames |
| missing stretches: median 10.7 s, max 221 s, none 1-3 s, 85% of missing frames in stretches of 10 s or more | log-normal stretches, median 10.7 s, sigma 1, clipped to 3-221 s, placed uniformly (seeded); sigma 1 puts ~86% of the mass in stretches of 10 s or more |
| no ball height, ball speed channel unusable (D015) | height dropped, no ball speed channel |
| 46 track slots, no identities | a fresh slot per person and half |
| player speed in m/s (D012) | speed from positions (`kinematics.displacement_speed`) |

The degradation numbers are aggregate facts about our match's conditions from docs/EDA.md, used
to simulate them. No parameter of any predictor is estimated from our match.

Known gaps, next steps:

- Hidden stretches are placed at random. In our match the ball may vanish more in some phases
  (touchline, crowds, dead ball). Compare placements once we know more.
- Slots are split only at half-time. Our tracklets break mid-half too.
- The DFL pitch is 105 x 68 m; `FrameTable.ball_xy_m()` rescales with ours (105.3 x 68), a 0.3%
  stretch along the length.

## Possession rules: do their labels mean what ours mean?

`python -m saha.external.build` prints the same aggregates for each external match and for the
public 5 minutes of our match. This checks whether the definitions agree. It never tunes anything.

| | DFL J03WPY (97.5 min) | our public 5 min |
|---|---|---|
| possessions per minute | 4.32 | 3.40 |
| mean / median possession | 7.7 s / 3.8 s | 14.2 s / 7.6 s |
| possessions shorter than 2 s | 29.5% | 11.8% |
| D rate | 44.3% | 19.6% |
| median stoppage | 22.8 s | 16.4 s |

The definitions **differ**, as TEAM.md warned:

- DFL's `BallPossession` seems to change on every touch, contested balls included, so its
  possessions are shorter and more numerous.
- DFL's `BallStatus` is dead for much longer: more than twice our D rate.

The public 5 minutes are only 17 possessions, so this is indicative. Until it is resolved,
parameters that depend on possession length (the switching costs above all) should not be tuned
on raw DFL labels alone. Next: a documented, reasoned relabelling of DFL (for example, a touch
by the other team counts as a change only if it keeps the ball for a minimum time), with the
choice justified by football reasoning and compared on all seven matches.

## Limitations

- One match converted so far; the seven together are needed before any value is chosen.
- The possession definition mismatch above.
- External scores are about DFL matches with simulated occlusion, not our match. They rank
  versions; they do not predict our final score.

## Reproduction

```bash
python -m saha.external.build --all
python -m saha.external.score --bootstrap 1000
```

Everything is seeded (`--seed`, default 0). Downloads are checked by the SHA256 recorded in each
converted file's metadata.
