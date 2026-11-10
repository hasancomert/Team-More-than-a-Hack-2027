---
date: 2026-11-06
package: ws2
name: whistle-detector
author: Claude Code (WS2), run by Alparslan
commit: 685f21aa1bef
config: configs/pipeline.toml
public_spell_f1: 0.4848
public_spell_low: 0.315
public_spell_high: 0.667
public_macro_f1: 0.7693
n_possessions: 305
mean_duration_s: 10.3
d_rate: 0.504
verdict: adopted
---

## What changed

STATUS.md, WS2 item 1: the whistle detector (`saha/deadball/audio.py`) no longer uses anything
measured on this match. The prototype's two bands (3.70–3.96 and 4.18–4.40 kHz, both tones
required) and its 20 dB threshold were read off this match's audio. Now:

- score: how far the strongest frequency between 3 and 6 kHz stands above that band's median,
  in dB, on 44.1 kHz audio with one 1024-sample Hamming window per video frame;
- band: the tones that referee-whistle makers publish (Dolfin PRO 3.67 / 4.15 kHz, Acme Tornado
  3.74 / 4.10 kHz, Fox 40 about 3.8 kHz) and those of the external recordings (4.7–5.3 kHz);
- threshold 38 dB, chosen on TU Dortmund's labelled whistle recordings (CC BY-SA 4.0);
- the audio streams from ffmpeg, so 44.1 kHz needs no more memory than 16 kHz did;
- `tests/test_deadball.py`: the streamed score equals a one-pass score on a synthetic blast,
  with uneven reads. It caught two bugs before anything ran on real audio.

The whistle stays off by default (item 8, a team decision). The submission does not change.

## Why

CLAUDE.md rule 1 and D016/D017: measurements taken by hand from this match's audio, and
calibration on the provided video, are not allowed.

## Result

External: TU Dortmund Whistle Detection Dataset, test split (Kleingarn and Brämer 2024,
doi:10.17877/tudodata-2024-m0f6bmi1; 147,440 windows of RoboCup referee whistles recorded on NAO
robots, 249 blasts, 0.45 h without whistles), read as `audio.py` reads our audio (one window per
1/25 s, blasts of at least 3 frames, gaps of up to 3 bridged). The threshold rule was fixed
before the full numbers were seen: the lowest threshold with no false blast.

| threshold | blasts found | false blasts per hour |
|---|---|---|
| 30 dB | 86.3% | 30.9 |
| 34 dB | 83.9% | 6.6 |
| 36 dB | 81.9% | 4.4 |
| **38 dB** | **73.1%** | **0** (95% upper bound 6.6) |
| 40 dB | 53.4% | 2.2 |

A score against side bands did slightly better there (94% of blasts at its own no-false-blast
threshold on 81% of the data, against 92% for this one), but its upper side band (6.5–8 kHz) holds
the second harmonic of a football whistle. Those recordings' whistles sit higher, so they cannot
show that, and the side-band score was left out.

Our match: `build --whistle` runs end to end (25 s, 31 whistles of which 11 in live play). Those
counts are printed by the builder and were not compared with anything; the detector was never
checked against this match (STATUS.md, item 1). Default pipeline unchanged: `SAHA.csv` SHA256
`081f57c1b588…`. `pytest`: 78 passed.

## Decision

Adopted as the whistle detector, still off by default. Whether to use the whistle at all is
the team's (item 8): it needs the video and ffmpeg on every machine, and a check that two
machines find the same whistle times.
