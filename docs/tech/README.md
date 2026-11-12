# Technical document: how it is assembled

The deliverable is `SAHA.md` (or `SAHA.pdf`): a few pages, in English, uploaded with the form.
WS1 assembles it on 13–15 November:

```bash
python -m saha.tools.techdoc        # writes SAHA.md at the repository root
```

It takes `saha_main.md` (the cross-package sections, WS1) and appends each package's section as
an appendix, in WS order. Each section is written here, by its owner:

| file | owner |
|---|---|
| `ws1_pipeline.md` | WS1: data layer, evidence interface, combiner, reproduction |
| `ws2_deadball.md` | WS2 |
| `ws3_visible.md` | WS3 |
| `ws4_occluded.md` | WS4 |
| `ws5_external.md` | WS5: datasets, licences and attribution, degradation protocol |

Each section covers what SUBMISSION.md asks for:

- **Approach.** What you built, and why that rather than something else.
- **Assets.** Which assets you used and how. List the ones you tried and dropped too.
- **Training data.** Every labelled dataset: source, who labelled it, how much. None of it may
  come from this match.
- **Possession rules.** Where you drew the line on loose balls, challenges and restarts.
- **Limitations.** Where it fails, and what you would fix with more time.
- **Reproduction.** How to run it.

Write it as you go. The experiment records in `experiments/` are your raw material, rejected
ideas included.
