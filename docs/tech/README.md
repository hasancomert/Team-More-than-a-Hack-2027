# Technical document: how it is assembled

The deliverable is `SAHA.md` (or `SAHA.pdf`): a few pages, in English, uploaded with the form.
WS1 assembles it on 13–15 November:

```bash
python -m saha.tools.techdoc          # writes SAHA.md at the repository root
python -m saha.tools.techdoc --full   # the same with every package's whole section
```

It takes `saha_main.md` (the cross-package sections, WS1) and appends one short appendix per
package from `appendix_summaries.md` (WS1's summaries, each linking to the full section). The
form asks for a short document, so that is the submitted version (team leader, 14 November);
`--full` appends each package's whole section instead, in WS order. Each section is written
here, by its owner:

| file | owner |
|---|---|
| `ws1_pipeline.md` | WS1: data layer, evidence interface, combiner, reproduction |
| `ws2_deadball.md` | WS2 |
| `ws3_visible.md` | WS3 |
| `ws4_occluded.md` | WS4 |
| `ws5_external.md` | WS5: datasets, licences and attribution, degradation protocol |
| `appendix_summaries.md` | WS1: one short appendix per package, from the sections above |

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
