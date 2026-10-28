# EDA: not generated yet

This file is produced by `python -m saha.tools.eda` from the real data. It can only run on a
member's own machine: the data may not be made available to a cloud-hosted LLM (CLAUDE.md, rule
3), so the Phase 0 cloud session could not run it.

To generate it (also part of the Phase 0 checks):

```bash
python -m saha.tools.data_checks       # every Phase 0 check, then this file and its figures
python -m saha.tools.eda               # this file only
```

The output is aggregate only: distributions, counts and percentages, with three figures in
`docs/figures/`. It covers:

- coverage
- ball-missing stretches
- the ball outside the pitch
- player speeds in dead ball
- Smart Tagging restarts against the public ground truth
- track-slot fragmentation and identity switches

Read it before committing it.
