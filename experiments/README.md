# Experiments

One file per experiment: `YYYYMMDD_<package>_<short-name>.md`, for example
`20261103_ws2_restarts-v1.md`. Nobody shares a log file, so there are no merge conflicts.

```bash
python -m saha.pipeline                                  # run the change first
python -m saha.common.experiments new ws2 restarts-v1    # creates the record, numbers filled in
python -m saha.common.experiments                        # every experiment in one table
python -m saha.common.experiments check                  # what CI runs
```

`new` copies the numbers of the last pipeline run (`submissions/SAHA.summary.json`):

- public spellF1 with its bootstrap interval
- macroF1
- match-wide possession count, mean duration and D rate

Then you write four short sections: what changed, why, the result, and the decision.

The **why** must be a football or physics reason, or a WS5 external-validation result. "The
public score went up" is never a reason on its own (CLAUDE.md). Record rejected experiments too:
they are half the technical document.
