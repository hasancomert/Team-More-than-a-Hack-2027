"""Score our pipeline on external matches with the official metric.

    python -m saha.external.score                    # every converted match, configs/pipeline.toml
    python -m saha.external.score --sweep cost_switch=15,25,35,50
    python -m saha.external.score --matches J03WPY --no-degrade

For each match: degrade it to our conditions (saha.external.degrade), run every evidence builder
of the config on that table -- the same `build(table)` the pipeline runs on our match -- fuse and
decode with WS1's combiner, then score with evaluation/possession_metrics.py, unmodified.

The numbers are about external matches, so they may be shown and shared (unlike our match's
data). This is where parameters are chosen (DECISIONS D016): sweep them here, not on the public
5 minutes.

A builder that cannot run on external data (for example one that reads our Smart Tagging) is
reported as skipped, with the error, and left out of the fusion.
"""

import importlib
import itertools
from dataclasses import dataclass, replace

import numpy as np

from saha.combine import fuse, viterbi
from saha.combine.decode import CombineParams
from saha.common import evidence, submission
from saha.common.constants import X

HEADLINE = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "macro_f1",
            "alive_dead_f1", "team_acc_both_alive", "n_pred_spells", "n_gt_spells",
            "mean_spell_duration_s", "gt_mean_spell_duration_s")


@dataclass
class Source:
    name: str
    evidence: object      # saha.common.evidence.Evidence, or None if the builder was skipped
    scale: float
    skipped: str = ""


def half_runs(half):
    """[(start, stop)] half-open runs of each half number > 0, in order."""
    runs = []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], half, [0]))) != 0)
    for start, stop in zip(edges[:-1], edges[1:]):
        if half[start] > 0:
            runs.append((int(start), int(stop)))
    return runs


def build_source(name, module, scale, table):
    """Run one package's build(table) on an external table, as Evidence (quantised, checked)."""
    try:
        builder = importlib.import_module(module).build
        p_dead, p_home, weight = builder(table)
        p_dead, p_home, weight = (evidence.quantise(a) for a in (p_dead, p_home, weight))
        found = evidence.problems(np.arange(table.n_frames, dtype=np.int32), p_dead, p_home,
                                  weight, n_frames=table.n_frames)
        if found:
            return Source(name, None, scale, "invalid: " + "; ".join(found))
    except Exception as error:  # a package that needs our match's files, say
        return Source(name, None, scale, "{}: {}".format(type(error).__name__, error))
    return Source(name, evidence.Evidence(name, p_dead, p_home, weight, {}), scale)


def has_opinion(source):
    e = source.evidence
    return e is not None and bool(((np.isfinite(e.p_dead) | np.isfinite(e.p_home))
                                   & (e.weight > 0)).any())


def decode(sources, params, half):
    """Label codes: the combiner's fuse + Viterbi on every half, X elsewhere."""
    pairs = [(s.evidence, s.scale) for s in sources if s.evidence is not None]
    labels = np.full(len(half), X, dtype=np.int8)
    if not pairs:
        return labels
    dead, home = fuse.fuse(pairs, params.eps, params.prior_dead)
    costs = [2 * fuse.nats_to_units(c)
             for c in (params.cost_switch, params.cost_to_dead, params.cost_from_dead)]
    for start, stop in half_runs(half):
        labels[start:stop] = viterbi.decode(viterbi.emissions(dead[start:stop],
                                                              home[start:stop]), *costs)
    return labels


def score(labels, codes, bootstrap=0):
    result = submission.metrics().evaluate(labels, np.asarray(codes, dtype=np.int8),
                                           bootstrap_n=bootstrap)
    return {k: result[k] for k in HEADLINE}


def sweep_grid(params, sweep):
    """Every CombineParams in the cartesian product of sweep {name: [values]}."""
    if not sweep:
        return [params]
    names = sorted(sweep)
    unknown = [n for n in names if n not in CombineParams.__dataclass_fields__]
    if unknown:
        raise ValueError("unknown combine parameters: {}".format(unknown))
    return [replace(params, **dict(zip(names, values)))
            for values in itertools.product(*(sweep[n] for n in names))]


def evaluate_table(table, labels, entries, params, sweep=None, bootstrap=0, singles=True):
    """Rows of results for one degraded match: the full fusion per grid point, then each source
    alone at the base parameters (WS1's 'the combiner beats the best single source')."""
    sources = [build_source(e["name"], e["module"], float(e.get("scale", 1.0)), table)
               for e in entries]
    rows = []
    for point in sweep_grid(params, sweep or {}):
        codes = decode(sources, point, table.half)
        rows.append({"sources": "all", "params": point.to_dict(),
                     **score(labels, codes, bootstrap)})
    if singles:
        for source in sources:
            if not has_opinion(source):
                continue
            codes = decode([replace(source, scale=1.0)], params, table.half)
            rows.append({"sources": source.name, "params": params.to_dict(),
                         **score(labels, codes, bootstrap)})
    status = {s.name: (s.skipped or ("ok" if has_opinion(s) else "no opinion (stub)"))
              for s in sources}
    return rows, status
