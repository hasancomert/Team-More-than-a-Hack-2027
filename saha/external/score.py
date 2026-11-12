"""WS5: score the pipeline on external matches. Usage: see saha.external.harness.

    python -m saha.external.score [--matches J03WPY ...] [--sweep name=v1,v2 ...]
                                  [--no-degrade] [--seed 0] [--bootstrap 0] [--config ...]

Writes cache/external/reports/score.json and prints a table. Every number is about external
open data (DFL/IDSSE), never about our match.
"""

import argparse
import json
from dataclasses import replace

from saha.common import paths
from saha.external import degrade, dfl, harness
from saha.pipeline import DEFAULT_CONFIG, load_config


def parse_sweep(items):
    sweep = {}
    for item in items or []:
        name, _, values = item.partition("=")
        if not values:
            raise SystemExit("--sweep expects name=v1,v2,... not {!r}".format(item))
        sweep[name.strip()] = [float(v) for v in values.split(",")]
    return sweep


def fmt(value, spec):
    return "--" if value is None else format(value, spec)


def print_rows(name, rows, status):
    print("\n{}".format(name))
    for source, state in status.items():
        print("  evidence {:<10} {}".format(source, state))
    print("  {:<10} {:>7} {:>6} {:>6} {:>7} {:>7} {:>6} {:>11}  params".format(
        "sources", "spellF1", "lo", "hi", "macroF1", "aliveF1", "team", "spells p/gt"))
    for r in rows:
        changed = {k: v for k, v in r["params"].items()} if r["sources"] == "all" else {}
        print("  {:<10} {:>7} {:>6} {:>6} {:>7} {:>7} {:>6} {:>5}/{:<5}  {}".format(
            r["sources"], fmt(r["spell_f1_mean"], ".4f"), fmt(r["spell_bootstrap_low"], ".2f"),
            fmt(r["spell_bootstrap_high"], ".2f"), fmt(r["macro_f1"], ".4f"),
            fmt(r["alive_dead_f1"], ".3f"), fmt(r["team_acc_both_alive"], ".3f"),
            r["n_pred_spells"], r["n_gt_spells"],
            " ".join("{}={:g}".format(k, v) for k, v in sorted(changed.items())
                     if k.startswith("cost") or k == "prior_dead")))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the pipeline on external matches.")
    parser.add_argument("--matches", nargs="*", help="DFL match ids (default: converted ones)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sweep", nargs="*", help="combine parameters, name=v1,v2,...")
    parser.add_argument("--no-degrade", action="store_true", help="score the clean data")
    parser.add_argument("--seed", type=int, default=0, help="degradation seed")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--no-singles", action="store_true", help="skip each-source-alone rows")
    args = parser.parse_args(argv)

    params, entries = load_config(args.config)
    sweep = parse_sweep(args.sweep)
    matches = args.matches or dfl.converted_matches() or list(dfl.DEFAULT_MATCHES)
    config = degrade.NONE if args.no_degrade else replace(degrade.Degradation(), seed=args.seed)
    print("external scoring: {} match(es), degradation {}".format(
        len(matches), "none" if args.no_degrade else config.to_dict()))

    report = {"config": args.config, "degradation": config.to_dict(), "matches": {}}
    for match_id in matches:
        match = dfl.prepare(match_id)
        table, labels = degrade.degrade(match, config)
        rows, status = harness.evaluate_table(table, labels, entries, params, sweep,
                                              args.bootstrap, singles=not args.no_singles)
        print("  ball hidden on {:.1%} of scored frames".format(
            float((~table.ball_visible)[labels != 3].mean())))
        print_rows(match.name, rows, status)
        report["matches"][match_id] = {"rows": rows, "evidence": status}

    out = paths.cache_dir() / "external" / "reports" / "score.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
