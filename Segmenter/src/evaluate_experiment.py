#!/usr/bin/env python3

"""
Aggregates per-seed metrics from a SLURM hyperparameter experiment and prints
a ranked summary table.

Each run directory is expected to contain a results_*.json file written by
evaluate.py. Runs are grouped by hyperparameter config (seed excluded), and
mean ± std are computed across seeds for every metric.

Usage:
    python src/evaluate_experiment.py results/experiment_XXXXX

    # Sort by WindowDiff (lower is better) and write CSV summaries:
    python src/evaluate_experiment.py results/experiment_XXXXX \\
        --sort window_diff --sort-asc --csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Optional


METRICS = [
    "b_edu_f1",
    "b_edu_precision",
    "b_edu_recall",
    "exact_match",
    "window_diff",
]


def load_results(run_dir: Path) -> Optional[dict]:
    """Return the parsed results dict from run_dir, or None if no results file exists."""
    candidates = sorted(run_dir.glob("results_*.json"))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"  [warn] multiple results files in {run_dir.name}, using latest: {candidates[-1].name}")
    with open(candidates[-1]) as f:
        return json.load(f)


def parse_hparams(run_name: str) -> dict:
    """Parse lr1e-5_ep10_wd0.001_se42 style names into a dict."""
    hparams = {}
    for part in run_name.split("_"):
        for prefix in ("lr", "ep", "wd", "se"):
            if part.startswith(prefix):
                hparams[prefix] = part[len(prefix):]
    return hparams


def hparam_key(run_name: str) -> str:
    """Return a grouping key that excludes the seed (se)."""
    parts = run_name.split("_")
    return "_".join(p for p in parts if not p.startswith("se"))


def main():
    parser = argparse.ArgumentParser(
        description="Summarise results of an experiment, averaging over seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "exp_dir", 
        type=Path, 
        help="Path to experiment_slurmjobid directory."
    )

    parser.add_argument(
        "--sort", 
        default="b_edu_f1", 
        metavar="METRIC",
        help="Metric to sort runs by (default: b_edu_f1).",
    )

    parser.add_argument(
        "--sort-asc", 
        action="store_true",
        help="Sort ascending instead of descending (use for window_diff).",
    )

    parser.add_argument(
        "--csv", action="store_true",
        help="Write a summary CSV next to the experiment directory.",
    )
    
    args = parser.parse_args()

    exp_dir: Path = args.exp_dir.resolve()
    if not exp_dir.is_dir():
        sys.exit(f"Error: {exp_dir} is not a directory.")

    sort_metric = args.sort
    if sort_metric not in METRICS:
        sys.exit(f"Error: unknown metric '{sort_metric}'. Choose from: {', '.join(METRICS)}")

    # groups: key -> list of (run_name, results_dict)
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    missing = []

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        results = load_results(run_dir)
        if results is None:
            missing.append(run_dir.name)
            continue
        key = hparam_key(run_dir.name)
        groups[key].append((run_dir.name, results))

    if not groups:
        sys.exit("No results found.")

    if missing:
        print(f"[warn] no results file found in: {', '.join(missing)}\n")

    agg_rows = []

    for key, runs in groups.items():
        hparams = parse_hparams(key)           # excludes seed
        seeds = [parse_hparams(name).get("se", "?") for name, _ in runs]
        n = len(runs)

        row: dict = {
            "config": key,
            "lr": hparams.get("lr", ""),
            "ep": hparams.get("ep", ""),
            "wd": hparams.get("wd", ""),
            "n_seeds": n,
            "seeds": ",".join(sorted(seeds)),
        }

        for m in METRICS:
            values = [r.get(m, float("nan")) for _, r in runs]
            values = [v for v in values if v == v]   # drop NaN
            row[f"{m}_mean"] = mean(values) if values else float("nan")
            row[f"{m}_std"]  = stdev(values) if len(values) > 1 else 0.0

        agg_rows.append(row)

    agg_rows.sort(key=lambda r: r[f"{sort_metric}_mean"], reverse=not args.sort_asc)

    best_run_name, best_run_score = None, -float("inf")
    for _, runs in groups.items():
        for run_name, results in runs:
            score = results.get(sort_metric, float("nan"))
            if score == score and score > best_run_score:  # score == score is False for NaN
                best_run_score = score
                best_run_name = run_name

    best_agg = agg_rows[0]

    print(f"\nExperiment : {exp_dir.name}")
    print(f"Configs    : {len(agg_rows)}   |   total runs : {sum(r['n_seeds'] for r in agg_rows)}")
    print(f"Sorted by  : {sort_metric} (mean)\n")

    col_w = max(len(r["config"]) for r in agg_rows) + 2
    metric_col_w = 24

    header = f"{'config':<{col_w}}"
    for m in METRICS:
        header += f"  {m:^{metric_col_w}}"
    print(header)
    print("-" * (col_w + len(METRICS) * (metric_col_w + 2)))

    for row in agg_rows:
        marker = " ◀ best" if row is best_agg else ""
        line = f"{row['config']:<{col_w}}"
        for m in METRICS:
            cell = f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}"
            line += f"  {cell:^{metric_col_w}}"
        print(line + marker)

    print(f"\n{'─'*60}")
    print(f"Best config ({sort_metric}): {best_agg['config']}")
    for m in METRICS:
        print(f"  {m:>22} : {best_agg[f'{m}_mean']:.4f} ± {best_agg[f'{m}_std']:.4f}")

    print(f"\n{'─'*60}")
    print(f"Best individual run ({sort_metric} = {best_run_score:.4f}): {best_run_name}")

    if args.csv:
        csv_path = exp_dir.parent / f"{exp_dir.name}_summary.csv"

        # Paper-friendly columns: config | metric mean ± std ...
        fieldnames = ["config", "lr", "ep", "wd", "n_seeds", "seeds"]
        for m in METRICS:
            fieldnames += [f"{m}_mean", f"{m}_std"]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(agg_rows)

        print(f"CSV written to : {csv_path}")

        # write a compact "mean±std" version for direct copy-paste into a paper
        compact_path = exp_dir.parent / f"{exp_dir.name}_summary_compact.csv"
        with open(compact_path, "w", newline="") as f:
            compact_fields = ["config"] + METRICS
            writer = csv.DictWriter(f, fieldnames=compact_fields)
            writer.writeheader()
            for row in agg_rows:
                compact_row = {"config": row["config"]}
                for m in METRICS:
                    compact_row[m] = f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}"
                writer.writerow(compact_row)

        print(f"Compact CSV written to : {compact_path}")


if __name__ == "__main__":
    main()