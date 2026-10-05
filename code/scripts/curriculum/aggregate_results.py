"""Aggregate the multi-seed curriculum results into a condition comparison.

Reads scripts/curriculum/results/<condition>_seed<N>.json (each has a per-checkpoint `curve`
plus headline fields) and reports, per condition, across seeds:
  * env-steps to reach the target held-out success  (HEADLINE; lower = better)
  * final held-out success at the full budget
  * wall-clock seconds (secondary; approximate under parallel runs)
with mean +/- a 95% normal-approx CI (1.96 * sd / sqrt(n)).

Because the full curve is stored, steps-to-target is recomputed here for ANY --target without
re-running (the JSON's own steps_to_target used the run-time target). A condition that never
reaches the target contributes the budget as a right-censored value (reported separately).

Usage:
  python aggregate_results.py [--target 0.30] [--results-dir results] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

CONDITION_ORDER = ["random", "oracle", "geometry", "probe", "combined",
                   "adaptive_fine", "adaptive_coarse"]


def steps_to_target(curve, target):
    """First env_steps whose held_success >= target, else None (censored)."""
    for pt in curve:
        if pt["held_success"] >= target:
            return pt["env_steps"]
    return None


def mean_ci(xs):
    """Return (mean, half-width-95) using a normal approx; half-width 0 if n<2."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n), n


def load(results_dir):
    by_cond = defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, "*_seed*.json")):
        d = json.loads(open(path).read())
        by_cond[d["condition"]].append(d)
    return by_cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--target", type=float, default=None,
                    help="override target for steps-to-target (default: each run's own target)")
    ap.add_argument("--budget", type=int, default=None,
                    help="censoring value for non-reachers (default: each run's total_steps)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    by_cond = load(args.results_dir)
    if not by_cond:
        print(f"no results found in {args.results_dir}")
        return

    any_run = next(iter(by_cond.values()))[0]
    target = args.target if args.target is not None else any_run["target"]
    budget = args.budget if args.budget is not None else any_run["total_steps"]

    print(f"target held-out success = {target}   budget = {budget:,} steps")
    print(f"{'condition':16s} {'seeds':>5s}  {'reached':>7s}  "
          f"{'steps->target (mean+/-95)':>28s}  {'final success':>20s}  {'wall_s':>10s}")
    print("-" * 100)

    rows = []
    for cond in CONDITION_ORDER + [c for c in by_cond if c not in CONDITION_ORDER]:
        if cond not in by_cond:
            continue
        runs = by_cond[cond]
        stt = [steps_to_target(r["curve"], target) for r in runs]
        reached = [s for s in stt if s is not None]
        # for the headline, censor non-reachers at budget so the mean is comparable
        stt_censored = [s if s is not None else budget for s in stt]
        finals = [r["curve"][-1]["held_success"] for r in runs]
        walls = [r.get("wall_s", float("nan")) for r in runs]

        m_s, ci_s, n_s = mean_ci(stt_censored)
        m_f, ci_f, _ = mean_ci(finals)
        m_w, _, _ = mean_ci(walls)
        print(f"{cond:16s} {len(runs):5d}  {len(reached):3d}/{len(runs):<3d}  "
              f"{m_s:>12,.0f} +/- {ci_s:>10,.0f}  "
              f"{m_f:>8.3f} +/- {ci_f:<7.3f}  {m_w:>10,.0f}")
        rows.append(dict(condition=cond, n_seeds=len(runs), n_reached=len(reached),
                         steps_to_target_mean=round(m_s, 1), steps_to_target_ci95=round(ci_s, 1),
                         final_success_mean=round(m_f, 4), final_success_ci95=round(ci_f, 4),
                         wall_s_mean=round(m_w, 1), target=target, budget=budget))

    if "random" in by_cond:
        base = next(r for r in rows if r["condition"] == "random")["steps_to_target_mean"]
        print("-" * 100)
        print(f"(speedup vs random, higher=better)")
        for r in rows:
            if r["steps_to_target_mean"] > 0:
                print(f"  {r['condition']:16s} {base / r['steps_to_target_mean']:.2f}x")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
