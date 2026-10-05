#!/usr/bin/env python3
"""
Naive heuristic baselines for Karel task difficulty prediction.

Predicts difficulty using simple, hand-picked features (no ML model).
These serve as "dumb" baselines that the RTD predictor should beat.

Baselines:
  1. wall_fraction: More walls = harder
  2. dead_end_ratio: More dead ends relative to free cells = harder
  3. diameter_normalized: Larger graph diameter relative to grid = harder
  4. inverse_fiedler: Smaller algebraic connectivity = more bottlenecks = harder
  5. subgoal_density: More subgoals relative to free cells = harder
  6. composite_simple: Weighted combo of wall_fraction + dead_end_ratio + 1/fiedler

Usage:
  python scripts/naive_heuristic_baseline.py \
    --features-csv prog_policies/karel_tasks/all_karel_task_features.csv \
    --output naive_heuristic_predictions.csv
"""

import csv
import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def compute_heuristic_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute multiple naive heuristic difficulty scores."""
    results = pd.DataFrame()
    results["task_name"] = df["task_name"]

    # 1. Wall fraction (directly from features)
    results["heuristic_wall_fraction"] = df["graph_wall_fraction"]

    # 2. Dead-end ratio: dead ends / free cells
    results["heuristic_dead_end_ratio"] = (
        df["graph_num_deadends"] / df["graph_num_free_cells"].clip(lower=1)
    )

    # 3. Diameter normalized by grid perimeter
    grid_perimeter = 2 * (df["grid_height"] + df["grid_width"])
    results["heuristic_diameter_norm"] = df["graph_diameter"] / grid_perimeter.clip(lower=1)

    # 4. Inverse Fiedler value (algebraic connectivity)
    # Small lambda_1 = weak connectivity = harder navigation
    fiedler = df["spectral_lambda_1"].clip(lower=1e-10)
    results["heuristic_inverse_fiedler"] = 1.0 / fiedler
    # Normalize to [0, 1] for comparability
    inv_f = results["heuristic_inverse_fiedler"]
    inv_range = inv_f.max() - inv_f.min()
    if inv_range < 1e-10:
        inv_range = 1.0
    results["heuristic_inverse_fiedler_norm"] = (inv_f - inv_f.min()) / inv_range

    # 5. Subgoal density: estimated subgoals / free cells
    results["heuristic_subgoal_density"] = (
        df["reward_num_subgoals_est"] / df["graph_num_free_cells"].clip(lower=1)
    )

    # 6. Composite: weighted average of normalized features
    # Normalize each component to [0, 1] first
    def normalize(s):
        s_range = s.max() - s.min()
        if s_range < 1e-10:
            s_range = 1.0
        return (s - s.min()) / s_range

    wall_n = normalize(df["graph_wall_fraction"])
    dead_n = normalize(df["graph_num_deadends"] / df["graph_num_free_cells"].clip(lower=1))
    diam_n = normalize(df["graph_diameter"] / grid_perimeter.clip(lower=1))
    fiedler_n = normalize(1.0 / fiedler)
    subgoal_n = normalize(df["reward_num_subgoals_est"] / df["graph_num_free_cells"].clip(lower=1))

    # Equal weights
    results["heuristic_composite"] = (
        0.2 * wall_n + 0.2 * dead_n + 0.2 * diam_n + 0.2 * fiedler_n + 0.2 * subgoal_n
    )

    # 7. Graph complexity: bridges + articulation points normalized
    graph_complexity = (df["graph_num_bridges"] + df["graph_num_articulation_points"])
    results["heuristic_graph_complexity"] = normalize(
        graph_complexity / df["graph_num_free_cells"].clip(lower=1)
    )

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--features-csv",
        default="prog_policies/karel_tasks/all_karel_task_features.csv",
        help="CSV with pre-computed task features",
    )
    parser.add_argument("--output", default="naive_heuristic_predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)

    # Remove duplicate rows (e.g., CLEANHOUSE vs clean_house)
    df["task_lower"] = df["task_name"].str.lower().str.replace("_", "")
    df = df.drop_duplicates(subset="task_lower").drop(columns="task_lower")

    results = compute_heuristic_scores(df)
    results.to_csv(args.output, index=False)

    print(f"Wrote {args.output}")
    print(f"\nHeuristic predictions ({len(results)} tasks):")
    print(results.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
