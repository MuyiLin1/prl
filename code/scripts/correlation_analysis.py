#!/usr/bin/env python3
"""
Correlation analysis: compare all difficulty predictors against ground truth.

Reads:
  - ground_truth_difficulty.csv (from run_all_hc.py)
  - naive_heuristic_predictions.csv (from naive_heuristic_baseline.py)
  - llm_difficulty_predictions.csv (from llm_difficulty_baseline.py)
  - RTD leave-one-out predictions (computed inline)

Outputs:
  - correlation_results.csv: predictor → Pearson r, Spearman ρ, p-values
  - Prints a summary table

Usage:
  python scripts/correlation_analysis.py
  python scripts/correlation_analysis.py --ground-truth ground_truth_difficulty.csv
"""

import sys
import os
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(".")


def normalize_task_key(name: str) -> str:
    """Normalize task names for joining across different formats."""
    return name.lower().replace("_", "").replace("sparse", "").strip()


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load ground-truth difficulty from HC runs."""
    df = pd.read_csv(path)
    df["task_key"] = df["task_name"].apply(normalize_task_key)
    return df


def load_heuristic_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["task_key"] = df["task_name"].apply(normalize_task_key)
    return df


def load_llm_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["task_key"] = df["task_name"].apply(normalize_task_key)
    return df


def compute_rtd_leave_one_out(features_csv: str, ground_truth_df: pd.DataFrame,
                               lambda_ridge: float = 1.0) -> pd.DataFrame:
    """
    Compute RTD novelty score for each task using leave-one-out.
    For task i, use all other tasks as the 'prior' to compute RTD(i).
    Also compute Ridge-predicted difficulty via LOO cross-validation.
    """
    feat_df = pd.read_csv(features_csv)
    feat_df["task_key"] = feat_df["task_name"].apply(normalize_task_key)

    # Remove duplicates
    feat_df = feat_df.drop_duplicates(subset="task_key")

    # Merge with ground truth to get target variable
    gt_key = ground_truth_df[["task_key", "difficulty"]].copy()
    merged = feat_df.merge(gt_key, on="task_key", how="inner")

    # Feature columns (exclude metadata)
    feature_cols = [c for c in feat_df.columns if c not in ("task_name", "task_key")]
    X = merged[feature_cols].values.astype(float)
    y = merged["difficulty"].values.astype(float)
    task_names = merged["task_name"].values

    n, d = X.shape

    # Standardize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma

    rtd_scores = []
    ridge_predictions = []

    for i in range(n):
        # Leave one out
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = Xs[mask]
        y_train = y[mask]
        x_test = Xs[i]

        # RTD: precision matrix from training set
        V = lambda_ridge * np.eye(d) + X_train.T @ X_train
        V_inv = np.linalg.pinv(V)
        rtd = float(np.sqrt(max(0.0, x_test.T @ V_inv @ x_test)))
        rtd_scores.append(rtd)

        # Ridge regression prediction
        X_aug = np.column_stack([np.ones(X_train.shape[0]), X_train])
        x_aug = np.concatenate([[1.0], x_test])
        A = X_aug.T @ X_aug + lambda_ridge * np.eye(d + 1)
        A[0, 0] -= lambda_ridge  # don't regularize intercept
        coef = np.linalg.pinv(A) @ X_aug.T @ y_train
        pred = float(np.clip(x_aug @ coef, 0.0, 1.0))
        ridge_predictions.append(pred)

    result = pd.DataFrame({
        "task_name": task_names,
        "task_key": merged["task_key"].values,
        "rtd_novelty": rtd_scores,
        "rtd_ridge_predicted_difficulty": ridge_predictions,
    })
    return result


def compute_correlations(ground_truth: np.ndarray, predictions: np.ndarray,
                         name: str) -> dict:
    """Compute Pearson and Spearman correlations."""
    # Remove NaN pairs
    valid = ~(np.isnan(ground_truth) | np.isnan(predictions))
    if valid.sum() < 3:
        return {
            "predictor": name,
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "n_tasks": int(valid.sum()),
        }

    gt = ground_truth[valid]
    pred = predictions[valid]

    pearson_r, pearson_p = stats.pearsonr(gt, pred)
    spearman_rho, spearman_p = stats.spearmanr(gt, pred)

    return {
        "predictor": name,
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 6),
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p": round(spearman_p, 6),
        "n_tasks": int(valid.sum()),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--ground-truth", default="ground_truth_difficulty.csv")
    parser.add_argument("--heuristic", default="naive_heuristic_predictions.csv")
    parser.add_argument("--llm", default="llm_difficulty_predictions.csv")
    parser.add_argument("--features-csv", default="prog_policies/karel_tasks/all_karel_task_features.csv")
    parser.add_argument("--output", default="correlation_results.csv")
    args = parser.parse_args()

    # Check what files exist
    gt_exists = Path(args.ground_truth).exists()
    heur_exists = Path(args.heuristic).exists()
    llm_exists = Path(args.llm).exists()

    if not gt_exists:
        print(f"ERROR: Ground truth file not found: {args.ground_truth}")
        print("Run: python scripts/run_all_hc.py first")
        sys.exit(1)

    gt_df = load_ground_truth(args.ground_truth)
    print(f"Ground truth: {len(gt_df)} tasks loaded")
    print(f"  Difficulty range: [{gt_df['difficulty'].min():.4f}, {gt_df['difficulty'].max():.4f}]")

    # Use multiple ground truth signals if available
    gt_signals = ["difficulty"]
    if "log_mean_programs" in gt_df.columns:
        gt_signals.append("log_mean_programs")

    results = []

    # --- Naive heuristic baselines ---
    if heur_exists:
        heur_df = load_heuristic_predictions(args.heuristic)
        merged = gt_df.merge(heur_df, on="task_key", how="inner", suffixes=("", "_heur"))

        heuristic_cols = [c for c in heur_df.columns if c.startswith("heuristic_")]
        for gt_signal in gt_signals:
            for col in heuristic_cols:
                if col in merged.columns and gt_signal in merged.columns:
                    label = f"{col}" if len(gt_signals) == 1 else f"{col} vs {gt_signal}"
                    r = compute_correlations(
                        merged[gt_signal].values,
                        merged[col].values,
                        label
                    )
                    results.append(r)
    else:
        print(f"WARNING: Heuristic file not found: {args.heuristic}")
        print("  Run: python scripts/naive_heuristic_baseline.py first")

    # --- LLM baseline ---
    if llm_exists:
        llm_df = load_llm_predictions(args.llm)
        merged = gt_df.merge(llm_df, on="task_key", how="inner", suffixes=("", "_llm"))
        if "llm_difficulty_score" in merged.columns:
            for gt_signal in gt_signals:
                if gt_signal in merged.columns:
                    label = "llm_zero_shot" if len(gt_signals) == 1 else f"llm_zero_shot vs {gt_signal}"
                    r = compute_correlations(
                        merged[gt_signal].values,
                        merged["llm_difficulty_score"].values,
                        label
                    )
                    results.append(r)
    else:
        print(f"WARNING: LLM file not found: {args.llm}")
        print("  Run: python scripts/llm_difficulty_baseline.py first")

    # --- RTD (leave-one-out) ---
    if Path(args.features_csv).exists():
        rtd_df = compute_rtd_leave_one_out(args.features_csv, gt_df)
        merged = gt_df.merge(rtd_df, on="task_key", how="inner", suffixes=("", "_rtd"))

        for gt_signal in gt_signals:
            if gt_signal in merged.columns:
                # RTD novelty as difficulty predictor
                label = "rtd_novelty" if len(gt_signals) == 1 else f"rtd_novelty vs {gt_signal}"
                r = compute_correlations(
                    merged[gt_signal].values,
                    merged["rtd_novelty"].values,
                    label
                )
                results.append(r)

                # RTD Ridge-predicted difficulty
                label = "rtd_ridge_prediction" if len(gt_signals) == 1 else f"rtd_ridge_prediction vs {gt_signal}"
                r = compute_correlations(
                    merged[gt_signal].values,
                    merged["rtd_ridge_predicted_difficulty"].values,
                    label
                )
                results.append(r)

    # --- Output ---
    if not results:
        print("No results to report. Run the baseline scripts first.")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("spearman_rho", ascending=False, key=abs)
    results_df.to_csv(args.output, index=False)

    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS: Predictor vs Ground-Truth Difficulty")
    print(f"{'='*70}")
    print(f"\n{'Predictor':<35} {'Pearson r':>10} {'p-val':>10} {'Spearman ρ':>12} {'p-val':>10} {'N':>4}")
    print("-" * 85)
    for _, row in results_df.iterrows():
        print(f"{row['predictor']:<35} {row['pearson_r']:>10.4f} {row['pearson_p']:>10.4f} "
              f"{row['spearman_rho']:>12.4f} {row['spearman_p']:>10.4f} {row['n_tasks']:>4}")
    print("-" * 85)

    # Highlight best
    best = results_df.iloc[0]
    print(f"\nBest predictor: {best['predictor']} (Spearman ρ = {best['spearman_rho']:.4f})")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
