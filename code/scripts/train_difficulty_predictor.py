#!/usr/bin/env python3
"""
Train and evaluate difficulty predictors using the multi-seed dataset.

Direction B: Raw grid pixels → model (treating the flattened grid as features)
Direction C: Hand-crafted features → PCA → tree model

Both use stratified train/test/val splits ensuring each task type appears in all splits.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


def load_dataset(path):
    """Load the multi-seed dataset."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples from {df['task_name'].nunique()} task types")
    print(f"Task distribution:\n{df['task_name'].value_counts().to_string()}")
    return df


def stratified_split(df, train_frac=0.7, test_frac=0.2, val_frac=0.1, random_state=42):
    """
    Split data ensuring each task type is represented in train/test/val.
    Uses per-task-type splitting to ensure balance.
    """
    np.random.seed(random_state)
    
    train_idx, test_idx, val_idx = [], [], []
    
    for task_name, group in df.groupby('task_name'):
        n = len(group)
        indices = group.index.values.copy()
        np.random.shuffle(indices)
        
        n_train = max(1, int(n * train_frac))
        n_test = max(1, int(n * test_frac))
        n_val = max(1, n - n_train - n_test)
        
        train_idx.extend(indices[:n_train])
        test_idx.extend(indices[n_train:n_train + n_test])
        val_idx.extend(indices[n_train + n_test:])
    
    return train_idx, test_idx, val_idx


def get_feature_columns(df):
    """Get hand-crafted feature columns."""
    grid_cols = [c for c in df.columns if c.startswith('grid_') and c != 'grid_height' and c != 'grid_width' and c != 'grid_area']
    # Actually grid_0, grid_1, ... are the flattened grid
    raw_grid_cols = [c for c in df.columns if c.startswith('grid_') and c[5:].isdigit()]
    
    handcrafted_cols = [
        'grid_height', 'grid_width', 'grid_area',
        'graph_num_free_cells', 'graph_num_wall_cells', 'graph_wall_fraction',
        'graph_num_edges', 'graph_num_components', 'graph_largest_component_frac',
        'graph_mean_degree', 'graph_min_degree', 'graph_max_degree',
        'graph_num_deadends', 'graph_num_corridor_cells', 'graph_num_branch_cells',
        'graph_diameter', 'graph_mean_pair_distance',
        'graph_start_reachable_frac', 'graph_start_eccentricity',
        'graph_num_markers',
        'spectral_lambda_0', 'spectral_lambda_1', 'spectral_lambda_2',
        'spectral_lambda_3', 'spectral_lambda_4', 'spectral_lambda_5',
        'reward_pick_markers', 'reward_put_markers', 'reward_reach_goal',
        'reward_visit_cells', 'reward_sparse', 'reward_crash_penalty_used',
    ]
    # Only include columns that exist
    handcrafted_cols = [c for c in handcrafted_cols if c in df.columns]
    raw_grid_cols = [c for c in raw_grid_cols if c in df.columns]
    
    return handcrafted_cols, raw_grid_cols


def evaluate_model(y_true, y_pred, label):
    """Compute evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = spearmanr(y_true, y_pred)
    
    print(f"\n  {label}:")
    print(f"    MSE:        {mse:.4f}")
    print(f"    R²:         {r2:.4f}")
    print(f"    Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"    Spearman ρ: {spearman_rho:.4f} (p={spearman_p:.4f})")
    
    return {
        'label': label, 'mse': mse, 'r2': r2,
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_rho': spearman_rho, 'spearman_p': spearman_p,
    }


def direction_c(df, train_idx, test_idx, val_idx, handcrafted_cols, target_col='difficulty'):
    """Direction C: Hand-crafted features → PCA → Gradient Boosting."""
    print("\n" + "="*70)
    print("DIRECTION C: Hand-crafted features → PCA → Gradient Boosting")
    print("="*70)
    
    X_train = df.loc[train_idx, handcrafted_cols].values
    X_test = df.loc[test_idx, handcrafted_cols].values
    X_val = df.loc[val_idx, handcrafted_cols].values
    y_train = df.loc[train_idx, target_col].values
    y_test = df.loc[test_idx, target_col].values
    y_val = df.loc[val_idx, target_col].values
    
    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_val_s = scaler.transform(X_val)
    
    # PCA to reduce to 8 components
    n_components = min(8, X_train_s.shape[1], X_train_s.shape[0])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    X_val_pca = pca.transform(X_val_s)
    
    print(f"  PCA: {X_train_s.shape[1]} features → {n_components} components")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Also try without PCA for comparison
    results = []
    
    # Model 1: PCA + GradientBoosting
    model_pca = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_pca.fit(X_train_pca, y_train)
    
    pred_test = model_pca.predict(X_test_pca)
    pred_val = model_pca.predict(X_val_pca)
    
    results.append(evaluate_model(y_test, pred_test, "C1: PCA(8) + GBR [test]"))
    results.append(evaluate_model(y_val, pred_val, "C1: PCA(8) + GBR [val]"))
    
    # Model 2: Full features + RandomForest
    model_rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model_rf.fit(X_train_s, y_train)
    
    pred_test_rf = model_rf.predict(X_test_s)
    pred_val_rf = model_rf.predict(X_val_s)
    
    results.append(evaluate_model(y_test, pred_test_rf, "C2: Full features + RF [test]"))
    results.append(evaluate_model(y_val, pred_val_rf, "C2: Full features + RF [val]"))
    
    # Feature importance
    print("\n  Top 10 feature importances (RF):")
    importances = model_rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for i in top_idx:
        print(f"    {handcrafted_cols[i]:30s} {importances[i]:.4f}")
    
    return results


def direction_b(df, train_idx, test_idx, val_idx, raw_grid_cols, target_col='difficulty'):
    """Direction B: Raw grid pixels → model."""
    print("\n" + "="*70)
    print("DIRECTION B: Raw grid (flattened) → Model")
    print("="*70)
    
    if not raw_grid_cols:
        print("  ERROR: No grid columns found in dataset. Run generate_multiseed_dataset.py first.")
        return []
    
    X_train = df.loc[train_idx, raw_grid_cols].values
    X_test = df.loc[test_idx, raw_grid_cols].values
    X_val = df.loc[val_idx, raw_grid_cols].values
    y_train = df.loc[train_idx, target_col].values
    y_test = df.loc[test_idx, target_col].values
    y_val = df.loc[val_idx, target_col].values
    
    print(f"  Raw grid input dimension: {X_train.shape[1]}")
    
    results = []
    
    # Model 1: PCA on raw grid + GBR
    n_components = min(16, X_train.shape[1], X_train.shape[0] - 1)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_val_pca = pca.transform(X_val)
    
    print(f"  PCA: {X_train.shape[1]} grid pixels → {n_components} components")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    model_gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_gbr.fit(X_train_pca, y_train)
    
    pred_test = model_gbr.predict(X_test_pca)
    pred_val = model_gbr.predict(X_val_pca)
    
    results.append(evaluate_model(y_test, pred_test, "B1: Grid PCA(16) + GBR [test]"))
    results.append(evaluate_model(y_val, pred_val, "B1: Grid PCA(16) + GBR [val]"))
    
    # Model 2: Random Forest directly on raw grid (subsample features)
    model_rf = RandomForestRegressor(n_estimators=200, max_depth=5, 
                                      max_features=0.1,  # use 10% of features per split
                                      random_state=42)
    model_rf.fit(X_train, y_train)
    
    pred_test_rf = model_rf.predict(X_test)
    pred_val_rf = model_rf.predict(X_val)
    
    results.append(evaluate_model(y_test, pred_test_rf, "B2: Raw grid + RF [test]"))
    results.append(evaluate_model(y_val, pred_val_rf, "B2: Raw grid + RF [val]"))
    
    return results


def direction_b_plus_c(df, train_idx, test_idx, val_idx, handcrafted_cols, raw_grid_cols, target_col='difficulty'):
    """Combined: hand-crafted + grid PCA features together."""
    print("\n" + "="*70)
    print("DIRECTION B+C: Combined (hand-crafted + grid PCA) → Model")
    print("="*70)
    
    # Get hand-crafted features (standardized)
    X_hc_train = df.loc[train_idx, handcrafted_cols].values
    X_hc_test = df.loc[test_idx, handcrafted_cols].values
    X_hc_val = df.loc[val_idx, handcrafted_cols].values
    
    scaler = StandardScaler()
    X_hc_train = scaler.fit_transform(X_hc_train)
    X_hc_test = scaler.transform(X_hc_test)
    X_hc_val = scaler.transform(X_hc_val)
    
    # Get grid PCA
    X_grid_train = df.loc[train_idx, raw_grid_cols].values
    X_grid_test = df.loc[test_idx, raw_grid_cols].values
    X_grid_val = df.loc[val_idx, raw_grid_cols].values
    
    n_components = min(8, X_grid_train.shape[1], X_grid_train.shape[0] - 1)
    pca = PCA(n_components=n_components)
    X_grid_train_pca = pca.fit_transform(X_grid_train)
    X_grid_test_pca = pca.transform(X_grid_test)
    X_grid_val_pca = pca.transform(X_grid_val)
    
    # Combine
    X_train = np.hstack([X_hc_train, X_grid_train_pca])
    X_test = np.hstack([X_hc_test, X_grid_test_pca])
    X_val = np.hstack([X_hc_val, X_grid_val_pca])
    
    y_train = df.loc[train_idx, target_col].values
    y_test = df.loc[test_idx, target_col].values
    y_val = df.loc[val_idx, target_col].values
    
    print(f"  Combined features: {X_hc_train.shape[1]} hand-crafted + {n_components} grid PCA = {X_train.shape[1]} total")
    
    model = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    pred_test = model.predict(X_test)
    pred_val = model.predict(X_val)
    
    results = []
    results.append(evaluate_model(y_test, pred_test, "B+C: Combined + GBR [test]"))
    results.append(evaluate_model(y_val, pred_val, "B+C: Combined + GBR [val]"))
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='multiseed_dataset.csv')
    parser.add_argument('--target', default='difficulty', choices=['difficulty', 'hc_best_reward'])
    args = parser.parse_args()
    
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.dataset)
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run generate_multiseed_dataset.py first.")
        sys.exit(1)
    
    df = load_dataset(dataset_path)
    
    # Get column groups
    handcrafted_cols, raw_grid_cols = get_feature_columns(df)
    print(f"\nHand-crafted features: {len(handcrafted_cols)}")
    print(f"Raw grid features: {len(raw_grid_cols)}")
    
    # Split
    train_idx, test_idx, val_idx = stratified_split(df)
    print(f"\nSplit: train={len(train_idx)}, test={len(test_idx)}, val={len(val_idx)}")
    
    # Check target distribution
    print(f"\nTarget '{args.target}' stats:")
    print(f"  Train: mean={df.loc[train_idx, args.target].mean():.3f}, std={df.loc[train_idx, args.target].std():.3f}")
    print(f"  Test:  mean={df.loc[test_idx, args.target].mean():.3f}, std={df.loc[test_idx, args.target].std():.3f}")
    
    all_results = []
    
    # Direction C
    results_c = direction_c(df, train_idx, test_idx, val_idx, handcrafted_cols, args.target)
    all_results.extend(results_c)
    
    # Direction B
    if raw_grid_cols:
        results_b = direction_b(df, train_idx, test_idx, val_idx, raw_grid_cols, args.target)
        all_results.extend(results_b)
        
        # Combined
        results_bc = direction_b_plus_c(df, train_idx, test_idx, val_idx, 
                                         handcrafted_cols, raw_grid_cols, args.target)
        all_results.extend(results_bc)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: All Models Ranked by Test Spearman ρ")
    print("="*70)
    test_results = [r for r in all_results if '[test]' in r['label']]
    test_results.sort(key=lambda x: abs(x['spearman_rho']), reverse=True)
    
    print(f"{'Model':<40} {'Spearman ρ':>12} {'p-value':>10} {'R²':>8}")
    print("-" * 72)
    for r in test_results:
        print(f"{r['label']:<40} {r['spearman_rho']:>12.4f} {r['spearman_p']:>10.4f} {r['r2']:>8.4f}")


if __name__ == '__main__':
    main()
