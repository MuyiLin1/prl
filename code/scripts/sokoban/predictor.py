#!/usr/bin/env python3
"""
Sokoban predictor evaluation (paper section 6.5): compare the static geometric
descriptor against the cheap dynamics-aware probe at predicting the budgeted-solver
difficulty label, using the same protocol as Karel/MiniGrid:

  * label reliability rho_rel (read from the fast-path run / recomputed),
  * best single descriptor correlation max_j |rho_s(g_j, D)|,
  * multivariate 5-fold cross-validated Spearman of a RandomForest regressor,

for (a) geometry features and (b) cheap probe features. Reads the labels CSV
written by sample_and_label.py, computes probe features on the same levels, and
prints a table-ready summary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from sokoban_core import iter_levels_in_file
from probe import probe_features

_ROOT = Path(__file__).resolve().parents[2]

GEO_EXCLUDE = {"level_id", "tier", "difficulty", "mean_reward", "solved_frac"}


def load_labels(csv_path: Path) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
        feat_keys = [k for k in reader.fieldnames if k not in GEO_EXCLUDE]
    return rows, feat_keys


def cv_spearman(X: np.ndarray, y: np.ndarray, seed: int = 0, n_splits: int = 5) -> float:
    if X.shape[0] < n_splits or np.std(y) < 1e-9:
        return float("nan")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        preds[te] = rf.predict(X[te])
    return float(spearmanr(preds, y).correlation)


def best_single(X: np.ndarray, y: np.ndarray, names: List[str]) -> Tuple[str, float]:
    best_name, best_rho = "", 0.0
    for j, name in enumerate(names):
        col = X[:, j]
        if np.std(col) < 1e-9:
            continue
        rho = spearmanr(col, y).correlation
        if not np.isnan(rho) and abs(rho) > abs(best_rho):
            best_name, best_rho = name, rho
    return best_name, best_rho


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(_ROOT / "data/sokoban_fastpath_labels.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, geo_keys = load_labels(Path(args.labels))
    y = np.array([float(r["difficulty"]) for r in rows], dtype=float)

    # geometry matrix
    Xg = np.array([[float(r[k]) for k in geo_keys] for r in rows], dtype=float)

    # probe features: recompute on the same levels (cheap)
    # map level_id -> level object
    level_cache: Dict[str, object] = {}
    files_needed = set()
    for r in rows:
        # level_id = "tier/filestem/idx"
        files_needed.add((r["tier"], r["level_id"]))
    # build by iterating each unique file once
    by_file: Dict[Tuple[str, str], List[str]] = {}
    for r in rows:
        tier, lid = r["tier"], r["level_id"]
        parts = lid.split("/")
        filestem = parts[1] if len(parts) >= 3 else parts[0]
        by_file.setdefault((tier, filestem), []).append(lid)

    probe_rows: Dict[str, Dict[str, float]] = {}
    tier_dir = {
        "unfiltered": _ROOT / "data/boxoban/unfiltered/train",
        "medium": _ROOT / "data/boxoban/medium/train",
        "hard": _ROOT / "data/boxoban/hard",
    }
    for (tier, filestem), lids in by_file.items():
        fpath = tier_dir[tier] / f"{filestem}.txt"
        for lvl in iter_levels_in_file(fpath, tier=tier):
            if lvl.level_id in lids:
                probe_rows[lvl.level_id] = probe_features(lvl)

    probe_keys = sorted(next(iter(probe_rows.values())).keys())
    Xp = np.array([[probe_rows[r["level_id"]][k] for k in probe_keys] for r in rows],
                  dtype=float)

    # ---- evaluate ----
    g_name, g_rho = best_single(Xg, y, geo_keys)
    p_name, p_rho = best_single(Xp, y, probe_keys)
    g_cv = cv_spearman(Xg, y, seed=args.seed)
    p_cv = cv_spearman(Xp, y, seed=args.seed)
    gp_cv = cv_spearman(np.hstack([Xg, Xp]), y, seed=args.seed)

    print("=" * 64)
    print(f"SOKOBAN PREDICTOR EVALUATION  ({len(rows)} levels)")
    print("=" * 64)
    print(f"{'':24s}{'best single rho_s':>20s}{'  RF 5-fold CV rho':>20s}")
    print(f"{'geometry':24s}{g_name+': '+f'{g_rho:+.3f}':>20s}{g_cv:>20.3f}")
    print(f"{'cheap probe':24s}{p_name+': '+f'{p_rho:+.3f}':>20s}{p_cv:>20.3f}")
    print(f"{'geometry + probe':24s}{'':>20s}{gp_cv:>20.3f}")


if __name__ == "__main__":
    main()
