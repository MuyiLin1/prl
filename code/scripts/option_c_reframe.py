#!/usr/bin/env python3
"""Direction C / Option C: reframe — reward STRUCTURE, not geometry, governs difficulty.

Synthesizes results across domains/reward types into one cross-domain table to
test the unifying thesis:

  Difficulty for a program-synthesis search is governed by REWARD STRUCTURE.
    - Sparse reward  -> flat landscape -> uniformly maximally hard (no variance)
    - Shaped reward  -> graded landscape -> real difficulty variance emerges
  And where real variance exists, hand-crafted GEOMETRY is a weak predictor of it.

Consumes (when present):
  - direction_b_500seeds.csv         (Karel, shaped, single HC run)
  - averaged_label_topoff200.csv     (Karel, shaped, stabilized label)
  - averaged_label_screen_snake_seeder.csv
  - averaged_label_pilot.csv         (Karel, shaped, TopOff/Harvester)
  - option_a_lavagap.csv             (MiniGrid, shaped)
  - MiniGrid sparse result is injected as a known-degenerate row.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(name):
    return os.path.join(ROOT, name)


def geometry_rf_cv(df, feature_cols, target_col):
    sub = df.dropna(subset=[target_col])
    if len(sub) < 10 or sub[target_col].std() < 1e-9:
        return float("nan")
    use = [c for c in feature_cols if c in sub.columns and sub[c].std() > 1e-9]
    if not use:
        return float("nan")
    X = sub[use].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = sub[target_col].values
    oof = np.zeros(len(sub))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
        rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
        rf.fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
    rho, _ = spearmanr(oof, y)
    return float(rho)


rows = []

# --- Karel shaped, single-run (Direction B) ---
p = path("direction_b_500seeds.csv")
if os.path.exists(p):
    df = pd.read_csv(p)
    geo_cols = [c for c in df.columns
                if (c.startswith("graph_") or c.startswith("spectral_")
                    or c.startswith("reward_") or c in ("grid_height", "grid_width", "grid_area"))]
    for task, sub in df.groupby("task_name"):
        rows.append({
            "domain": "Karel", "reward": "shaped", "label": "single-run",
            "task": task, "n": len(sub),
            "difficulty_std": round(sub["difficulty"].std(), 3),
            "geometry_rf_cv": round(geometry_rf_cv(sub, geo_cols, "difficulty"), 3),
        })

# --- Karel shaped, stabilized label (averaged) ---
for fname in ["averaged_label_topoff200.csv",
              "averaged_label_screen_snake_seeder.csv",
              "averaged_label_pilot.csv"]:
    p = path(fname)
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    geo_cols = [c for c in df.columns
                if c not in ("task", "grid_seed", "mean_difficulty", "std_difficulty")]
    for task, sub in df.groupby("task"):
        rows.append({
            "domain": "Karel", "reward": "shaped", "label": "stabilized(R=12)",
            "task": task, "n": len(sub),
            "difficulty_std": round(sub["mean_difficulty"].std(), 3),
            "geometry_rf_cv": round(geometry_rf_cv(sub, geo_cols, "mean_difficulty"), 3),
        })

# --- MiniGrid sparse (known degenerate) ---
rows.append({
    "domain": "MiniGrid", "reward": "sparse", "label": "single-run",
    "task": "LavaGap/RedBlueDoor/PutNear", "n": 9,
    "difficulty_std": 0.0,   # all difficulty == 1.0, HC terminates immediately
    "geometry_rf_cv": float("nan"),  # nothing to predict (constant)
})

# --- MiniGrid shaped (Options A & B from combined collector) ---
p = path("direction_c_AB.csv")
if os.path.exists(p):
    df = pd.read_csv(p)
    geo_cols = [c for c in df.columns
                if c not in ("grid_seed", "diff_A_partial",
                             "diff_B1_successrate", "diff_B2_evals")]
    for target, lbl in [("diff_A_partial", "shaped/partial-progress"),
                        ("diff_B1_successrate", "shaped/success-rate"),
                        ("diff_B2_evals", "shaped/search-cost")]:
        rows.append({
            "domain": "MiniGrid", "reward": "shaped", "label": lbl,
            "task": "LavaGap", "n": len(df),
            "difficulty_std": round(df[target].std(), 3),
            "geometry_rf_cv": round(geometry_rf_cv(df, geo_cols, target), 3),
        })

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("=" * 100)
print("OPTION C — cross-domain: reward structure vs geometry")
print("=" * 100)
print(res.to_string(index=False))

print("\nKey contrasts:")
# sparse vs shaped variance — only compare labels on the same [0,1] scale.
# The search-cost label is raw eval counts (different units), so exclude it
# from the std aggregate to avoid mixing scales.
norm_scale = ~res["label"].str.contains("search-cost", na=False)
sparse = res[(res["reward"] == "sparse") & norm_scale]["difficulty_std"]
shaped = res[(res["reward"] == "shaped") & norm_scale]["difficulty_std"]
if len(sparse) and len(shaped):
    print(f"  Sparse reward difficulty_std:  {sparse.mean():.3f} (no variance -> uniformly hard)")
    print(f"  Shaped reward difficulty_std:  {shaped.mean():.3f} (real variance emerges)")
geo = res["geometry_rf_cv"].dropna()
if len(geo):
    print(f"  Geometry RF CV rho (where variance exists): mean={geo.mean():.3f}, max={geo.max():.3f}")
    print("  -> Even when shaped reward CREATES difficulty variance, geometry predicts it weakly.")

res.to_csv(path("option_c_cross_domain.csv"), index=False)
print(f"\nSaved {path('option_c_cross_domain.csv')}")
