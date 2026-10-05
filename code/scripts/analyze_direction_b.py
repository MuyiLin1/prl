"""Direction B analysis: does geometry predict difficulty when the objective is FIXED?

For each task independently (objective held constant), measure how well the
hand-crafted geometric features predict HC difficulty across 500 seeds.

Reports, per task:
  - Best single-feature Spearman |rho| (and which feature)
  - Multivariate within-task Random Forest, evaluated with 5-fold CV (Spearman of
    out-of-fold predictions vs. true difficulty)

Then applies the Direction B verdict thresholds.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

CSV = sys.argv[1] if len(sys.argv) > 1 else "direction_b_500seeds.csv"

NON_FEATURE = {"task_name", "seed", "hc_best_reward", "difficulty"}

df = pd.read_csv(CSV)

# Feature set = the hand-crafted geometric/structural columns (graph_*, spectral_*,
# reward_*, grid_height/width/area). Exclude the raw flattened grid_<int> pixels.
feature_cols = [
    c for c in df.columns
    if c not in NON_FEATURE and not c.startswith("grid_")
] + ["grid_height", "grid_width", "grid_area"]
feature_cols = [c for c in dict.fromkeys(feature_cols) if c in df.columns]

print(f"Loaded {len(df)} rows, {len(feature_cols)} structural features\n")

tasks = sorted(df["task_name"].unique())
results = []

for task in tasks:
    sub = df[df["task_name"] == task].copy()
    y = sub["difficulty"].values
    n = len(sub)

    # Drop constant / all-NaN features within this task
    X = sub[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    usable = [c for c in feature_cols if X[c].std() > 1e-9]
    X_use = X[usable]

    # --- Best single-feature Spearman ---
    best_feat, best_rho = None, 0.0
    for c in usable:
        rho, _ = spearmanr(X_use[c].values, y)
        if not np.isnan(rho) and abs(rho) > abs(best_rho):
            best_rho, best_feat = rho, c

    # --- Multivariate within-task RF with 5-fold CV ---
    if y.std() < 1e-9 or len(usable) == 0:
        cv_rho = float("nan")
    else:
        oof = np.zeros(n)
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in kf.split(X_use):
            rf = RandomForestRegressor(
                n_estimators=300, max_depth=None, n_jobs=-1, random_state=0
            )
            rf.fit(X_use.iloc[tr], y[tr])
            oof[te] = rf.predict(X_use.iloc[te])
        cv_rho, _ = spearmanr(oof, y)

    results.append({
        "task": task,
        "n": n,
        "difficulty_std": round(float(y.std()), 3),
        "best_feature": best_feat,
        "best_single_|rho|": round(abs(best_rho), 3),
        "rf_cv_rho": round(float(cv_rho), 3) if not np.isnan(cv_rho) else np.nan,
    })

res = pd.DataFrame(results)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
print("=" * 90)
print("WITHIN-TASK GEOMETRY -> DIFFICULTY  (objective fixed per task)")
print("=" * 90)
print(res.to_string(index=False))
print()

# --- Verdict (use the multivariate RF CV rho as the primary signal) ---
def verdict(r):
    if np.isnan(r):
        return "N/A (no variance)"
    if r >= 0.5:
        return "STRONG  (theory validated)"
    if r >= 0.3:
        return "WEAK    (partial signal)"
    return "FAIL    (pivot trigger)"

print("Per-task verdict (multivariate RF, 5-fold CV):")
for _, row in res.iterrows():
    print(f"  {row['task']:<12} rf_cv_rho={row['rf_cv_rho']!s:>6}  ->  {verdict(row['rf_cv_rho'])}")

valid = res["rf_cv_rho"].dropna()
mean_rho = valid.mean() if len(valid) else float("nan")
print(f"\nMean within-task RF CV rho across tasks: {mean_rho:.3f}")
print(f"Overall Direction B verdict: {verdict(mean_rho)}")
