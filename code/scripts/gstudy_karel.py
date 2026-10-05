"""Table tab:karel: single-run signal (one-way ICC), and search runs needed for reliability 0.7.

One-way random-effects ANOVA (grid vs search seed) from per-grid mean/SD over R=12 runs.
Usage: python scripts/gstudy_karel.py [results_dir]   (default: the export's results/01_karel)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = 12
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results/01_karel"
files = sorted((D / "full_screen_R12").glob("karel_*_grids.csv")) + [
    D / "averaged_label_topoff200.csv", D / "averaged_label_screen_snake_seeder.csv", D / "averaged_label_pilot.csv"]

seen, rows = set(), []
for f in files:
    df = pd.read_csv(f)
    for task, g in df.groupby("task"):
        if task in seen:
            continue
        seen.add(task)
        msb = R * g["mean_difficulty"].var(ddof=1)
        msw = (g["std_difficulty"] ** 2).mean()
        icc = max((msb - msw) / (msb + (R - 1) * msw), 1e-9)
        rows.append(dict(task=task, grids=len(g), single_run_icc=round(icc, 3),
                         reliability_R12=round(R * icc / (1 + (R - 1) * icc), 2),
                         runs_for_0_7=int(np.rint(0.7 / 0.3 * (1 - icc) / icc)),
                         runs_for_0_8=int(np.rint(0.8 / 0.2 * (1 - icc) / icc))))
out = pd.DataFrame(rows).sort_values("single_run_icc", ascending=False)
print(out.to_string(index=False))
