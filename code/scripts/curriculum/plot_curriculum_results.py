#!/usr/bin/env python3
"""Generate paper figures for the neural-curriculum LavaCrossing experiment.

Reads the per-seed result JSONs in ``results_oraclefree_lc91317/`` and emits
vector PDF figures into the repository ``images/`` directory (next to paper.tex
via \\graphicspath{{images/}}):

  F1  curriculum_main.pdf    bar chart: final held-out success +/- 95% CI
  F2  curriculum_curves.pdf  learning curves: held-out success vs env-steps
  F3  curriculum_seeds.pdf   per-seed strip plot of final success
  F4  curriculum_cells.pdf   per-cell (size x tier) held-out success heatmaps

It also prints an exact summary table to stdout so the LaTeX tables can be
populated with numbers that match the figures bit-for-bit.

Run:
    /Users/linmuyi/code/offline-rl-experiments/.venv/bin/python \
        scripts/curriculum/plot_curriculum_results.py
"""
from __future__ import annotations

import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results_oraclefree_lc91317")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IMAGES_DIR = os.path.join(REPO_ROOT, "images")

TARGET = 0.30  # held-out success target used by the run

# display order, pretty label, color
CONDITIONS = [
    ("random", "Random\n(flat)", "#c0392b"),
    ("oracle", "Oracle\n(expensive)", "#34495e"),
    ("geometry", "Geometry\n(cheap)", "#27ae60"),
    ("probe", "Probe", "#95a5a6"),
    ("combined", "Combined", "#2980b9"),
    ("adaptive_fine", "Adaptive\nfine", "#e67e22"),
    ("adaptive_coarse", "Adaptive\ncoarse", "#16a085"),
]


def ci95(x: np.ndarray) -> float:
    """Normal-approx 95% CI half-width (matches aggregate_results.py)."""
    n = len(x)
    if n < 2:
        return 0.0
    return 1.96 * float(np.std(x, ddof=1)) / math.sqrt(n)


def load_condition(cond: str):
    """Return list of result dicts for a condition, sorted by seed."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{cond}_seed*.json")))
    runs = [json.load(open(f)) for f in files]
    runs.sort(key=lambda r: r.get("seed", 0))
    return runs


def gather():
    data = {}
    for cond, _, _ in CONDITIONS:
        runs = load_condition(cond)
        if not runs:
            continue
        finals = np.array([r["curve"][-1]["held_success"] for r in runs], dtype=float)
        reached = sum(1 for r in runs if r.get("steps_to_target") is not None)
        ended = int(np.sum(finals >= TARGET))
        data[cond] = {
            "runs": runs,
            "finals": finals,
            "mean": float(np.mean(finals)),
            "ci": ci95(finals),
            "median": float(np.median(finals)),
            "reached": reached,
            "ended": ended,
            "n": len(runs),
            "wall": float(np.mean([r.get("wall_s", float("nan")) for r in runs])),
        }
    return data


def print_summary(data):
    print("\n=== Per-condition summary (target = %.2f) ===" % TARGET)
    hdr = f"{'condition':<16}{'n':>3}{'reached':>9}{'ended>=t':>9}"
    hdr += f"{'mean_final':>12}{'ci95':>8}{'median':>8}{'wall_s':>9}"
    print(hdr)
    for cond, _, _ in CONDITIONS:
        d = data.get(cond)
        if not d:
            continue
        print(
            f"{cond:<16}{d['n']:>3}{d['reached']:>6}/{d['n']:<2}"
            f"{d['ended']:>6}/{d['n']:<2}"
            f"{d['mean']:>12.3f}{d['ci']:>8.3f}{d['median']:>8.3f}{d['wall']:>9.0f}"
        )
    # per-seed matrix (for optional appendix table / sanity)
    print("\n=== Per-seed final held-out success ===")
    seeds = sorted({r.get("seed", 0) for cond, _, _ in CONDITIONS
                    for r in data.get(cond, {}).get("runs", [])})
    print(f"{'condition':<16}" + "".join(f"s{s:<5}" for s in seeds))
    for cond, _, _ in CONDITIONS:
        d = data.get(cond)
        if not d:
            continue
        by_seed = {r.get("seed", 0): r["curve"][-1]["held_success"] for r in d["runs"]}
        row = "".join(f"{by_seed.get(s, float('nan')):<6.2f}" for s in seeds)
        print(f"{cond:<16}{row}")


def fig_main(data):
    labels, means, cis, colors = [], [], [], []
    for cond, label, color in CONDITIONS:
        d = data.get(cond)
        if not d:
            continue
        labels.append(label)
        means.append(d["mean"])
        cis.append(d["ci"])
        colors.append(color)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(x, means, yerr=cis, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6, alpha=0.92)
    ax.axhline(TARGET, ls="--", lw=1.0, color="0.35")
    ax.text(len(labels) - 0.5, TARGET + 0.012, f"target = {TARGET:.2f}",
            ha="right", va="bottom", fontsize=8, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Final held-out success")
    ax.set_ylim(0, max(m + c for m, c in zip(means, cis)) * 1.18)
    ax.set_title("Neural curriculum: final held-out success (mean $\\pm$ 95% CI, 10 seeds)",
                 fontsize=10)
    for xi, m, c in zip(x, means, cis):
        ax.text(xi, m + c + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, "curriculum_main.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_curves(data):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for cond, label, color in CONDITIONS:
        d = data.get(cond)
        if not d:
            continue
        runs = d["runs"]
        # all seeds share the same checkpoint grid; align by index
        n_ck = min(len(r["curve"]) for r in runs)
        xs = np.array([runs[0]["curve"][i]["env_steps"] for i in range(n_ck)]) / 1e6
        mat = np.array([[r["curve"][i]["held_success"] for i in range(n_ck)]
                        for r in runs], dtype=float)
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / math.sqrt(mat.shape[0])
        lw = 2.4 if cond in ("oracle", "geometry", "adaptive_coarse", "random") else 1.4
        a = 1.0 if lw > 2 else 0.75
        clean = label.replace("\n", " ")
        ax.plot(xs, mean, color=color, lw=lw, alpha=a, label=clean)
        ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.12, lw=0)
    ax.axhline(TARGET, ls="--", lw=1.0, color="0.4")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Held-out success")
    ax.set_title("Held-out learning curves (mean $\\pm$ 1 SEM over 10 seeds)", fontsize=10)
    ax.set_ylim(-0.01, None)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, "curriculum_curves.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_seeds(data):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rng = np.random.default_rng(0)
    xticklabels = []
    for i, (cond, label, color) in enumerate(CONDITIONS):
        d = data.get(cond)
        if not d:
            continue
        finals = d["finals"]
        jitter = (rng.random(len(finals)) - 0.5) * 0.28
        ax.scatter(np.full(len(finals), i) + jitter, finals, s=34,
                   color=color, edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
        ax.hlines(d["mean"], i - 0.28, i + 0.28, color="black", lw=2.0, zorder=4)
        xticklabels.append(label)
    ax.axhline(TARGET, ls="--", lw=1.0, color="0.4")
    ax.text(len(xticklabels) - 0.5, TARGET + 0.012, f"target = {TARGET:.2f}",
            ha="right", va="bottom", fontsize=8, color="0.4")
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=8.5)
    ax.set_ylabel("Final held-out success (per seed)")
    ax.set_title("Per-seed robustness: each dot is one of 10 seeds (bar = mean)", fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, "curriculum_seeds.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_cells(data, conds=("random", "geometry", "oracle")):
    sizes = [9, 13, 17]
    tiers = [1, 2, 3]
    panels = [c for c in conds if c in data]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4),
                             constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    label_of = {c: lbl.replace("\n", " ") for c, lbl, _ in CONDITIONS}
    im = None
    for ax, cond in zip(axes, panels):
        runs = data[cond]["runs"]
        grid = np.zeros((len(sizes), len(tiers)))
        for si, s in enumerate(sizes):
            for ti, t in enumerate(tiers):
                key = f"s{s}N{t}"
                vals = [r["curve"][-1]["held_by_cell"].get(key, np.nan) for r in runs]
                grid[si, ti] = np.nanmean(vals)
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(tiers)), [f"{t} river" for t in tiers], fontsize=8)
        ax.set_yticks(range(len(sizes)), [f"{s}x{s}" for s in sizes], fontsize=8)
        ax.set_title(label_of.get(cond, cond), fontsize=9.5)
        for si in range(len(sizes)):
            for ti in range(len(tiers)):
                v = grid[si, ti]
                ax.text(ti, si, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Held-out success")
    fig.suptitle("Where the curriculum helps: held-out success by (grid size $\\times$ rivers)",
                 fontsize=10)
    out = os.path.join(IMAGES_DIR, "curriculum_cells.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    data = gather()
    if not data:
        raise SystemExit(f"no result JSONs found in {RESULTS_DIR}")
    print_summary(data)
    fig_main(data)
    fig_curves(data)
    fig_seeds(data)
    fig_cells(data)


if __name__ == "__main__":
    main()
