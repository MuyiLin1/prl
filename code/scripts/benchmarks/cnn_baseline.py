"""CNN-on-raw-grid baseline for the Sokoban 3000 public benchmark: is the grader's edge from our features,
or would any learned model do? Same 5 folds as the RF rows; early stopping uses a slice of the training fold only."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

torch.set_num_threads(2)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CH = {"#": 0, ".": 1, "$": 2, "@": 3}


def encode(ascii_grid):
    rows = ascii_grid.split("\n")
    x = np.zeros((4, len(rows), len(rows[0])), np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            if c in CH:
                x[CH[c], i, j] = 1
    return x


def dihedral(x):  # 8 rotations/flips; Sokoban difficulty is invariant to them
    out = []
    for k in range(4):
        r = np.rot90(x, k, axes=(-2, -1))
        out += [r, np.flip(r, -1)]
    return np.ascontiguousarray(np.stack(out, 1))  # N x 8 x C x H x W


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                               nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def predict(net, A):
    net.eval()
    with torch.no_grad():
        n, k = A.shape[:2]
        return net(torch.from_numpy(A.reshape(n * k, *A.shape[2:])).to(DEV)).reshape(n, k).mean(1).cpu().numpy()


def fit_predict(A, y, tr, te, classify, seed, epochs=60):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tr = rng.permutation(tr); va, tr = tr[: len(tr) // 10], tr[len(tr) // 10:]
    mu, sd = (0.0, 1.0) if classify else (y[tr].mean(), y[tr].std())
    yt = (y - mu) / sd
    Xtr = torch.from_numpy(A[tr].reshape(-1, *A.shape[2:])).to(DEV)
    Ytr = torch.from_numpy(np.repeat(yt[tr], A.shape[1]).astype(np.float32)).to(DEV)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss() if classify else nn.MSELoss()
    score = (lambda p, t: roc_auc_score(t, p)) if classify else (lambda p, t: spearmanr(p, t)[0])
    best, state = -np.inf, None
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(Xtr), device=DEV).split(256):
            opt.zero_grad(); loss_fn(net(Xtr[b]), Ytr[b]).backward(); opt.step()
        s = score(predict(net, A[va]), y[va])
        if s > best:
            best, state = s, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return predict(net, A[te])


def main():
    b = pd.read_csv(ROOT / "data/benchmarks/boxoban_astar/benchmark_levels.csv")
    A = dihedral(np.stack([encode(s) for s in b.ascii]))
    tasks = [("log A* nodes expanded", np.log10(b.astar_search_steps.values).astype(np.float32), False),
             ("A* optimal solution length", b.astar_solution_len.values.astype(np.float32), False),
             ("Boxoban tier (medium = DRC agents failed)", (b.tier == "medium").astype(np.float32).values, True)]
    rows = []
    for label, y, classify in tasks:
        split = (StratifiedKFold if classify else KFold)(5, shuffle=True, random_state=0)
        for seed in range(3):
            p = np.zeros(len(y))
            for tr, te in split.split(A, y):
                p[te] = fit_predict(A, y, tr, te, classify, seed)
            v = roc_auc_score(y, p) if classify else spearmanr(p, y)[0]
            rows.append(dict(label=label, seed=seed, metric="AUC" if classify else "Spearman", value=v))
            print(f"{label:45s} seed{seed}: {v:.3f}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "data/benchmarks/cnn_baseline_sokoban3000.csv", index=False)
    print(r.groupby(["label", "metric"]).value.agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
