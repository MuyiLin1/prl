"""EXPLORATORY (post hoc, hypothesis-generating only): do path-chokepoint features explain the reliable
within-cell progress label on 9/13/17 that the global descriptor misses?"""
import json, glob, sys, warnings; warnings.filterwarnings("ignore")
ROOT="/Users/linmuyi/code/temp-prl"; sys.path[:0]=[ROOT, ROOT+"/scripts/curriculum"]
import numpy as np, pandas as pd
from collections import deque
from scipy.stats import spearmanr
from difficulty_scoring import LAVA, WALL, _grid_objects
OUT="/Users/linmuyi/code/temp-prl/notes/rescue_audit_2026-10-01/derisk_out"
def bfs(img, s):
    d={s:0}; q=deque([s]); par={s:None}
    while q:
        x,y=q.popleft()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            n=(x+dx,y+dy)
            if n not in d and img[n] not in (WALL,LAVA): d[n]=d[(x,y)]+1; par[n]=(x,y); q.append(n)
    return d,par
def feats(size,n,seed):
    img,s,g=_grid_objects(size,n,seed); d,par=bfs(img,s)
    path=[]; c=g
    while c is not None: path.append(c); c=par[c]
    path=path[::-1]
    # chokepoints = path cells adjacent to lava on both sides along the crossing axis (river gaps)
    gaps=[i for i,(x,y) in enumerate(path) if (img[x-1,y]==LAVA and img[x+1,y]==LAVA) or (img[x,y-1]==LAVA and img[x,y+1]==LAVA)]
    L=len(path)-1
    first = gaps[0]/L if gaps else 1.0
    return dict(first_gap_frac=first, first_gap_dist=gaps[0] if gaps else L, last_gap_frac=gaps[-1]/L if gaps else 1.0,
                mean_gap_frac=np.mean(gaps)/L if gaps else 1.0, n_gaps=len(gaps), path_len=L)
runs=[json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/p91317_ref*.json"))]
base=pd.DataFrame(runs[0]["rows"]); cell=base["size"].astype(str)+"N"+base["n"].astype(str)
P=np.array([[1-np.mean(r["prog"]) for r in run["rows"]] for run in runs]).T.mean(1)
F=pd.DataFrame([feats(*t) for t in zip(base["size"],base["n"],base.seed)])
res=lambda v: v-pd.Series(v).groupby(cell.values).transform("mean").values
print("n_gaps == N:", (F.n_gaps.values==base.n.values).mean())
for c in F.columns:
    print(f"  {c:15s} overall rho {spearmanr(F[c],P)[0]:+.2f}   within-cell rho {spearmanr(res(F[c].values.astype(float)),res(P))[0]:+.2f}")
