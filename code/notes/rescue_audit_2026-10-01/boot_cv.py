import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
def oof(X,y):
    p=np.zeros(len(y))
    for tr,te in KFold(5,shuffle=True,random_state=0).split(X):
        p[te]=RandomForestRegressor(300,random_state=0,n_jobs=4).fit(X[tr],y[tr]).predict(X[te])
    return p
def feats(d):
    cols=[c for c in d.columns if c.startswith(("graph_","spectral_","grid_","reward_","local_")) and d[c].std()>1e-9]
    return d[cols].values.astype(float)
sets={}
for t in ["WallAvoider","DoorKey","Maze"]: sets[t]=pd.read_csv(f"data/karel_difficulty/karel_{t}_grids.csv")
sets["TopOff"]=pd.read_csv("averaged_label_topoff200.csv")
s=pd.read_csv("averaged_label_screen_snake_seeder.csv"); sets["Snake"]=s[s.task=="Snake"]
rng=np.random.default_rng(0)
for t,d in sets.items():
    X,y=feats(d),d.mean_difficulty.values; p=oof(X,y)
    bs=[spearmanr(p[i],y[i])[0] for i in (rng.integers(0,len(y),len(y)) for _ in range(5000))]
    print(f"{t:12s} n={len(y)} CV rho {spearmanr(p,y)[0]:+.2f}  95% CI [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]", flush=True)
