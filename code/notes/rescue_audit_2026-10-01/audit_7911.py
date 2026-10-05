import json, sys, collections, numpy as np, warnings, glob
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from difficulty_scoring import enumerate_pool, geometric_score, bin_layouts, fit_geometry_oof, feature_vector, GEOM_FEATURE_NAMES
from scipy.stats import spearmanr
def tk(s):
    def t(x): return tuple(t(i) for i in x) if isinstance(x,list) else x
    return t(json.loads(s))
pool = enumerate_pool([7,9,11],[1,2,3],80,10)
orc = {tk(k): v for k,v in json.load(open("cache/oracle_6ccc84561c_r3e20.json")).items()}
print("pool", len(pool), "keys match:", all(lo.sig in orc for lo in pool))
v=np.array([orc[lo.sig] for lo in pool]); print("tied@1:", (v>=.999).mean(), "distinct", len(set(np.round(v,4))), "median", np.median(v))
cell=collections.defaultdict(list)
for lo in pool: cell[(lo.size,lo.n)].append(orc[lo.sig])
for k,x in sorted(cell.items()): print(f"  s{k[0]}N{k[1]} mean {np.mean(x):.2f} sd {np.std(x):.2f} range {min(x):.2f}-{max(x):.2f}")
# variance decomposition: how much of label variance is explained by cell (knobs)?
grand=v.mean(); between=sum(len(x)*(np.mean(x)-grand)**2 for x in cell.values()); total=((v-grand)**2).sum()
print(f"eta^2 (fraction of variance explained by (size,N) cell) = {between/total:.2f}")
# spearman of knobs vs label
print("rho(geometric_score, oracle) =", round(spearmanr([geometric_score(l) for l in pool], v)[0],3))
print("rho(size, oracle)=", round(spearmanr([l.size for l in pool], v)[0],3), " rho(N, oracle)=", round(spearmanr([l.n for l in pool], v)[0],3))
# within-cell rho of each existing feature
F=np.array([feature_vector(l) for l in pool]); 
for j,name in enumerate(GEOM_FEATURE_NAMES):
    rs=[]
    for k in cell:
        idx=[i for i,l in enumerate(pool) if (l.size,l.n)==k]
        if np.std(F[idx,j])>0 and np.std(v[idx])>0: rs.append(spearmanr(F[idx,j], v[idx])[0])
    print(f"  within-cell mean rho {name:14s} {np.mean(rs) if rs else float('nan'):+.2f} (cells {len(rs)})")
oof=fit_geometry_oof(pool, orc, seed=0); p=np.array([oof[l.sig] for l in pool])
print("RF OOF CV rho (current 8 features incl n):", round(spearmanr(p,v)[0],3))
# results for 7/9/11 runs
for d in ["results_v1_buggygate","results_lc7911neg"]:
    r=collections.defaultdict(list)
    for f in glob.glob(f"{d}/*.json"):
        j=json.load(open(f)); r[j['condition']].append(j['final_held_success'])
    print(d, {c:(round(np.mean(x),3), len(x)) for c,x in sorted(r.items())}, "steps", j['total_steps'], j['sizes'])
