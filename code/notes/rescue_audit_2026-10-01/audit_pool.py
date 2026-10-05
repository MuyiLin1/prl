import json, sys, collections, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from difficulty_scoring import enumerate_pool, geometric_score, bin_layouts, fit_geometry_oof, random_walk_probe
pool = enumerate_pool([9,13,17],[1,2,3],80,10)
print("pool", len(pool), "sig example", pool[0].sig)
cache = json.load(open("cache/oracle_b5fce063f3_r1e12.json"))
def tk(s):
    def t(x): return tuple(t(i) for i in x) if isinstance(x,list) else x
    return t(json.loads(s))
orc = {tk(k): v for k,v in cache.items()}
print("oracle keys match pool:", all(lo.sig in orc for lo in pool))
v = np.array([orc[lo.sig] for lo in pool]); print("oracle tied at 1.0:", (v>=0.999).sum(), "/", len(v), " distinct:", len(set(np.round(v,4))))
def comp(bins):
    return [dict(collections.Counter(f"s{lo.size}N{lo.n}" for lo in b)) for b in bins]
print("\nGEOMETRY bins:", comp(bin_layouts(pool, geometric_score)))
print("\nORACLE bins (stable sort):", comp(bin_layouts(pool, lambda lo: orc[lo.sig])))
for s in range(3):
    oof = fit_geometry_oof(pool, orc, seed=s)
    print(f"\nFITTED-RF-on-oracle (seed {s}) bins:", comp(bin_layouts(pool, lambda lo: oof[lo.sig])))
# rivers-only variant
print("\nRIVERS+detour bins:", comp(bin_layouts(pool, lambda lo: lo.features['est_rivers'] + lo.features['detour']/100)))
print("\nest_rivers == n for all:", all(lo.features['est_rivers']==lo.n for lo in pool))
# oracle by cell
cell = collections.defaultdict(list)
for lo in pool: cell[(lo.size,lo.n)].append(orc[lo.sig])
print({f"s{k[0]}N{k[1]}": round(float(np.mean(x)),3) for k,x in sorted(cell.items())})
