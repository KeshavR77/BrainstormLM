import sys, json, numpy as np
from brainstormlm.eval import novelty_bench_distinct, conditional_vendi
d = json.load(open(sys.argv[1]))
print("TOP:", type(d).__name__, "| keys:", list(d.keys())[:15] if isinstance(d,dict) else len(d))
sample_arm=list(d.keys())[0]; sv=d[sample_arm]
print(f"sample '{sample_arm}': {type(sv).__name__}", (f"len={len(sv)}" if hasattr(sv,'__len__') else ""),
      ("| elem0="+type(sv[0]).__name__ if isinstance(sv,list) and sv else ""))
def to_groups(v):
    if isinstance(v, dict):
        out=[]
        for _,ts in sorted(v.items()): out.append([t for t in ts if isinstance(t,str)] if isinstance(ts,list) else ([ts] if isinstance(ts,str) else []))
        return out
    if isinstance(v, list):
        if v and isinstance(v[0], list): return [[t for t in g if isinstance(t,str)] for g in v]
        if v and isinstance(v[0], str): return [[t] for t in v]
    return []
print(f"\n{'arm':<24}{'NBdist':>8}{'vendi':>9}{'briefs':>8}{'empty':>7}")
for arm,v in d.items():
    g=to_groups(v); emp=sum(1 for grp in g for t in grp if not (t and t.strip()))
    g=[[t for t in grp if t and t.strip()] for grp in g]; g=[grp for grp in g if grp]
    if not g: print(f"{arm:<24} (no texts)"); continue
    try:
        nd=float(np.mean([novelty_bench_distinct(x)['n_distinct'] for x in g]))
        vd=float(conditional_vendi(g)['mean_within_vendi'])
        print(f"{arm:<24}{nd:>8.2f}{vd:>9.3f}{len(g):>8}{emp:>7}")
    except Exception as e:
        print(f"{arm:<24} ERR {type(e).__name__}: {str(e)[:60]}")
print("DONE_RECOMPUTE")
