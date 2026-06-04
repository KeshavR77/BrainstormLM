import sys, numpy as np
sys.path.insert(0, "scripts")
from validate_diversity import BUILTIN_BRIEFS, _judge_one
from brainstormlm.refine import refine, make_llm, make_embedder
from brainstormlm.eval import novelty_bench_distinct, conditional_vendi
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
briefs = BUILTIN_BRIEFS[:30]; OVER=14; K=8
llm = make_llm(temperature=0.9); embedder = make_embedder()
def vs_overgen(b):
    try: return [c for c in refine(b, k=OVER, llm=llm, embedder=embedder).ideas if c and c.strip()]
    except Exception as e: print("refine_err", str(e)[:90], flush=True); return []
print("[1] vs over-gen (14/brief) ...", flush=True)
with ThreadPoolExecutor(max_workers=8) as ex: cands_per=list(ex.map(vs_overgen, briefs))
print("  avg cands/brief:", round(float(np.mean([len(c) for c in cands_per])),1), flush=True)
pairs=[(i,c) for i,cs in enumerate(cands_per) for c in cs]
print(f"[2] judging {len(pairs)} candidates ...", flush=True)
def judge(p): i,c=p; return (i,c,_judge_one(briefs[i], c))
with ThreadPoolExecutor(max_workers=12) as ex: judged=list(ex.map(judge, pairs))
passers=defaultdict(list)
for i,c,s in judged:
    if s["coherence"]>=3 and s["on_topic"]>=4: passers[i].append(c)
selected=[ (passers[i] or cands_per[i])[:K] for i in range(len(briefs)) ]
groups=[g for g in selected if g]
nd=float(np.mean([novelty_bench_distinct(g)["n_distinct"] for g in groups]))
vd=float(conditional_vendi(groups)["mean_within_vendi"])
avgk=float(np.mean([len(g) for g in groups])); frac8=float(np.mean([len(g)>=8 for g in groups]))
passrate=float(np.mean([1 if (s["coherence"]>=3 and s["on_topic"]>=4) else 0 for _,_,s in judged]))
print("\n=== vs + quality-filter + select(8) ===")
print(f"NBdist={nd:.2f}  vendi={vd:.3f}  avg_selected={avgk:.1f}  briefs_with_>=8={frac8:.0%}  raw_pass_rate={passrate:.3f}")
print("--- antimode (ship):  NBdist=7.47 vendi=3.06 qpr=0.996")
print("--- raw vs (sweep1):  NBdist=7.90 vendi=4.28 qpr=0.921")
print("DONE_VSFILTER")
