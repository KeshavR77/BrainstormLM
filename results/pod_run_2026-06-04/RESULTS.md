# Diversity validation — 14B run (2026-06-04)

Pod: RunPod RTX A6000 (48GB, CUDA-12.8 driver). Harness: `scripts/validate_diversity.py`.

## Headline
**Activation steering does NOT beat the shipping `antimode` baseline at 14B (FP16 or AWQ).**
**The one validated diversity lever is verbalized sampling + a quality filter (+31% Vendi at full quality).**

## Results — FP16 14B, K=8, 30 briefs, baseline = `antimode`
| arm | NB-Distinct | Vendi | quality | result |
|---|---|---|---|---|
| greedy | 1.00 | 1.00 | 1.00 | floor |
| **antimode (ship)** | **7.47** | **3.06** | **0.996** | baseline |
| vs (raw) | 7.90 | 4.28 | 0.921 | +Vendi, VOID (−7.5pp quality) |
| **vs + quality-filter** | 7.27 | **4.01** | ~1.0 | **WIN on Vendi at full quality** |
| ortho_L17 @0.2 | 3.47 | 2.02 | 1.00 | worse |
| ortho_L17 @0.4 | 6.23 | 2.75 | 0.99 | worse |
| ortho @0.2 | 3.10 | 1.90 | 1.00 | worse |
| ortho @0.4 | 4.00 | 2.68 | 0.53 | worse + broke |
| ortho_temp @0.3 | 5.93 | 2.97 | 0.82 | worse + VOID |
| ortho_temp @0.4 | 7.10 | 3.76 | 0.46 | +Vendi but quality broke |

## Ship-precision (AWQ 4-bit) transfer — same 4 briefs
| | NB-Distinct | Vendi | quality |
|---|---|---|---|
| AWQ ortho_L17@0.4 | 7.25 | 3.22 | 0.938 |
| FP16 ortho_L17@0.4 | 6.75 | 3.02 | 0.99 |

Steering behaves the same at 4-bit (+ quantization noise). Verdict transfers to ship precision.
`vs+filter` (the winner) is API-based → precision-independent.

## Why NB-Distinct doesn't move
At K=8 the baseline already hits 7.47/8 distinct clusters — NB-Distinct is **saturated**, so the
pre-registered "≥ antimode +1 class" bar (≥8.47) is structurally near-unbeatable. The real,
measurable diversity gain shows up in **Vendi** (continuous spread). Use Vendi as the north-star.

## Recommendations
1. Don't ship activation steering for diversity (loses at 14B; AWQ-confirmed). Re-test at 30B.
2. Diverge step → verbalized sampling + quality filter (validated; API/refiner; precision-independent).
3. Keep `antimode` as the strong base; layer vs+filter's quality-select on top.
4. North-star = Vendi (NB-Distinct saturates at K=8).
5. AWQ at speed needs a CUDA-13 pod (torch 2.11+cu130 → gptqmodel Marlin) or nvcc for runtime
   Marlin compile. On CUDA-12.8 AWQ ran ~1.3 tok/s (TorchAten) → FP16 used for the full sweep.

## Files here
- `validate_fp16_L17_sweep.generations.json` — greedy, antimode, vs, ortho_L17@{0.2,0.4}, ortho@{0.2,0.4}
- `validate_fp16_ortho_temp.generations.json` — ortho_temp@{0.3,0.4}
- `validate_awq_ortho.generations.json` — AWQ ortho_L17@0.4 (ship-precision)
- `baseline.json` / `baseline_besteffort.json` — frontier (GPT-4o) demo baselines: naive sampling
  mode-collapses 8/8 to "asymmetrical twist"; self-diversifies only when explicitly asked for 8.
- `sweep.log` / `sweep3.log` / `awq_ortho.log` — run logs (qpr + metric prints)
- `vsfilter.py` / `recompute.py` — the vs+filter test and the robust standalone metric recompute
  (the harness bootstrap hangs on degenerate arms; recompute.py computes metrics once per arm).

Method note: NB-Distinct/Vendi recomputed from saved generations via `recompute.py` (the harness's
in-line bootstrap CIs hung on degenerate arms — a no-timeout path worth fixing).
