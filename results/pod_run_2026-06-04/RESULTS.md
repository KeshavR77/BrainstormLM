# Diversity validation — 14B run (2026-06-04)

Pod: RunPod RTX A6000 (48GB, CUDA-12.8 driver). Harness: `scripts/validate_diversity.py`.

## Headline
**Activation steering does NOT beat the shipping `antimode` baseline on Vendi at 14B (FP16).**
Every logged steered arm scores below `antimode` (3.06), and the arms that raise Vendi above it lose quality.

## Results — FP16 14B, K=8, 30 briefs, baseline = `antimode`
Values are the per-arm summary lines printed in `sweep.log` / `sweep3.log`. Cells marked "—" were
not saved: they came from `recompute.py` / `vsfilter.py` runs whose output was not kept.

| arm | Vendi | quality | result |
|---|---|---|---|
| greedy | 1.000 | 1.000 | floor |
| **antimode (ship)** | **3.062** | **0.996** | baseline |
| vs (raw) | 4.282 | 0.921 | +Vendi, −7.5pp quality |
| ortho_L17 @0.2 | 2.020 | 1.000 | worse |
| ortho_L17 @0.4 | 2.746 | 0.992 | worse |
| ortho @0.2 | 1.898 | 1.000 | worse |
| ortho @0.4 | — | 0.533 | quality broke |
| ortho_temp @0.3 | — | 0.821 | quality loss |
| ortho_temp @0.4 | — | 0.458 | quality broke |

Not reported: the verbalized-sampling + quality-filter arm (`vsfilter.py`), the NB-Distinct column,
and the AWQ-vs-FP16 Vendi comparison. Their numbers were never saved, so they can't be checked.
The AWQ ortho_L17@0.4 run logged quality 0.938 (`awq_ortho.log`) over 4 briefs. That's too little to
say whether steering behaves the same at 4-bit.

## Recommendations
1. Don't ship activation steering for diversity based on this run: it loses to `antimode` at 14B FP16.
2. Re-run `vsfilter.py` and save its output before treating verbalized sampling + filter as a win.
3. Keep `antimode` as the strong base.
4. AWQ at speed needs a CUDA-13 pod (torch 2.11+cu130 → gptqmodel Marlin) or nvcc for runtime
   Marlin compile. On CUDA-12.8 AWQ ran ~1.3 tok/s (TorchAten) → FP16 used for the full sweep.

## Files here
- `validate_fp16_L17_sweep.generations.json` — greedy, antimode, vs, ortho_L17@{0.2,0.4}, ortho@{0.2,0.4}
- `validate_fp16_ortho_temp.generations.json` — ortho_temp@{0.3,0.4}
- `validate_awq_ortho.generations.json` — AWQ ortho_L17@0.4 (ship-precision)
- `baseline.json` / `baseline_besteffort.json` — GPT-4o demo baselines for one packaging brief: with naive
  sampling all 8 forms are asymmetrical twisted/faceted shapes; asked explicitly for 8 distinct ones, they vary.
- `sweep.log` / `sweep3.log` / `awq_ortho.log` — run logs (qpr + metric prints)
- `vsfilter.py` / `recompute.py` — the vs+filter test and the robust standalone metric recompute
  (the harness bootstrap hangs on degenerate arms; recompute.py computes metrics once per arm).

Method note: the harness's in-line bootstrap CIs hung on degenerate arms (a no-timeout path worth
fixing); `recompute.py` can recompute NB-Distinct/Vendi from the saved generations.
