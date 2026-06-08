#!/bin/bash
# v3 SWEEPS ONLY — run on a pod that already has the derived vectors
# (artifacts/qwen14b_run_v3/vectors/<position>/). Skips the derive phase (already complete:
# 118 pos / 120 neg), so only the small vectors need to be on the pod, not the 3.3 GB pool.
#
#   broad (all 48 layers) -> novelty gate -> fine (top-5 x alphas) -> Exp-2, for both positions.
#
# Usage (on RunPod):
#   nohup bash creativity_steering/scripts/downstream_v3_sweep.sh \
#       > artifacts/qwen14b_run_v3/sweep.log 2>&1 &
set -e
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a

CFG_DIR=creativity_steering/configs
RUN=artifacts/qwen14b_run_v3
POSITIONS="mean_response last_response_token"

echo "SWEEP_V3_START $(date)"
for POS in $POSITIONS; do
  PD="$RUN/patching/$POS"
  if [ ! -f "$RUN/vectors/$POS/vectors.npz" ]; then
    echo "!! no vectors for $POS at $RUN/vectors/$POS — skipping"; continue
  fi
  echo "############### POSITION: $POS ###############"

  echo "== ANALYZE ($POS): norm/cosine/oracle (informational) =="
  creativity-steering analyze --run "$RUN" --position "$POS"

  echo "== BROAD ($POS): all layers, a=1, eval=2 =="
  time creativity-steering sweep --run "$RUN" --config "$CFG_DIR/broad_v3.json" --position "$POS"
  cp "$PD/sweep_results.json" "$PD/broad_sweep_results.json"
  echo "BROAD_DONE $POS $(date)"

  echo "== NOVELTY GATE ($POS) =="
  if python3 creativity_steering/scripts/novelty_gate.py "$PD/broad_sweep_results.json"; then
    echo "NOVELTY_GATE_PASS $POS $(date)"
  else
    echo "NOVELTY_GATE_FAIL $POS $(date) — skipping fine/Exp-2 for $POS (saves judge budget)"
    continue
  fi

  LAYERS=$(python3 -c "
import json
from creativity_steering.patching.sweep import select_promising_layers
br = json.load(open('$PD/broad_sweep_results.json'))
print(','.join(str(L) for L in select_promising_layers(br, 5)))
")
  echo "FINE_LAYERS $POS $LAYERS $(date)"
  echo "== FINE ($POS): top-5 layers, a in [-1,1,2,4], eval=5, judged =="
  time creativity-steering sweep --run "$RUN" --config "$CFG_DIR/fine_v3.json" --position "$POS" --layers "$LAYERS"
  cp "$PD/sweep_results.json" "$PD/fine_sweep_results.json"
  echo "FINE_DONE $POS $(date)"

  echo "== EXP-2 ($POS): top-5 promising layers, judged =="
  time creativity-steering experiments --run "$RUN" --config "$CFG_DIR/fine_v3.json" --position "$POS" --n-promising 5
  echo "EXP2_DONE $POS $(date)"
done

echo "SWEEP_V3_DONE_OK $(date)"
