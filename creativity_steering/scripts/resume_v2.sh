#!/bin/bash
# Resume v2 — fine + Exp-2 only, assuming derive/analyze/broad are already persisted.
# Use when downstream_v2.sh was interrupted mid-fine/exp-2 (e.g. pod restart) and the
# vectors/, analysis/, and broad_sweep_results.json under artifacts/qwen14b_run_v2/ are
# still on disk. Reads the top-5 fine layers from broad_sweep_results.json via
# select_promising_layers, runs the judged fine sweep, then Exp-2 on the same layers.
#
# Usage (on RunPod):
#   nohup bash creativity_steering/scripts/resume_v2.sh > artifacts/qwen14b_run_v2/downstream.log 2>&1 &
set -e
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a

CFG_DIR=creativity_steering/configs
RUN=artifacts/qwen14b_run_v2
POS=last_response_token
PD=$RUN/patching/$POS

echo "RESUME_V2_START $(date)"

LAYERS=$(python3 -c "
import json, sys
sys.path.insert(0, 'creativity_steering')
from creativity_steering.patching.sweep import select_promising_layers
br = json.load(open('$PD/broad_sweep_results.json'))
print(','.join(str(L) for L in select_promising_layers(br, 5)))
")
echo "FINE_LAYERS $LAYERS $(date)"

echo "== FINE: top-5 layers, a in [-1,1,2,4], eval=5, judged =="
time creativity-steering sweep --run "$RUN" --config "$CFG_DIR/fine_v2.json" --position "$POS" --layers "$LAYERS"
cp "$PD/sweep_results.json" "$PD/fine_sweep_results.json"
echo "FINE_DONE $(date)"

echo "== EXP-2: top-5 promising layers, judged =="
time creativity-steering experiments --run "$RUN" --config "$CFG_DIR/fine_v2.json" --position "$POS" --n-promising 5
echo "EXP2_DONE $(date)"

echo "RESUME_V2_DONE_OK $(date)"
