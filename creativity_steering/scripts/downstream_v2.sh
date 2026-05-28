#!/bin/bash
# v2 downstream: derive (score_split) -> broad all-48 sweep -> fine judged sweep -> Exp-2.
# Runs on the last_response_token vectors. All steps use the two-stage evaluator with
# reward (Skywork 8B) in Stage 1 and judges gated to the top-K cells in Stage 2.
#
# Usage (on RunPod):
#   nohup bash scripts/downstream_v2.sh > artifacts/qwen14b_run_v2/downstream.log 2>&1 &
set -e
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a

CFG_DIR=creativity_steering/configs
RUN=artifacts/qwen14b_run_v2
POS=last_response_token
PD=$RUN/patching/$POS

mkdir -p "$RUN"

echo "DOWNSTREAM_V2_START $(date)"

# ----------------------------------------------------------------------------
# Phase 1 — derive (score_split, no system prompts, samples_per_prompt=1)
# ----------------------------------------------------------------------------
echo "== DERIVE: score_split, unframed temp=1.15, 1 sample/prompt =="
time creativity-steering derive --config "$CFG_DIR/runpod_qwen14b_v2.json" --out "$RUN"
echo "DERIVE_DONE $(date)"

# ----------------------------------------------------------------------------
# Phase 2 — vector analysis (cheap, generation-free; informational only —
#           downstream layer selection comes from the broad sweep, not this).
# ----------------------------------------------------------------------------
echo "== ANALYZE: norm/cosine/oracle (informational) =="
creativity-steering analyze --run "$RUN" --position "$POS"

# ----------------------------------------------------------------------------
# Phase 3a — broad sweep across all 48 layers at alpha=1, judges gated to top-5
# ----------------------------------------------------------------------------
echo "== BROAD: all 48 layers, a=1, eval=2, judges on top-5 =="
time creativity-steering sweep --run "$RUN" --config "$CFG_DIR/broad_v2.json" --position "$POS"
cp "$PD/sweep_results.json" "$PD/broad_sweep_results.json"
echo "BROAD_DONE $(date)"

# Pick top-5 fine layers from the broad sweep results (judged-first, then Stage-1 rank).
LAYERS=$(python3 -c "
import json, sys
sys.path.insert(0, 'creativity_steering')
from creativity_steering.patching.sweep import select_promising_layers
br = json.load(open('$PD/broad_sweep_results.json'))
print(','.join(str(L) for L in select_promising_layers(br, 5)))
")
echo "FINE_LAYERS $LAYERS $(date)"

# ----------------------------------------------------------------------------
# Phase 3b — fine sweep on the top-5 layers, alphas [-1,1,2,4], eval=5, judged
# ----------------------------------------------------------------------------
echo "== FINE: top-5 layers, a in [-1,1,2,4], eval=5, judged =="
time creativity-steering sweep --run "$RUN" --config "$CFG_DIR/fine_v2.json" --position "$POS" --layers "$LAYERS"
cp "$PD/sweep_results.json" "$PD/fine_sweep_results.json"
echo "FINE_DONE $(date)"

# ----------------------------------------------------------------------------
# Phase 3c — Exp-2 (all-layers-single-vector + multi-vector combo) on top-5 promising
# ----------------------------------------------------------------------------
echo "== EXP-2: top-5 promising layers, judged =="
time creativity-steering experiments --run "$RUN" --config "$CFG_DIR/fine_v2.json" --position "$POS" --n-promising 5
echo "EXP2_DONE $(date)"

echo "DOWNSTREAM_V2_DONE_OK $(date)"
