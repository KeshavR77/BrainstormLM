#!/bin/bash
# v3 downstream: framed contrast (revert-to-v1) + quality-gated-novelty metric.
#
# One derive (framed creative-vs-neutral, greedy neutral, save_scored_pool, ALL capture
# positions), then the broad -> novelty-gate -> fine -> Exp-2 sweep run on BOTH mean_response
# (v1's proven position) and last_response_token (v2's, kept as an A/B). Because the derive
# captures every position and saves the full scored pool, both positions and any re-cut
# threshold are available with no re-generation (see scripts/refilter_v3.py).
#
# Selection: top 10% creative / bottom 10% neutral by novelty, capped at one response per
# prompt (filtering.one_per_prompt), positives quality-gated at judge Pragmaticality >= 3
# (filtering.quality_floor). Reward model OFF; the LLM judge carries the quality gate.
#
# Usage (on RunPod):
#   nohup bash scripts/downstream_v3.sh > artifacts/qwen14b_run_v3/downstream.log 2>&1 &
set -e
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a

CFG_DIR=creativity_steering/configs
RUN=artifacts/qwen14b_run_v3
POSITIONS="mean_response last_response_token"

mkdir -p "$RUN"
echo "DOWNSTREAM_V3_START $(date)"

# ----------------------------------------------------------------------------
# Phase 1 — derive ONCE (framed, greedy neutral, quality-gated, 1-per-prompt,
#           all capture positions + scored pool persisted)
# ----------------------------------------------------------------------------
echo "== DERIVE: framed evaluator_stack, top/bottom 10% novelty, 1/prompt, quality_floor=3 =="
time creativity-steering derive --config "$CFG_DIR/runpod_qwen14b_v3.json" --out "$RUN"
echo "DERIVE_DONE $(date)"

for POS in $POSITIONS; do
  PD="$RUN/patching/$POS"
  echo "############### POSITION: $POS ###############"

  echo "== ANALYZE ($POS): norm/cosine/oracle (informational) =="
  creativity-steering analyze --run "$RUN" --position "$POS"

  # --- Broad scan: all 48 layers, alpha=1, judge-free novelty ranking ---
  echo "== BROAD ($POS): all layers, a=1, eval=2 =="
  time creativity-steering sweep --run "$RUN" --config "$CFG_DIR/broad_v3.json" --position "$POS"
  cp "$PD/sweep_results.json" "$PD/broad_sweep_results.json"
  echo "BROAD_DONE $POS $(date)"

  # --- Novelty gate: a usable vector must raise Stage-1 novelty for >=1 layer ---
  echo "== NOVELTY GATE ($POS) =="
  if python3 creativity_steering/scripts/novelty_gate.py "$PD/broad_sweep_results.json"; then
    echo "NOVELTY_GATE_PASS $POS $(date)"
  else
    echo "NOVELTY_GATE_FAIL $POS $(date) — skipping fine/Exp-2 for $POS (saves judge budget)"
    continue
  fi

  # --- Fine sweep: top-5 layers x alphas, judged (quality-gated composite) ---
  LAYERS=$(python3 -c "
import json, sys
sys.path.insert(0, 'creativity_steering')
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

echo "DOWNSTREAM_V3_DONE_OK $(date)"
