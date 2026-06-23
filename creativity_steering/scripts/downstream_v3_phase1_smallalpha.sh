#!/bin/bash
# v3 PHASE 1 — small-alpha FULL 48-layer scan, novelty only (no LLM-judge on-pod).
#
# Sweeps every layer at alpha in {0.2, 0.3, 0.4} on mean_response, scoring only the cheap
# Stage-1 embedding-novelty composite (filtering.enable_judge=false -> the judge gate is never
# hit; evaluate_stack runs Pass A over all 144 cells and stops). No API, no rate limits.
#
# Crucially this ALSO persists every generation to sweep_generations.jsonl so the LLM-judge can
# run later OFF-POD (laptop, $0 GPU) without regenerating: one pod session total. Pick the
# highest-novelty (layer, alpha) cells from the novelty table, then judge just those off-pod.
#
# Usage (on RunPod):  nohup bash creativity_steering/scripts/downstream_v3_phase1_smallalpha.sh \
#                         > artifacts/qwen14b_run_v3/phase1_smallalpha.log 2>&1 &
set -e
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a

CFG_DIR=creativity_steering/configs
RUN=artifacts/qwen14b_run_v3
POS=mean_response

echo "PHASE1_SMALLALPHA_V3_START $(date)  position=$POS layers=ALL(48) alphas=0.2/0.3/0.4 judge=OFF"
PD="$RUN/patching/$POS"

# No --layers => all 48 layers (cli.py: layer_list defaults to range(vs.n_layers)).
time creativity-steering sweep --run "$RUN" --config "$CFG_DIR/phase1_smallalpha_v3.json" \
  --position "$POS"

cp "$PD/sweep_results.json"      "$PD/phase1_smallalpha_sweep_results.json"
cp "$PD/sweep_generations.jsonl" "$PD/phase1_smallalpha_generations.jsonl"
echo "PHASE1_SMALLALPHA_V3_DONE_OK $(date)  novelty -> $PD/phase1_smallalpha_sweep_results.json"
echo "  generations (for off-pod judging) -> $PD/phase1_smallalpha_generations.jsonl"
