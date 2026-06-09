#!/bin/bash
# v3 SMALL-ALPHA sweep — the follow-up to the alpha=1 layer scan, which showed every layer goes
# degenerate (Pragmaticality 1.0) at alpha=1. This sweeps the high-novelty layers at SMALL
# alphas (0.1/0.25/0.5) to find where coherence survives while novelty stays up — i.e. the
# highest-novelty cell that clears the 2.0 quality bar. Each cell stores raw Pragmaticality.
#
# Layers = the top-novelty layers from mean_response's alpha=1 broad scan (early band L5-12 +
# the mid-late high-novelty L32/33/37/39). Runs on mean_response (the position we have a novelty
# ranking for). Uses the derived vectors only (no re-derive).
#
# Usage (on RunPod):  nohup bash creativity_steering/scripts/downstream_v3_smallalpha.sh \
#                         > artifacts/qwen14b_run_v3/smallalpha.log 2>&1 &
set -e
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a

CFG_DIR=creativity_steering/configs
RUN=artifacts/qwen14b_run_v3
POS=mean_response
LAYERS="5,6,7,8,12,32,33,37,39"

echo "SMALLALPHA_V3_START $(date)  position=$POS layers=$LAYERS alphas=0.1/0.25/0.5"
PD="$RUN/patching/$POS"
time creativity-steering sweep --run "$RUN" --config "$CFG_DIR/small_alpha_v3.json" \
  --position "$POS" --layers "$LAYERS"
cp "$PD/sweep_results.json" "$PD/smallalpha_sweep_results.json"
echo "SMALLALPHA_V3_DONE_OK $(date)"
