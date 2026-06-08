#!/bin/bash
# v3 LAYER SWEEP ONLY — the broad all-48-layer scan at alpha=1, for BOTH capture positions,
# using the derived vectors (artifacts/qwen14b_run_v3/vectors/). No fine/Exp-2: the point is to
# get the per-layer (novelty, raw Pragmaticality, gated composite) table so we can pick the
# highest-novelty layers that clear the quality barrier, per position.
#
# Quality barrier = filtering.quality_floor in broad_v3.json (now 2.0 = "coherent/refinable",
# since outputs get refined downstream). judge_top_k bumped so the high-novelty layers all get a
# real quality read; each cell now stores raw mean Pragmaticality (1-5) in sweep_results.json.
#
# Usage (on RunPod):  nohup bash creativity_steering/scripts/downstream_v3_layersweep.sh \
#                         > artifacts/qwen14b_run_v3/layersweep.log 2>&1 &
set -e
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a

CFG_DIR=creativity_steering/configs
RUN=artifacts/qwen14b_run_v3
POSITIONS="mean_response last_response_token"

echo "LAYERSWEEP_V3_START $(date)"
for POS in $POSITIONS; do
  PD="$RUN/patching/$POS"
  if [ ! -f "$RUN/vectors/$POS/vectors.npz" ]; then
    echo "!! no vectors for $POS at $RUN/vectors/$POS — skipping"; continue
  fi
  echo "############### POSITION: $POS ###############"

  echo "== ANALYZE ($POS) =="
  creativity-steering analyze --run "$RUN" --position "$POS"

  echo "== LAYER SWEEP ($POS): all 48 layers, a=1, eval=3, judges on top-20, quality_floor=2 =="
  time creativity-steering sweep --run "$RUN" --config "$CFG_DIR/broad_v3.json" --position "$POS"
  cp "$PD/sweep_results.json" "$PD/broad_sweep_results.json"
  echo "LAYERSWEEP_DONE $POS $(date)"

  echo "== NOVELTY GATE ($POS, informational) =="
  python3 creativity_steering/scripts/novelty_gate.py "$PD/broad_sweep_results.json" || true
done

echo "LAYERSWEEP_V3_DONE_OK $(date)"
