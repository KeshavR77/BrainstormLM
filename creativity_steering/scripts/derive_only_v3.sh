#!/bin/bash
# v3 derive only — 1100 unframed prompts (data/v3_topics.json), top/bottom 10% quantile
# split, save_scored_pool=true so vectors at any (cutoff, N) can be re-derived later via
# scripts/refilter_v3.py without re-running generation or scoring.
#
# Use this when you want the derive output staged but haven't yet picked the downstream
# config. After this finishes, refilter at any cutoff:
#   python -m creativity_steering.scripts.refilter_v3 --run artifacts/qwen14b_run_v3 \
#       --top-percentile 10 --bot-percentile 10 \
#       --out artifacts/qwen14b_run_v3/refiltered/p10
#
# Then run downstream_v3.sh (broad/fine/Exp-2) against the chosen vector set.
#
# Usage (on RunPod):
#   nohup bash creativity_steering/scripts/derive_only_v3.sh > artifacts/qwen14b_run_v3/derive.log 2>&1 &
set -e
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a

CFG_DIR=creativity_steering/configs
RUN=artifacts/qwen14b_run_v3

mkdir -p "$RUN"

echo "DERIVE_V3_START $(date)"

echo "== DERIVE: 1100 unframed prompts, score_split quantile 0.9/0.1, save_scored_pool=true =="
time creativity-steering derive --config "$CFG_DIR/runpod_qwen14b_v3.json" --out "$RUN"
echo "DERIVE_V3_DONE $(date)"

echo "== ANALYZE: norm/cosine/oracle (informational only) =="
creativity-steering analyze --run "$RUN" --position last_response_token

echo "DERIVE_V3_DONE_OK $(date)"
