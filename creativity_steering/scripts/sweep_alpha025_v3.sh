#!/bin/bash
# Full 48-layer sweep at alpha=0.25, mean_response, judge OFF (Phase-1, novelty only).
# Mirrors the 0.2/0.3/0.4 night run's S1 stage so the 0.25 cells can later be judged
# OFF-POD in the SAME batch as the existing 144 cells (one comparable scale).
#
# Robust by design (same lessons as night_v3.sh): chunked into 6 layer-groups so a crash
# loses <=1 chunk; generations persist only at sweep end, so chunking caps the loss; each
# chunk is copied to a stage-named file and merged at the end. No judging on-pod.
#
# Launch (on pod, under tmux):
#   tmux new-session -d -s a025 'bash creativity_steering/scripts/sweep_alpha025_v3.sh'
set -u
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a
export HF_HOME=/workspace/.cache/huggingface
# Replace the placeholder or set HF_TOKEN in .keys.env or the environment.
export HF_TOKEN="${HF_TOKEN:-your_hugging_face_access_token}"
export TOKENIZERS_PARALLELISM=false

RUN=artifacts/qwen14b_run_v3
CFG=creativity_steering/configs/phase1_alpha025_v3.json
POS=mean_response
PD=$RUN/patching/$POS
STATUS=$RUN/A025_STATUS.txt
START=$(date +%s)
: > "$STATUS"
elapsed() { echo $(( ($(date +%s) - START) / 60 )); }
log() { echo "[$(date '+%m-%d %H:%M:%S') +$(elapsed)m] $*" | tee -a "$STATUS"; }

CHUNKS=("0,1,2,3,4,5,6,7" "8,9,10,11,12,13,14,15" "16,17,18,19,20,21,22,23" \
        "24,25,26,27,28,29,30,31" "32,33,34,35,36,37,38,39" "40,41,42,43,44,45,46,47")

log "A025_START pid=$$ alpha=0.25 layers=ALL(48) judge=OFF gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
FILES=()
ci=0
for ch in "${CHUNKS[@]}"; do
  ci=$((ci+1))
  log "A025c${ci}_START layers=$ch"
  if creativity-steering sweep --run "$RUN" --config "$CFG" --position "$POS" --layers "$ch" \
       > "$RUN/A025c${ci}.log" 2>&1; then
    cp "$PD/sweep_results.json"      "$PD/A025_chunk${ci}_results.json"
    cp "$PD/sweep_generations.jsonl" "$PD/A025_chunk${ci}_generations.jsonl"
    n=$(grep -c . "$PD/A025_chunk${ci}_generations.jsonl" 2>/dev/null || echo 0)
    FILES+=("$PD/A025_chunk${ci}_results.json")
    log "A025c${ci}_DONE_OK (+${n} gens)"
  else
    log "A025c${ci}_FAILED (tail $RUN/A025c${ci}.log) -- continuing"
    tail -5 "$RUN/A025c${ci}.log" | sed 's/^/    /' | tee -a "$STATUS"
  fi
done

if [ "${#FILES[@]}" -gt 0 ]; then
  python creativity_steering/scripts/merge_cells.py "$PD/A025_alpha025_novelty_results.json" "${FILES[@]}" \
    >> "$STATUS" 2>&1 || log "merge WARN"
  log "A025_MERGED ${#FILES[@]} chunks"
fi
log "A025_DONE_ALL total=$(elapsed)m  novelty -> $PD/A025_alpha025_novelty_results.json"
log "  generations (for off-pod judging) -> $PD/A025_chunk{1..6}_generations.jsonl"
