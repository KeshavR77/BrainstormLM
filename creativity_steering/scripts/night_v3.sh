#!/bin/bash
# night_v3.sh — overnight v3 statistics-optimization run. Built to NEVER fail catastrophically.
#
# MEASURED cost on this pod (A100-80GB): ~1.8 min/cell at eval_samples=3 (15 responses/cell),
# judge OFF. Judging adds ~6-8 min per judged cell (Tier-1 rate-limited, 3-judge ensemble).
#
# Design for "do not fail overnight":
#   * GPU-FIRST: all irreplaceable generation is produced + persisted to disk BEFORE any fragile
#     API/judge work, so a rate-limit death cannot destroy the night's value.
#   * CHUNKED novelty sweep: the full 48-layer map is split into 6 layer-groups. Generations only
#     persist when a sweep finishes, so a monolithic 4.6h sweep that crashes loses everything —
#     chunks cap any single loss at ~45 min and checkpoint progress.
#   * FAULT-ISOLATED: no `set -e`; a stage/chunk failure is logged and the run CONTINUES.
#   * WALL-CLOCK GUARDED: optional later work is skipped if behind. Heartbeat -> NIGHT_STATUS.txt.
#   * RESUMABLE judging via the content-hash judge cache.
#
# Stages:
#   S1  novelty map, mean_response, 48L (6 chunks) x a{0.2,0.3,0.4}, judge OFF   ~4.6h
#   S2  judged frontier, mean_response, top-6 novelty layers x a{0.2..0.4 /.05}, judge ON  ~2.2h
#   S3  judged frontier, last_response_token (same layers) — ONLY if comfortably ahead
#
# Launch (tmux so it survives disconnect):
#   tmux new-session -d -s night 'bash creativity_steering/scripts/night_v3.sh'
set -u
cd /workspace/BrainstormLM
set -a; . ./.keys.env; set +a
export HF_HOME=/workspace/.cache/huggingface
# Replace the placeholder or set HF_TOKEN in .keys.env or the environment.
export HF_TOKEN="${HF_TOKEN:-your_hugging_face_access_token}"
export TOKENIZERS_PARALLELISM=false

RUN=artifacts/qwen14b_run_v3
CFG=creativity_steering/configs
PD=$RUN/patching/mean_response
PDL=$RUN/patching/last_response_token
STATUS=$RUN/NIGHT_STATUS.txt
START=$(date +%s)
HARD_BUDGET_MIN=440          # never START new heavy work past this (leaves buffer under 8h)
: > "$STATUS"

elapsed() { echo $(( ($(date +%s) - START) / 60 )); }
log() { echo "[$(date '+%m-%d %H:%M:%S') +$(elapsed)m] $*" | tee -a "$STATUS"; }

# run one sweep; on success copy outputs to stage-named files. Never aborts the script.
#   $1 tag  $2 config  $3 position  $4 layers|ALL  $5 out-prefix  $6 patch-dir
run_sweep() {
  local tag="$1" cfg="$2" pos="$3" layers="$4" prefix="$5" pd="$6"
  local args=(sweep --run "$RUN" --config "$cfg" --position "$pos")
  [ "$layers" != "ALL" ] && args+=(--layers "$layers")
  log "${tag}_START pos=$pos layers=$layers cfg=$(basename "$cfg")"
  if creativity-steering "${args[@]}" > "$RUN/${tag}.log" 2>&1; then
    [ -f "$pd/sweep_results.json" ]      && cp "$pd/sweep_results.json"      "$pd/${prefix}_results.json"
    [ -f "$pd/sweep_generations.jsonl" ] && cp "$pd/sweep_generations.jsonl" "$pd/${prefix}_generations.jsonl"
    local n; n=$(grep -c . "$pd/${prefix}_generations.jsonl" 2>/dev/null || echo 0)
    log "${tag}_DONE_OK -> ${prefix}_results.json (+${n} gens)"
    return 0
  else
    log "${tag}_FAILED (tail below) -- continuing"
    tail -6 "$RUN/${tag}.log" | sed 's/^/    /' | tee -a "$STATUS"
    return 1
  fi
}

log "NIGHT_START pid=$$ gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) cost~1.8min/cell"

# ---------------- S1: chunked full-48-layer novelty map (mean_response) ----------------
CHUNKS=("0,1,2,3,4,5,6,7" "8,9,10,11,12,13,14,15" "16,17,18,19,20,21,22,23" \
        "24,25,26,27,28,29,30,31" "32,33,34,35,36,37,38,39" "40,41,42,43,44,45,46,47")
S1_FILES=()
ci=0
for ch in "${CHUNKS[@]}"; do
  ci=$((ci+1))
  if [ "$(elapsed)" -ge "$HARD_BUDGET_MIN" ]; then
    log "S1_chunk${ci}_SKIP past hard budget ($(elapsed)m) -- proceeding to judged frontier"
    break
  fi
  if run_sweep "S1c${ci}" "$CFG/phase1_smallalpha_v3.json" mean_response "$ch" "S1_chunk${ci}" "$PD"; then
    S1_FILES+=("$PD/S1_chunk${ci}_results.json")
  fi
done

# merge whatever chunks landed into one novelty map
if [ "${#S1_FILES[@]}" -gt 0 ]; then
  python creativity_steering/scripts/merge_cells.py "$PD/S1_mean_novelty_results.json" "${S1_FILES[@]}" \
    >> "$STATUS" 2>&1 || log "S1 merge WARN"
  log "S1_MERGED ${#S1_FILES[@]} chunks ($(elapsed)m elapsed)"
else
  log "S1_NO_CHUNKS landed -- judged frontier will use fallback layers"
fi
log "CHECKPOINT novelty generation persisted; everything below is recoverable off-pod"

# ---------------- S2: judged frontier on the mean_response winners ----------------
FALLBACK="5,6,7,8,12,32"
LAYERS=$(python creativity_steering/scripts/pick_top_layers.py "$PD/S1_mean_novelty_results.json" 6 2>/dev/null)
[ -z "$LAYERS" ] && { LAYERS="$FALLBACK"; log "S2 layer-pick FELL BACK to $FALLBACK"; }
log "S2 frontier layers = $LAYERS"
if [ "$(elapsed)" -lt "$HARD_BUDGET_MIN" ]; then
  run_sweep S2 "$CFG/phase3_judged_v3.json" mean_response "$LAYERS" S2_mean_judged "$PD"
else
  log "S2_SKIP past hard budget ($(elapsed)m)"
fi

# ---------------- S3: judged frontier on last_response_token (only if well ahead) ----------------
EL=$(elapsed)
if [ "$EL" -lt 330 ]; then
  log "S3 attempting last_response_token judged frontier (ahead of schedule, ${EL}m)"
  run_sweep S3 "$CFG/phase3_judged_v3.json" last_response_token "$LAYERS" S3_lasttok_judged "$PDL"
else
  log "S3_SKIP not enough buffer for a second position (${EL}m elapsed)"
fi

log "NIGHT_DONE_ALL total=$(elapsed)m"
log "Key artifacts: $PD/S1_mean_novelty_results.json  $PD/S2_mean_judged_results.json (+ *_generations.jsonl)"
