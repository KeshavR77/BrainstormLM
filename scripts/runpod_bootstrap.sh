#!/bin/bash
# RunPod A40/A6000 bootstrap — clone, install, run AWQ steering pipeline.
# Usage: curl/scp this to the pod, then:
#   chmod +x runpod_bootstrap.sh && ./runpod_bootstrap.sh
#
# Requires: GITHUB_TOKEN env var (fine-grained PAT with repo read access)
#   export GITHUB_TOKEN=github_pat_...
#   ./runpod_bootstrap.sh [--smoke|--compare|--compare-quick]

set -euo pipefail

SMOKE="${1:-}"
REPO="KeshavR77/BrainstormLM"
WORKDIR="/workspace/BrainstormLM"

echo "============================================================"
echo " RunPod A6000 — BrainstormLM AWQ Steering Pipeline"
echo "============================================================"

# --- Preflight checks ---
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "ERROR: GITHUB_TOKEN not set. Create a fine-grained PAT at:"
    echo "  https://github.com/settings/personal-access-tokens/new"
    echo "  Scope: Contents (read) on KeshavR77/BrainstormLM"
    echo ""
    echo "Then: export GITHUB_TOKEN=github_pat_..."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ERROR: nvidia-smi failed — no GPU detected"; exit 1
}

echo ""
echo "[1/5] Cloning repo..."
if [ -d "$WORKDIR" ]; then
    echo "  $WORKDIR exists, pulling latest..."
    cd "$WORKDIR"
    git pull origin main
else
    git clone "https://${GITHUB_TOKEN}@github.com/${REPO}.git" "$WORKDIR"
    cd "$WORKDIR"
fi

# --- Use system Python + pip (avoid uv venv on FUSE filesystem) ---
echo ""
echo "[2/5] Redirecting HF cache to volume..."
export HF_HOME=/workspace/.cache/huggingface
mkdir -p "$HF_HOME"

echo ""
echo "[3/5] Installing Python dependencies (system pip)..."
pip install transformers==4.46.3 autoawq==0.2.6 accelerate nltk safetensors --quiet 2>/dev/null
pip install -e "$WORKDIR" --quiet 2>/dev/null

echo ""
echo "[4/5] Verifying CUDA..."
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'torch {torch.__version__} CUDA {torch.version.cuda} GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "[5/5] Running pipeline..."
MODE="${1:-}"
echo "  Start time: $(date)"
echo ""

if [ "$MODE" = "--compare" ]; then
    echo "  Mode: FULL COMPARISON (32 extraction + 20 validation × 3 arms)"
    python3 scripts/compare_awq_fp16_steering.py \
        --awq-model-path "Qwen/Qwen2.5-14B-Instruct-AWQ" \
        --force-extract 2>&1 | tee /workspace/pipeline_output.log
elif [ "$MODE" = "--compare-quick" ]; then
    echo "  Mode: QUICK COMPARISON (32 extraction + 8 validation × 3 arms)"
    python3 scripts/compare_awq_fp16_steering.py \
        --awq-model-path "Qwen/Qwen2.5-14B-Instruct-AWQ" \
        --force-extract --quick 2>&1 | tee /workspace/pipeline_output.log
elif [ "$MODE" = "--smoke" ]; then
    echo "  Mode: SMOKE TEST (3 extraction prompts, 2 validation prompts)"
    python3 scripts/quantize_and_steer_14b.py \
        --skip-quantize --awq-model-path "Qwen/Qwen2.5-14B-Instruct-AWQ" \
        --smoke 2>&1 | tee /workspace/pipeline_output.log
else
    echo "  Mode: FULL RUN (32 extraction prompts, 5 validation prompts)"
    python3 scripts/quantize_and_steer_14b.py \
        --skip-quantize --awq-model-path "Qwen/Qwen2.5-14B-Instruct-AWQ" \
        2>&1 | tee /workspace/pipeline_output.log
fi

echo ""
echo "============================================================"
echo " Done: $(date)"
echo " Log:     /workspace/pipeline_output.log"
echo " Results: artifacts/"
echo " Vectors: data/steering_14b_awq/"
echo "============================================================"
