#!/bin/bash
# One-command resume of the v3 derive on a FRESH pod from the laptop's saved checkpoint.
#
# Pushes the code + the banked checkpoint (artifacts/qwen14b_run_v3, ~1.1 GB), reinstalls the
# env (transformers pinned), RE-CAPTURES any activation positions lost in transit (forward pass
# per saved response — no regeneration), then relaunches downstream_v3.sh, which RESUMES the
# top-up loop from the checkpoint (generation continues from where it stopped) and runs the
# sweeps once 100/pole is reached.
#
# Usage:
#   bash creativity_steering/scripts/resume_v3_on_pod.sh <pod_host> <pod_port> [positions_to_recapture...]
#   e.g.  bash creativity_steering/scripts/resume_v3_on_pod.sh 154.54.102.44 15352 mean_response
set -euo pipefail
HOST=${1:?pod host}; PORT=${2:?pod port}; shift 2
RECAP_POS="${*:-mean_response}"
KEY=~/.ssh/id_ed25519
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SSHO="-p $PORT -i $KEY -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15"
SSH="ssh $SSHO"
RE="ssh $SSHO"
DST="root@$HOST:/workspace/BrainstormLM"

echo "== 1/5 install rsync on pod =="
$SSH "root@$HOST" 'which rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)'

echo "== 2/5 push code + .env + banked checkpoint + judge cache =="
rsync -rlt --no-o --no-g --no-p -e "$RE" \
  --exclude '.venv' --exclude '*/.venv' --exclude '**/__pycache__' --exclude '*.pyc' \
  --exclude '**/.pytest_cache' --exclude '.ruff_cache' --exclude '*/.ruff_cache' \
  "$REPO/creativity_steering" "$REPO/creativity_evaluators" "$REPO/.env" "$DST/"
rsync -rlt --no-o --no-g --no-p --partial -e "$RE" \
  "$REPO/artifacts/qwen14b_run_v3/" "$DST/artifacts/qwen14b_run_v3/"
rsync -rlt --no-o --no-g --no-p -e "$RE" \
  "$REPO/.cache/creativity_evaluators/" "$DST/.cache/creativity_evaluators/" || true

echo "== 3/5 keys + env install =="
$SSH "root@$HOST" 'cd /workspace/BrainstormLM && cp .env .keys.env && chmod 600 .keys.env &&
  pip install -q -e "creativity_evaluators[stage1,nlp,judges]" &&
  pip install -q -e "creativity_steering[hf]" &&
  pip install -q "transformers==4.46.3" &&
  (python -m spacy download en_core_web_sm >/dev/null 2>&1 || true)'

echo "== 4/5 re-capture lost activation positions: $RECAP_POS =="
$SSH "root@$HOST" "cd /workspace/BrainstormLM && set -a; . ./.keys.env; set +a;
  for p in $RECAP_POS; do
    if [ -f artifacts/qwen14b_run_v3/activations/\$p.npz ]; then echo \"\$p present, skip\";
    else python creativity_steering/scripts/recapture_activations.py \
      --run artifacts/qwen14b_run_v3 \
      --config creativity_steering/configs/runpod_qwen14b_v3.json --positions \$p; fi
  done"

echo "== 5/5 resume derive + sweeps (continues from checkpoint) =="
$SSH "root@$HOST" 'cd /workspace/BrainstormLM &&
  nohup bash creativity_steering/scripts/downstream_v3.sh \
    >> artifacts/qwen14b_run_v3/downstream.log 2>&1 </dev/null & echo "resumed pid $!"'
echo "DONE. watch:  ssh $SSHO root@$HOST \"grep 'top-up round' /workspace/BrainstormLM/artifacts/qwen14b_run_v3/downstream.log | tail\""
