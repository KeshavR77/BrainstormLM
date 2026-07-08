# Porting the steering stack to Qwen2.5-32B-Instruct

The chosen next base model for the bigger production run. Decision rationale and the full
audit are in the vault; the short version:

- **Same architecture family as the proven 14B** — identical **5120 hidden dim**, so the
  DiffMeans steering mechanics are *unchanged*; only the layer count grows (48 → **64**).
- **Same Qwen chat template** — `apply_chat_template` is drop-in; no tokenizer/framing changes.
- **Fits a single A100-80GB in bf16** (~63 GB peak: 60.8 GB weights + KV-cache), so we stay on
  the full-fidelity bf16 steering path and off the AWQ trap (gptqmodel Marlin needs CUDA-13/nvcc;
  ~1.3 tok/s otherwise).
- 72B needs 2×80 GB (throughput cliff); Qwen3-32B is same-footprint but unproven tokenizer/hooks;
  Llama-70B is a different architecture. 32B is the most capable we can run the whole stack on
  one GPU.

`HFBackend` reads `num_layers` / `hidden_dim` from `model.config` at runtime, so **no code
changes** are needed — the port is a config swap plus a fresh derivation + sweep.

## Pod

- **GPU:** 1× A100-80GB (SXM or PCIe), Community Cloud. bf16 32B peak ~63 GB.
- **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (ships torch
  2.4.1+cu124, CUDA available out of the box; don't let pip swap torch).
- **Disk:** 30 GB container + 80 GB volume at `/workspace` (the 32B bf16 weights are ~61 GB).
- **Est. cost:** full derive + sweep ~8-10 h on one A100 ≈ **$20-30**.

## Setup (RunPod-Lessons baked in)

```bash
# 1. Verify torch/CUDA FIRST; do not let pip disturb it.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # want True
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 2. Clone + install (accelerate for device_map; evaluator extra for the judge-gated filter).
git clone -b main https://github.com/KeshavR77/BrainstormLM.git && cd BrainstormLM
pip install -e "creativity_steering[hf,serve,refine,evaluator]" accelerate
pip install -e "creativity_evaluators[all]"       # DSI/PPL/judges for evaluator_stack filtering
python -c "import torch; print('cuda', torch.cuda.is_available())"              # re-verify: still True

# 3. Auth + caches on the volume (HF_TOKEN avoids the unauthenticated rate-limit download stall;
#    HFBackend now reads HF_TOKEN automatically, but set it for the CLI paths too).
export HF_HOME=/workspace/.hf PYTHONUNBUFFERED=1
set -a; . ./.env; set +a          # HF_TOKEN + OPENAI_API_KEY (judges) + the rest
export OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY   # the 3-judge ensemble

# 4. Pre-download the 32B weights (authenticated) BEFORE deriving, so a stall is isolated.
HF_XET_HIGH_PERFORMANCE=1 python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-32B-Instruct')"
# If a shard's .incomplete freezes at a fixed size, it's the Xet finalize hang: kill, then
# `mv blobs/<oid>.*.incomplete blobs/<oid>` and symlink it into snapshots/<rev>/. See the vault.
```

## Derive → sweep → ship

```bash
# 1. Derive the 64-layer DiffMeans vectors (framed quality-gated-novelty contrast). ~35-40 min.
creativity-steering derive \
  --config creativity_steering/configs/runpod_qwen32b_v1.json \
  --out artifacts/qwen32b_run_v1
# -> artifacts/qwen32b_run_v1/vectors/{mean_response,last_response_token,assistant_start}/
#    manifest.json should report  n_layers=64, hidden_dim=5120, model_id=Qwen/Qwen2.5-32B-Instruct.

# 2. Layer/alpha sweep (broad 64L then judged frontier). ~4-6 h. The 14B sweet spot was a
#    SMALL alpha (0.2-0.25) in the FIRST ~15% of layers (L6/L7 of 48); on 64 layers expect the
#    analogous band ~L8-L10. Do NOT reuse the 14B layer index; the sweep re-finds it.
creativity-steering sweep \
  --run artifacts/qwen32b_run_v1 \
  --config creativity_steering/configs/runpod_qwen32b_v1.json

# 3. Ship: write a serving package pointing at the sweep-winning layer/alpha, e.g.
#    creativity_steering/configs/ship_v1_32b.json (copy ship_v3_bracket.json, set
#    "vectors_path": "artifacts/qwen32b_run_v1/vectors/mean_response",
#    "model_id": "Qwen/Qwen2.5-32B-Instruct", and the winning {layer, alpha} variants).
```

## Validate before the big run

- Manifest sanity: `n_layers=64`, `hidden_dim=5120`, correct `model_id`.
- One A/B: `serve_steering` with the ship package, `POST /compare` on a brief — steered must
  differ from baseline and stay coherent (Pragmaticality-equivalent eyeball). α past ~0.3 breaks
  coherence, same cliff as 14B.
- Then run the briefs exactly as the 14B designer run, pointing the batch driver at the 32B ship
  package (`REFINER=openai` for steer→refine):
  ```bash
  STEERING_PACKAGE=creativity_steering/configs/ship_v1_32b.json REFINER=openai \
  HF_HOME=/workspace/.hf python creativity_steering/scripts/generate_briefs.py \
    --variant <winning_layer> --samples 12 --refine --baseline --out results/designer_run_32b
  ```
  The driver is fault-isolated (a bad sample records + continues) and checkpointed (re-run to
  resume). Promote `IMAGE_PROMPTS.md` to `artifacts/designer_run_32b/`; raw jsonl stays gitignored.

## What must change (and what must NOT)

**Change (config only):** `model_id` → 32B in the ship/derive configs; `vectors_path` →
`artifacts/qwen32b_run_v1/...`; the shipped `{layer, alpha}` (from the sweep, not copied from 14B).

**Do NOT change:** `HFBackend` (auto-reads dims), the rescale formula `(v/‖v‖)·‖h‖·α`, the
sampling presets, the alpha grid, the capture positions, the tokenizer path — all model-agnostic.

**Do NOT reuse** the 14B vectors (`data/steering_14b/`, `artifacts/qwen14b_run_v3/`) — they are in
14B's representation space; a bigger model needs freshly-derived vectors even though the hidden
dim happens to match.
