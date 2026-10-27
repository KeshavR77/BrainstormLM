#!/usr/bin/env python3
"""Full evaluation pipeline for the 2026-05-18 webpage stats refresh.

Generates outputs from both ``conceptial-14b`` (full stack — steering +
iterative anti-mode + prompt-level anti-mode) and ``baseline-14b`` (raw
Qwen/Qwen2.5-14B-Instruct, no steering, no anti-mode), then computes all
eight metrics for the landing-page stats block:

    Existing four (repeated-prompt stress test):
      1. Near-duplicate rate  (cosine ≥ 0.92 on OpenAI text-embedding-3-small)
      2. Vendi score          (Friedman & Dieng 2022)
      3. Pass@5               (≥1 of 5 outputs passes the quality gate)
      4. Quality gate pass %  (gpt-4o-mini judge: coherence ≥ 3 AND on_topic ≥ 4)

    New four:
      5. Self-BLEU-4          (Zhu et al. 2018, Texygen)
      6. MAUVE                (Pillutla et al. NeurIPS 2021, vs human refs)
      7. EQ-Bench Creative v3 composite (subset, gpt-4o-mini judge)
      8. Mode-Collapse@K      (greedy cluster count at cos ≥ 0.85, K=10)

Phases::

    --phase generate          # RunPod A100, ~80 min, ~$2 GPU
    --phase metrics           # local or pod, ~10 min, ~$0.40 OpenAI
    --phase all               # both

Output::

    artifacts/eval_2026-05-18/
      baseline-14b/<prompt-id>.json
      conceptial-14b/<prompt-id>.json
      eq_bench/<arm>/<prompt-id>.json
      stress/<arm>.json
      SUMMARY.json
      WEBPAGE_STATS.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "artifacts" / "eval_2026-05-18"

CANONICAL_STRESS_PROMPT_ID = "logo-stanford-engineering"
N_PER_PROMPT = 5
STRESS_K = 10

EQ_BENCH_MAX_NEW_TOKENS = 1100
BRIEFS_MAX_NEW_TOKENS = 350


# ─── Generation phase (RunPod) ────────────────────────────────────────────────


def _resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_conceptial_engine(corpius_cfg: dict, sweep_cfg_path: Path | None):
    """Build the Conceptial SteeringEngine with the v17 preset installed."""
    import torch  # noqa: F401

    from brainstormlm.conceptial.engine.config import SteeringConfig
    from brainstormlm.conceptial.engine.pipeline import SteeringEngine
    from brainstormlm.conceptial.engine.steering_presets import (
        apply_v10_parametric_preset,
    )

    model_id = corpius_cfg["base_model"]
    creative_vec = ROOT / corpius_cfg["steering_vectors"]["creative"]
    technical_vec = ROOT / corpius_cfg["steering_vectors"]["technical"]
    creative_layer = int(corpius_cfg["layers"]["creative_layer"])
    technical_layer = int(corpius_cfg["layers"]["technical_layer"])
    schedule = dict(corpius_cfg["schedule"])

    if sweep_cfg_path and sweep_cfg_path.exists():
        chosen = json.loads(sweep_cfg_path.read_text())
        creative_layer = int(chosen.get("creative_layer", creative_layer))
        creative_vec = ROOT / chosen.get("creative_vector_path", creative_vec)
        schedule["creative_start_alpha"] = float(
            chosen.get("creative_alpha", schedule["creative_start_alpha"])
        )

    device = _resolve_device()
    cfg_engine = SteeringConfig(model_id=model_id)
    cfg_engine.preferred_device = device
    engine = SteeringEngine(cfg_engine)
    engine.load()
    apply_v10_parametric_preset(
        engine,
        creative_vec.parent,
        "v17",
        creative_vector_path=creative_vec,
        technical_vector_path=technical_vec,
        creative_layer=creative_layer,
        technical_layer=technical_layer,
        steering_math=str(schedule.get("steering_math", "additive")),
        creative_start_alpha=float(schedule["creative_start_alpha"]),
        exp_zero_by_step=int(schedule["exp_zero_by_step"]),
        technical_start_alpha=float(schedule["technical_start_alpha"]),
        sigmoid_mid_step=float(schedule["sigmoid_mid_step"]),
        sigmoid_steepness=float(schedule["sigmoid_steepness"]),
    )
    return engine, creative_layer, technical_layer


def _generate_conceptial_arm(engine, creative_layer, brief: str, n: int, *, max_new_tokens: int) -> list[dict]:
    """Conceptial full-stack arm: prompt-level anti-mode + iterative L19 residual orthogonalization."""
    import torch

    from brainstormlm.conceptial.anti_mode import extract_summary, format_anti_mode
    from brainstormlm.conceptial.engine.pipeline import load_steering_checkpoint
    from brainstormlm.conceptial.engine.vector_purify import purify_vector

    prior_summaries: list[str] = []
    prior_hiddens: list[torch.Tensor] = []
    outputs: list[dict] = []

    creative_vec_path = ROOT / json.loads((ROOT / "config" / "conceptial_14b.json").read_text())[
        "steering_vectors"
    ]["creative"]

    for k in range(n):
        if prior_hiddens:
            bundle = load_steering_checkpoint(creative_vec_path, map_location="cpu")
            v = bundle["vector"].float()
            mean_h = torch.stack(
                [h.float().reshape(-1) for h in prior_hiddens], dim=0
            ).mean(0)
            v_iter = purify_vector(v, [mean_h])
            p_param = next(engine.model.parameters())
            v_iter = v_iter.to(device=p_param.device, dtype=p_param.dtype)
            current = dict(engine.steering_vectors or {})
            current[creative_layer] = v_iter
            engine.set_steering_vectors(current)

        prompt = format_anti_mode(
            user_text=brief,
            prior_summaries=prior_summaries,
            exemplars=None,
            output_mode="generic",
        )
        temp = float(0.8 + (((k * 0.61803) % 1.0 - 0.5) * 0.1))
        first_token_buf: list[torch.Tensor] = []
        text = engine.complete(
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=temp,
            do_sample=True,
            do_steering=True,
            use_ghost_rejection=False,
            capture_first_decode_residual=(creative_layer, first_token_buf),
        )
        text = text.strip()
        if first_token_buf:
            prior_hiddens.append(first_token_buf[0].detach().to("cpu").float())
        summary = extract_summary(text) or text.split("\n")[0][:80]
        prior_summaries.append(summary)
        outputs.append(
            {
                "model": "conceptial-14b",
                "k": k,
                "text": text,
                "summary": summary,
                "trajectory_id": uuid.uuid4().hex,
                "temperature": temp,
            }
        )
    return outputs


def _load_baseline_engine(model_id: str):
    """Plain HF AutoModelForCausalLM for the unsteered baseline."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device


def _generate_baseline_arm(
    model, tokenizer, device, brief: str, n: int, *, max_new_tokens: int
) -> list[dict]:
    """Baseline arm: same brief asked N times, no anti-mode, no steering, sampled with the same temp schedule."""
    import torch

    outputs: list[dict] = []
    for k in range(n):
        temp = float(0.8 + (((k * 0.61803) % 1.0 - 0.5) * 0.1))
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a brand creative director. When given a brief, respond with ONE "
                    "concept direction: a memorable 2-6 word name as the first line, then a "
                    "one-paragraph body. No preamble, no alternatives."
                ),
            },
            {"role": "user", "content": brief},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=temp,
                do_sample=True,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(
            {
                "model": "baseline-14b",
                "k": k,
                "text": text,
                "trajectory_id": uuid.uuid4().hex,
                "temperature": temp,
            }
        )
    return outputs


def phase_generate(args) -> None:
    """Generate all outputs (creative briefs + EQ-Bench + stress test) for both arms."""
    briefs_cfg = json.loads((ROOT / "config" / "benchmark_creative_briefs.json").read_text())
    corpius_cfg = json.loads((ROOT / "config" / "conceptial_14b.json").read_text())
    sweep_cfg = ROOT / "data" / "steering_14b" / "sweep" / "chosen_config.json"

    from brainstormlm.eval.eq_bench_creative import load_prompts as load_eq

    eq_prompts = load_eq()
    briefs = briefs_cfg["prompts"][: args.limit_briefs] if args.limit_briefs else briefs_cfg["prompts"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("baseline-14b", "conceptial-14b", "eq_bench/baseline-14b", "eq_bench/conceptial-14b", "stress"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    # ── Conceptial-14B arm ────────────────────────────────────────────────────
    if "conceptial-14b" in args.arms:
        print(f"\n=== Loading conceptial-14b ({corpius_cfg['base_model']}) ===")
        engine, creative_layer, _technical_layer = _load_conceptial_engine(corpius_cfg, sweep_cfg)

        print(f"\n=== Creative briefs · conceptial-14b · N={args.n} ===")
        for b in briefs:
            t0 = time.time()
            outs = _generate_conceptial_arm(engine, creative_layer, b["brief"], args.n, max_new_tokens=BRIEFS_MAX_NEW_TOKENS)
            (OUT_DIR / "conceptial-14b" / f"{b['id']}.json").write_text(
                json.dumps({"prompt_id": b["id"], "arm": "conceptial-14b", "outputs": outs}, indent=2)
            )
            print(f"  [{b['id']:35s}] {len(outs)} outputs in {time.time() - t0:.1f}s")

        print("\n=== Repeated-prompt stress test · conceptial-14b · K=10 ===")
        canon = next(b for b in briefs if b["id"] == CANONICAL_STRESS_PROMPT_ID)
        outs = _generate_conceptial_arm(engine, creative_layer, canon["brief"], STRESS_K, max_new_tokens=BRIEFS_MAX_NEW_TOKENS)
        (OUT_DIR / "stress" / "conceptial-14b.json").write_text(
            json.dumps({"prompt_id": canon["id"], "arm": "conceptial-14b", "k": STRESS_K, "outputs": outs}, indent=2)
        )

        print("\n=== EQ-Bench Creative v3 subset · conceptial-14b · 1/prompt ===")
        for p in eq_prompts:
            t0 = time.time()
            outs = _generate_conceptial_arm(engine, creative_layer, p["prompt"], 1, max_new_tokens=EQ_BENCH_MAX_NEW_TOKENS)
            (OUT_DIR / "eq_bench" / "conceptial-14b" / f"{p['id']}.json").write_text(
                json.dumps({"prompt_id": p["id"], "category": p["category"], "title": p["title"], "arm": "conceptial-14b", "outputs": outs}, indent=2)
            )
            print(f"  [eq-{p['id']:>3s} {p['title'][:30]:30s}] {time.time() - t0:.1f}s")

        del engine
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Baseline-14B arm ──────────────────────────────────────────────────────
    if "baseline-14b" in args.arms:
        model_id = corpius_cfg["base_model"]
        print(f"\n=== Loading baseline-14b ({model_id}) ===")
        model, tokenizer, device = _load_baseline_engine(model_id)

        print(f"\n=== Creative briefs · baseline-14b · N={args.n} ===")
        for b in briefs:
            t0 = time.time()
            outs = _generate_baseline_arm(model, tokenizer, device, b["brief"], args.n, max_new_tokens=BRIEFS_MAX_NEW_TOKENS)
            (OUT_DIR / "baseline-14b" / f"{b['id']}.json").write_text(
                json.dumps({"prompt_id": b["id"], "arm": "baseline-14b", "outputs": outs}, indent=2)
            )
            print(f"  [{b['id']:35s}] {len(outs)} outputs in {time.time() - t0:.1f}s")

        print("\n=== Repeated-prompt stress test · baseline-14b · K=10 ===")
        canon = next(b for b in briefs if b["id"] == CANONICAL_STRESS_PROMPT_ID)
        outs = _generate_baseline_arm(model, tokenizer, device, canon["brief"], STRESS_K, max_new_tokens=BRIEFS_MAX_NEW_TOKENS)
        (OUT_DIR / "stress" / "baseline-14b.json").write_text(
            json.dumps({"prompt_id": canon["id"], "arm": "baseline-14b", "k": STRESS_K, "outputs": outs}, indent=2)
        )

        print("\n=== EQ-Bench Creative v3 subset · baseline-14b · 1/prompt ===")
        for p in eq_prompts:
            t0 = time.time()
            outs = _generate_baseline_arm(model, tokenizer, device, p["prompt"], 1, max_new_tokens=EQ_BENCH_MAX_NEW_TOKENS)
            (OUT_DIR / "eq_bench" / "baseline-14b" / f"{p['id']}.json").write_text(
                json.dumps({"prompt_id": p["id"], "category": p["category"], "title": p["title"], "arm": "baseline-14b", "outputs": outs}, indent=2)
            )
            print(f"  [eq-{p['id']:>3s} {p['title'][:30]:30s}] {time.time() - t0:.1f}s")


# ─── Metrics phase (CPU, OpenAI API) ──────────────────────────────────────────


def _read_briefs() -> list[dict]:
    return json.loads((ROOT / "config" / "benchmark_creative_briefs.json").read_text())["prompts"]


def _load_arm_outputs(arm: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted((OUT_DIR / arm).glob("*.json")):
        data = json.loads(f.read_text())
        out[data["prompt_id"]] = [o["text"] for o in data["outputs"] if o.get("text")]
    return out


def _judge_quality_gate(brief: str, response: str) -> dict:
    """gpt-4o-mini judge — score coherence 1-5, on_topic 1-5, gate = (coh ≥ 3 AND ot ≥ 4)."""
    from brainstormlm.clients.openai import complete as openai_complete

    system = (
        "You are a brutally fair design / strategy reviewer. Score the concept on two axes, "
        "each 1-5 (5 best). Return ONLY a JSON object with keys: coherence, on_topic, "
        "one_line_rationale. No markdown fences."
    )
    user = f"BRIEF:\n{brief.strip()}\n\nCONCEPT:\n{response.strip()[:6000]}\n\nReturn JSON."
    raw, _u = openai_complete(
        system=system,
        messages=[{"role": "user", "content": user}],
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.0,
        use_cache=True,
    )
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        d = json.loads(raw)
        coh = int(round(float(d.get("coherence", 0))))
        ot = int(round(float(d.get("on_topic", 0))))
    except Exception:
        coh, ot = 0, 0
    coh = max(1, min(5, coh)) if coh else 0
    ot = max(1, min(5, ot)) if ot else 0
    return {"coherence": coh, "on_topic": ot, "passes_gate": coh >= 3 and ot >= 4}


def _eq_bench_judge(user_msg: str) -> dict:
    from brainstormlm.clients.openai import complete as openai_complete

    raw, _u = openai_complete(
        system="You are an expert literary judge. Return only valid JSON.",
        messages=[{"role": "user", "content": user_msg}],
        model="gpt-4o-mini",
        max_tokens=900,
        temperature=0.0,
        use_cache=True,
    )
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


def phase_metrics(args) -> None:
    """Compute all 8 metrics from generated outputs."""
    from brainstormlm.eval.diversity import near_duplicate_rate, vendi_score
    from brainstormlm.eval.eq_bench_creative import (
        aggregate as eq_aggregate,
        load_prompts as load_eq,
        score_response as eq_score,
    )
    from brainstormlm.eval.mode_collapse import mode_collapse_at_k
    from brainstormlm.eval.self_bleu import self_bleu

    briefs = {b["id"]: b for b in _read_briefs()}
    summary: dict = {"per_arm": {}, "config": {"n_per_prompt": N_PER_PROMPT, "stress_k": STRESS_K}}

    for arm in ("baseline-14b", "conceptial-14b"):
        arm_outputs = _load_arm_outputs(arm)
        if not arm_outputs:
            print(f"  [{arm}] no outputs — skipping. Run --phase generate first.")
            continue

        # ── Per-prompt stress-test diversity (briefs suite, N=5 each) ─────────
        per_prompt = []
        for pid, texts in arm_outputs.items():
            if len(texts) < 2:
                continue
            per_prompt.append(
                {
                    "prompt_id": pid,
                    "n": len(texts),
                    "vendi": vendi_score(texts),
                    "near_dup": near_duplicate_rate(texts),
                    "self_bleu": self_bleu(texts, n=4),
                }
            )

        # ── Quality gate + Pass@5 (per-prompt, per-output) ────────────────────
        gate_per_prompt = []
        n_judged = 0
        n_passed_total = 0
        n_pass_at_5 = 0
        for pid, texts in arm_outputs.items():
            brief = briefs.get(pid, {}).get("brief", "")
            judged = [_judge_quality_gate(brief, t) for t in texts]
            n_passed = sum(1 for j in judged if j["passes_gate"])
            n_judged += len(judged)
            n_passed_total += n_passed
            if n_passed >= 1:
                n_pass_at_5 += 1
            gate_per_prompt.append({"prompt_id": pid, "n": len(judged), "n_passed": n_passed})
        n_prompts = len(gate_per_prompt)
        pass_at_5_rate = (n_pass_at_5 / n_prompts) if n_prompts else 0.0
        gate_pass_rate = (n_passed_total / n_judged) if n_judged else 0.0

        # ── Mode-Collapse@K (stress run) ──────────────────────────────────────
        stress_path = OUT_DIR / "stress" / f"{arm}.json"
        mode_k = {"n_clusters": None, "k": None, "coverage_ratio": None}
        if stress_path.exists():
            stress = json.loads(stress_path.read_text())
            stress_texts = [o["text"] for o in stress["outputs"]]
            mode_k = mode_collapse_at_k(stress_texts)

        # ── EQ-Bench Creative v3 subset composite ─────────────────────────────
        eq_dir = OUT_DIR / "eq_bench" / arm
        eq_prompts = {p["id"]: p for p in load_eq()}
        eq_scored = []
        if eq_dir.exists():
            for f in sorted(eq_dir.glob("*.json")):
                data = json.loads(f.read_text())
                pid = data["prompt_id"]
                if not data["outputs"]:
                    continue
                response = data["outputs"][0]["text"]
                meta = eq_prompts.get(pid)
                if meta is None:
                    continue
                eq_scored.append(eq_score(meta["prompt"], response, judge=_eq_bench_judge))
        eq_agg = eq_aggregate(eq_scored)

        # ── MAUVE vs human reference set ──────────────────────────────────────
        mauve_val = None
        ref_path = ROOT / "data" / "mauve_reference" / "references.jsonl"
        if ref_path.exists():
            try:
                from brainstormlm.eval.mauve_metric import mauve_score

                refs = [json.loads(line)["text"] for line in ref_path.read_text().splitlines() if line.strip()]
                model_texts = [t for texts in arm_outputs.values() for t in texts]
                mauve_val = mauve_score(model_texts, refs, device_id=-1, max_text_length=256)
            except Exception as e:
                print(f"  [{arm}] MAUVE failed: {e}")

        # ── Aggregate ─────────────────────────────────────────────────────────
        mean_vendi = sum(p["vendi"] for p in per_prompt) / len(per_prompt) if per_prompt else 0.0
        mean_near_dup = sum(p["near_dup"] for p in per_prompt) / len(per_prompt) if per_prompt else 0.0
        mean_self_bleu = sum(p["self_bleu"] for p in per_prompt) / len(per_prompt) if per_prompt else 0.0

        summary["per_arm"][arm] = {
            "n_prompts": n_prompts,
            "mean_vendi": mean_vendi,
            "mean_near_dup": mean_near_dup,
            "mean_self_bleu": mean_self_bleu,
            "pass_at_5_rate": pass_at_5_rate,
            "quality_gate_pass_rate": gate_pass_rate,
            "mode_collapse_at_k": mode_k,
            "eq_bench_composite": eq_agg.get("mean_composite", 0.0),
            "eq_bench_n": eq_agg.get("n", 0),
            "mauve": mauve_val,
            "per_prompt": per_prompt,
        }

        print(
            f"  [{arm}] vendi={mean_vendi:.3f} near_dup={mean_near_dup:.3f} "
            f"self_bleu={mean_self_bleu:.3f} pass@5={pass_at_5_rate:.2f} "
            f"gate={gate_pass_rate:.2f} eq={eq_agg.get('mean_composite', 0.0):.2f} "
            f"mode@k={mode_k.get('n_clusters')}/{mode_k.get('k')}"
        )

    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR / 'SUMMARY.json'}")
    _write_webpage_stats(summary)


def _write_webpage_stats(summary: dict) -> None:
    """Render artifacts/eval_2026-05-18/WEBPAGE_STATS.md with the eight stats."""
    arms = summary.get("per_arm", {})
    base = arms.get("baseline-14b", {})
    conc = arms.get("conceptial-14b", {})
    if not base or not conc:
        print("  [WEBPAGE_STATS] missing one arm — skipping render.")
        return

    def f(x, fmt=".3f"):
        return "—" if x is None else format(x, fmt)

    lines = [
        "# Webpage stats — refreshed 2026-05-18",
        "",
        "Conceptial vs the same model (Qwen2.5-14B-Instruct) unsteered, on a "
        f"repeated-prompt stress test ({summary['config']['n_per_prompt']} regenerations × "
        f"{base.get('n_prompts', 0)} prompts).",
        "",
        "## Results (from `SUMMARY.json`)",
        "",
        "Small sample, one run, no significance tests. Labels are neutral on purpose: check each "
        "metric's direction before quoting it as a gain.",
        "",
        "| Metric | Conceptial | Baseline | Better when |",
        "|---|---|---|---|",
        f"| Near-duplicate rate (cos ≥ 0.92) | {f(conc['mean_near_dup'])} | {f(base['mean_near_dup'])} | lower |",
        f"| Vendi score | {f(conc['mean_vendi'])} | {f(base['mean_vendi'])} | higher |",
        f"| Pass@5 (≥1 of {summary['config']['n_per_prompt']} passes gate) | {f(conc['pass_at_5_rate'])} | {f(base['pass_at_5_rate'])} | higher |",
        f"| Quality gate pass rate | {f(conc['quality_gate_pass_rate'])} | {f(base['quality_gate_pass_rate'])} | higher |",
        f"| Self-BLEU-4 (n-gram overlap) | {f(conc['mean_self_bleu'])} | {f(base['mean_self_bleu'])} | lower |",
        f"| MAUVE vs human reference | {f(conc.get('mauve'))} | {f(base.get('mauve'))} | higher |",
        f"| EQ-Bench Creative v3 composite | {f(conc['eq_bench_composite'], '.2f')} | {f(base['eq_bench_composite'], '.2f')} | higher |",
        f"| Distinct modes at K={STRESS_K} (one prompt) | {conc['mode_collapse_at_k']['n_clusters']} / {conc['mode_collapse_at_k']['k']} | {base['mode_collapse_at_k']['n_clusters']} / {base['mode_collapse_at_k']['k']} | higher |",
        "",
        "## Methodology",
        "",
        "- **Model**: Qwen/Qwen2.5-14B-Instruct, fp16, A100 80GB.",
        f"- **Conceptial stack**: CAA L17 (creative, α₀={summary.get('config', {}).get('creative_alpha', '1.5')}) + L24 (technical, α₀=0.86), v17 exp+sigmoid schedule, iterative L17 residual orthogonalization, prompt-level anti-mode.",
        "- **Baseline**: Same model, no steering, no anti-mode, identical sampling temps (0.75–0.85 jitter).",
        f"- **Prompt suite**: `config/benchmark_creative_briefs.json` ({base['n_prompts']} creative-operations briefs).",
        f"- **N per prompt**: {summary['config']['n_per_prompt']} regenerations.",
        f"- **Repeated-prompt stress K**: {STRESS_K} regenerations of `{CANONICAL_STRESS_PROMPT_ID}`.",
        "- **Vendi / Near-dup**: OpenAI text-embedding-3-small, cosine kernel, Friedman & Dieng 2022.",
        "- **Quality gate / Pass@5**: gpt-4o-mini, coherence ≥ 3 AND on_topic ≥ 4 (1–5 scale).",
        "- **Self-BLEU-4**: nltk sentence_bleu, smoothing method 1 (Zhu et al. 2018).",
        "- **MAUVE**: gpt2-large featurizer, vs `data/mauve_reference/references.jsonl` human samples.",
        "- **EQ-Bench Creative**: 10-prompt subset of v3, gpt-4o-mini judge, 11 positive + 9 negative criteria, composite mean.",
        "- **Mode-Collapse@K**: greedy cosine clustering at threshold 0.85.",
        "",
    ]
    (OUT_DIR / "WEBPAGE_STATS.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR / 'WEBPAGE_STATS.md'}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("generate", "metrics", "all"), default="all")
    p.add_argument(
        "--arms",
        default="baseline-14b,conceptial-14b",
        help="Comma-separated list of arms to generate.",
    )
    p.add_argument("--n", type=int, default=N_PER_PROMPT, help="Outputs per brief per arm.")
    p.add_argument("--limit-briefs", type=int, default=None, help="Smoke-test with first N briefs.")
    args = p.parse_args()
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("generate", "all"):
        phase_generate(args)
    if args.phase in ("metrics", "all"):
        phase_metrics(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
