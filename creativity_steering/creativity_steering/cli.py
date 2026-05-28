"""Command-line interface: derive → analyze → patch, plus an offline end-to-end demo.

    creativity-steering derive    --config configs/local_mock.json --out artifacts/run
    creativity-steering analyze   --run artifacts/run
    creativity-steering sweep     --run artifacts/run --config configs/local_mock.json
    creativity-steering multi     --run artifacts/run --source-layer 12 --layers 9,12,15
    creativity-steering demo      --out artifacts/demo          # full pipeline on the mock

Swap to the real model by pointing ``--config`` at a config with ``backend="hf"`` and
``model_id="Qwen/Qwen2.5-14B-Instruct"`` (RunPod). No code changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from creativity_steering.config import ExperimentConfig, load_config

app = typer.Typer(add_completion=False, help=__doc__)


def _scorer(config: ExperimentConfig):
    from creativity_steering.scoring import make_scorer

    return make_scorer(config.filtering.scorer)


def _load_vector_set(run: Path, position: str | None):
    from creativity_steering.artifacts import SteeringVectorSet

    vectors_root = run / "vectors"
    positions = sorted(p.name for p in vectors_root.iterdir() if (p / "manifest.json").exists())
    if not positions:
        raise typer.BadParameter(f"no vector sets under {vectors_root}")
    chosen = position or positions[0]
    if chosen not in positions:
        raise typer.BadParameter(f"position {chosen!r} not in {positions}")
    return SteeringVectorSet.load(vectors_root / chosen), chosen


@app.command()
def derive(
    config: Annotated[Path, typer.Option(help="ExperimentConfig JSON.")],
    out: Annotated[Path, typer.Option(help="Output run directory.")] = Path("artifacts/run"),
) -> None:
    """Phase 1 — gather → filter → DiffMeans at every layer."""
    from creativity_steering.derivation.pipeline import derive_vectors, save_derivation
    from creativity_steering.models.factory import build_backend

    cfg = load_config(config)
    backend = build_backend(cfg)
    typer.echo(f"Deriving vectors with backend={cfg.backend} model={backend.model_id} ...")
    output = derive_vectors(backend, cfg, _scorer(cfg))
    save_derivation(output, out)
    cfg.save(out / "config.json")
    typer.echo(json.dumps(output.summary(), indent=2))
    typer.echo(f"Saved vectors + summary to {out}")


@app.command()
def analyze(
    run: Annotated[Path, typer.Option(help="Run directory from `derive`.")],
    position: Annotated[str | None, typer.Option(help="Capture position (default: first).")] = None,
    n_candidates: Annotated[int, typer.Option(help="Candidate layers to surface.")] = 5,
) -> None:
    """Phase 2 — norm + cosine + activation-oracle (stub) → candidate layers."""
    from creativity_steering.analysis.oracle import StubOracle
    from creativity_steering.analysis.report import build_report

    vs, chosen = _load_vector_set(run, position)
    oracle = StubOracle(reference_norms=vs.layer_norms)
    report = build_report(vs.vectors, oracle=oracle, n_candidates=n_candidates)

    out_dir = run / "analysis" / chosen
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    (out_dir / "report.md").write_text(report.to_markdown())
    typer.echo(f"[{chosen}] candidate layers: {report.candidate_layers}")
    typer.echo(f"  norm top layer={report.norm.top_layer}  cosine band={report.cosine.band}")
    typer.echo(f"Saved analysis to {out_dir}")


@app.command()
def sweep(
    run: Annotated[Path, typer.Option(help="Run directory from `derive`.")],
    config: Annotated[Path, typer.Option(help="ExperimentConfig JSON.")],
    position: Annotated[str | None, typer.Option()] = None,
    layers: Annotated[str | None, typer.Option(help="Comma-separated layers (default: all).")] = None,
) -> None:
    """Phase 3, Experiment 1 — per-layer matched alpha sweep.

    Scores each (layer, alpha) cell with ``patching.sweep_evaluator``: the cheap per-item
    scorer (``"per_item"``) or the full two-stage ``creativity_evaluators`` stack
    (``"two_stage"`` — DSI + LLM-judge, judges gated to the top-k cells).
    """
    from creativity_steering.models.factory import build_backend

    cfg = load_config(config)
    vs, chosen = _load_vector_set(run, position)
    backend = build_backend(cfg)
    layer_list = list(range(vs.n_layers)) if not layers else [int(x) for x in layers.split(",")]

    out_dir = run / "patching" / chosen
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.patching.sweep_evaluator == "two_stage":
        from creativity_steering.patching.evaluate import DEFAULT_BRIEFS
        from creativity_steering.patching.evaluate_stack import (
            build_alpha_cells,
            two_stage_cell_sweep,
        )

        fc = cfg.filtering
        cells = build_alpha_cells(vs, layer_list, cfg.patching.alphas, rescale=cfg.patching.rescale)
        typer.echo(
            f"two-stage sweep: {len(cells)} cells, judges on top-{cfg.patching.judge_top_k} "
            f"(enable_judge={fc.enable_judge})"
        )
        result = two_stage_cell_sweep(
            backend, cells,
            briefs=DEFAULT_BRIEFS,
            n_samples=cfg.patching.eval_samples,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            enable_judge=fc.enable_judge,
            judge_top_k=cfg.patching.judge_top_k,
            generator_family=fc.generator_family,
            stage1_floor_percentile=fc.stage1_floor_percentile,
            enable_reward=fc.enable_reward,
            use_reward_fallback=fc.use_reward_fallback,
            stage1_device=cfg.device,
        )
        (out_dir / "sweep_results.json").write_text(json.dumps(result.to_dict(), indent=2))
        best = result.best_cell()
        typer.echo(f"baseline composite={result.baseline.composite:.3f}  judged={result.judged_labels}")
        typer.echo(f"best cell: layer={best['layer']} alpha={best['alpha']} "
                   f"composite={best['composite']:.3f} (delta={best['delta']:+.3f}, judged={best['judged']})")
        typer.echo(f"Saved sweep to {out_dir}")
        return

    from creativity_steering.patching.sweep import per_layer_alpha_sweep

    result = per_layer_alpha_sweep(
        backend, vs, _scorer(cfg),
        alphas=cfg.patching.alphas, layers=layer_list, rescale=cfg.patching.rescale,
        n_samples=cfg.patching.eval_samples,
        max_new_tokens=cfg.generation.max_new_tokens, temperature=cfg.generation.temperature,
    )
    (out_dir / "sweep_results.json").write_text(json.dumps(result.to_dict(), indent=2))
    best = result.best_cell()
    typer.echo(f"baseline composite={result.baseline.composite:.3f}")
    typer.echo(f"best cell: layer={best['layer']} alpha={best['alpha']} "
               f"composite={best['composite']:.3f} (delta={best['delta']:+.3f})")
    typer.echo(f"Saved sweep to {out_dir}")


@app.command()
def multi(
    run: Annotated[Path, typer.Option(help="Run directory from `derive`.")],
    config: Annotated[Path, typer.Option(help="ExperimentConfig JSON.")],
    source_layer: Annotated[int, typer.Option(help="Layer whose vector goes at every layer (Exp 2).")],
    layers: Annotated[str, typer.Option(help="Comma-separated layers for the multi-vector combo.")],
    position: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Phase 3, Experiment 2 — all-layer single-vector + multi-vector combo."""
    from creativity_steering.models.factory import build_backend
    from creativity_steering.patching.multilayer import (
        all_layers_single_vector,
        multi_vector_combo,
    )

    cfg = load_config(config)
    vs, chosen = _load_vector_set(run, position)
    backend = build_backend(cfg)
    layer_list = [int(x) for x in layers.split(",")]

    all_layers = all_layers_single_vector(
        backend, vs, _scorer(cfg), source_layer=source_layer,
        alpha=cfg.patching.all_layers_alpha, n_samples=cfg.patching.eval_samples,
    )
    combo = multi_vector_combo(
        backend, vs, _scorer(cfg), layers=layer_list, alpha=1.0,
        n_samples=cfg.patching.eval_samples,
    )
    out_dir = run / "patching" / chosen
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "multilayer_results.json").write_text(
        json.dumps({"all_layers": all_layers.to_dict(), "combo": combo.to_dict()}, indent=2)
    )
    typer.echo(f"{all_layers.label}: delta={all_layers.delta:+.3f}")
    typer.echo(f"{combo.label}: delta={combo.delta:+.3f}")
    typer.echo(f"Saved multilayer results to {out_dir}")


@app.command()
def experiments(
    run: Annotated[Path, typer.Option(help="Run directory from `derive`.")],
    config: Annotated[Path, typer.Option(help="ExperimentConfig JSON.")],
    position: Annotated[str | None, typer.Option()] = None,
    layers: Annotated[
        str | None, typer.Option(help="Promising layers (default: top from the sweep results).")
    ] = None,
    n_promising: Annotated[
        int, typer.Option(help="How many promising layers to use when --layers is omitted.")
    ] = 3,
) -> None:
    """Phase 3, Experiment 2 — multi-layer experiments on promising layers, two-stage scored.

    Builds *all-layers-single-vector* cells (each promising layer's vector injected at every
    layer, small alphas) and a *multi-vector-combo* cell (the promising layers' own vectors at
    their own layers), then scores them with the DSI + LLM-judge stack (judges gated to the
    top-k cells). Promising layers default to the top layers from Experiment 1's
    ``sweep_results.json`` (best behavioural creativity), not the cheaper vector analysis.
    """
    from creativity_steering.models.factory import build_backend
    from creativity_steering.patching.evaluate import DEFAULT_BRIEFS
    from creativity_steering.patching.evaluate_stack import (
        build_all_layers_cells,
        build_multi_vector_cells,
        two_stage_cell_sweep,
    )

    cfg = load_config(config)
    vs, chosen = _load_vector_set(run, position)
    backend = build_backend(cfg)

    if layers:
        promising = [int(x) for x in layers.split(",")]
    else:
        from creativity_steering.patching.sweep import select_promising_layers

        sweep_path = run / "patching" / chosen / "sweep_results.json"
        if not sweep_path.exists():
            raise typer.BadParameter(
                f"no sweep results at {sweep_path}; run `sweep` first or pass --layers"
            )
        promising = select_promising_layers(json.loads(sweep_path.read_text()), n_promising)
    if not promising:
        raise typer.BadParameter("no promising layers to experiment on")

    a = cfg.patching.all_layers_alpha
    cells = build_all_layers_cells(
        vs, source_layers=promising[:2], alphas=[a, -a, 2 * a, -2 * a], rescale=cfg.patching.rescale
    )
    cells += build_multi_vector_cells(
        vs, layer_groups=[promising], alphas=[1.0, 2.0], rescale=cfg.patching.rescale
    )

    fc = cfg.filtering
    typer.echo(
        f"Exp-2 on promising layers {promising}: {len(cells)} cells, "
        f"judges on top-{cfg.patching.judge_top_k} (enable_judge={fc.enable_judge})"
    )
    result = two_stage_cell_sweep(
        backend, cells,
        briefs=DEFAULT_BRIEFS,
        n_samples=cfg.patching.eval_samples,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        enable_judge=fc.enable_judge,
        judge_top_k=cfg.patching.judge_top_k,
        generator_family=fc.generator_family,
        stage1_floor_percentile=fc.stage1_floor_percentile,
        enable_reward=fc.enable_reward,
        use_reward_fallback=fc.use_reward_fallback,
        stage1_device=cfg.device,
    )
    out_dir = run / "patching" / chosen
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiments_results.json").write_text(json.dumps(result.to_dict(), indent=2))
    best = result.best_cell()
    typer.echo(f"baseline composite={result.baseline.composite:.3f}  judged={result.judged_labels}")
    typer.echo(
        f"best experiment: {best['label']} composite={best['composite']:.3f} "
        f"(delta={best['delta']:+.3f}, judged={best['judged']})"
    )
    typer.echo(f"Saved Exp-2 results to {out_dir}")


@app.command()
def demo(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("artifacts/demo"),
) -> None:
    """Run the entire pipeline offline on the mock backend and print a summary."""
    from creativity_steering.analysis.oracle import StubOracle
    from creativity_steering.analysis.report import build_report
    from creativity_steering.config import FilterConfig, GenerationConfig, PatchingConfig
    from creativity_steering.derivation.pipeline import derive_vectors, save_derivation
    from creativity_steering.derivation.prompts import (
        CREATIVE_SYSTEM,
        NEUTRAL_SYSTEM,
        TOPICS,
        PromptSet,
    )
    from creativity_steering.models.factory import build_backend
    from creativity_steering.patching.multilayer import all_layers_single_vector
    from creativity_steering.patching.sweep import (
        per_layer_alpha_sweep,
        select_promising_layers,
    )
    from creativity_steering.scoring import HeuristicScorer

    cfg = ExperimentConfig(
        backend="mock", num_layers=24, hidden_dim=128, mock_creative_layer=12,
        generation=GenerationConfig(samples_per_prompt=3, max_new_tokens=240),
        filtering=FilterConfig(creative_min=0.55, neutral_max=0.45, min_per_class=0, scorer="heuristic"),
        patching=PatchingConfig(alphas=(-4.0, -1.0, 1.0, 4.0), eval_samples=4),
    )
    backend = build_backend(cfg)
    scorer = HeuristicScorer()
    prompts = PromptSet(CREATIVE_SYSTEM, NEUTRAL_SYSTEM, TOPICS)

    typer.echo("== Phase 1: derive ==")
    output = derive_vectors(backend, cfg, scorer, prompt_set=prompts)
    save_derivation(output, out)
    typer.echo(json.dumps(output.summary(), indent=2))

    typer.echo("\n== Phase 2: analyze ==")
    report = build_report(
        output.vector_set.vectors,
        oracle=StubOracle(reference_norms=output.vector_set.layer_norms),
    )
    typer.echo(f"candidate layers: {report.candidate_layers}  (planted at 12)")
    typer.echo(f"norm top={report.norm.top_layer}  cosine band={report.cosine.band}")

    typer.echo("\n== Phase 3: Experiment 1 (per-layer alpha sweep over candidates) ==")
    sweep = per_layer_alpha_sweep(
        backend, output.vector_set, scorer,
        alphas=cfg.patching.alphas, layers=report.candidate_layers, n_samples=cfg.patching.eval_samples,
    )
    best = sweep.best_cell()
    typer.echo(f"baseline={sweep.baseline.composite:.3f}  best: L{best['layer']} "
               f"alpha={best['alpha']} composite={best['composite']:.3f} (delta={best['delta']:+.3f})")

    typer.echo("\n== Phase 3: Experiment 2 (best vector at every layer) ==")
    source = select_promising_layers(sweep.to_dict(), 1)[0]
    typer.echo(f"promising source layer from sweep: L{source}")
    res = all_layers_single_vector(
        backend, output.vector_set, scorer, source_layer=source,
        alpha=cfg.patching.all_layers_alpha, n_samples=cfg.patching.eval_samples,
    )
    typer.echo(f"{res.label}: delta={res.delta:+.3f}")
    typer.echo(f"\nArtifacts written under {out}")


if __name__ == "__main__":
    app()
