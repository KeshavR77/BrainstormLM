"""SSE streaming endpoint for phased brainstorm generation.

POST /brainstorm/stream returns Server-Sent Events showing the two-phase
generation process: diverse exploration followed by quality synthesis.

Two execution modes:
- ``CONCEPTIAL_DEMO_MODE=true`` (default for the live demo): high-quality
  pre-written ideas streamed at realistic pacing via the Conceptial demo
  source. Project-scoped memory persists across regenerations through
  ``IdeaMemoryStore``. No model load required.
- Default (with ML stack available): real Corpius generation per the
  existing ``_live_stream`` path. Kept for future swap-in.

The demo-mode path is the production default because Qwen 3B on Apple
Silicon takes ~30s to cold-load and per-idea latency is 5-8s on MPS —
which makes the live demo feel slower than it needs to. The cached
14B + curated pack outputs we have on disk already match or beat what
a live 3B run would produce.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from brainstormlm.api.schemas import BrainstormRequest, MemoryCountResponse
from brainstormlm.api.safety import validate_user_prompt
from brainstormlm.config import get_settings
from brainstormlm.feedback.human import append_human, new_trajectory_id, utc_now_iso
from brainstormlm.feedback.evaluator import RUBRIC_VERSION
from brainstormlm.feedback.trajectories import Trajectory
from brainstormlm.conceptial.idea_store import SOURCE_GENERATED, SOURCE_SEEDED
from brainstormlm.conceptial.streaming_source import (
    ConceptialStreamingSource,
    get_demo_store,
)


def _conceptial_enabled() -> bool:
    """The Conceptial streaming source is active unless an opt-out is set.

    Set ``CONCEPTIAL_STREAM_DISABLED=1`` to fall back to the legacy
    ``_smoke_ideas`` path or the real-model ``_live_stream`` path.
    """
    return os.environ.get("CONCEPTIAL_STREAM_DISABLED", "").lower() not in (
        "1",
        "true",
        "yes",
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, sort_keys=True)}\n\n"


def build_streaming_router(
    *,
    factory: Any,
    demo_mode: bool,
    ml_available: bool,
) -> APIRouter:
    router = APIRouter()
    settings = get_settings()

    # Import smoke ideas generator from the main server module
    from brainstormlm.api.server import _smoke_ideas

    def _persist(traj: Trajectory) -> None:
        try:
            append_human(traj, path=settings.feedback_human_path)
        except OSError:
            pass

    def _make_trajectory(prompt: str, text: str, temp: float, tid: str) -> Trajectory:
        return Trajectory(
            id=tid,
            timestamp=utc_now_iso(),
            domain="brainstorm",
            prompt=prompt,
            response=text,
            generator_model="conceptial-stream",
            generator_temperature=temp,
            prompter_model="(human)",
            evaluator_model=settings.evaluator_model,
            rubric_version=RUBRIC_VERSION,
            creativity=1,
            logic=1,
            rationale="(awaiting feedback)",
            failure_modes=[],
            memory_context_ids=[],
            source="human",
            human_rating=None,
        )

    @router.post("/brainstorm/stream")
    def brainstorm_stream(req: BrainstormRequest) -> StreamingResponse:
        try:
            validate_user_prompt(req.prompt, max_len=settings.user_prompt_max_chars)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # Conceptial streaming source — project-scoped, anti-mode aware,
        # zero model-load latency. Default path for the live demo.
        if _conceptial_enabled():
            return StreamingResponse(
                _conceptial_stream(req),
                media_type="text/event-stream",
            )

        if demo_mode or not ml_available:
            return StreamingResponse(
                _demo_stream(req.prompt, req.n_ideas),
                media_type="text/event-stream",
            )

        return StreamingResponse(
            _live_stream(req),
            media_type="text/event-stream",
        )

    @router.get("/v1/memory/count", response_model=MemoryCountResponse)
    def memory_count(project_id: str) -> MemoryCountResponse:
        """Return seeded / generated / total memory size for a project.

        Powers the 'Memory: N seeded · M generated' chip on the canvas.
        """
        store = get_demo_store()
        breakdown = store.count_by_source(project_id)
        seeded = int(breakdown.get(SOURCE_SEEDED, 0))
        generated = int(breakdown.get(SOURCE_GENERATED, 0))
        return MemoryCountResponse(
            project_id=project_id,
            seeded=seeded,
            generated=generated,
            total=seeded + generated,
        )

    @router.post("/v1/memory/reset")
    def memory_reset(project_id: str) -> dict[str, int | str]:
        """Wipe a project's idea memory. Used by the 'reset' demo control."""
        store = get_demo_store()
        deleted = store.delete_project(project_id)
        return {"project_id": project_id, "deleted": int(deleted)}

    def _conceptial_stream(req: BrainstormRequest) -> Iterator[str]:
        """Stream ideas via the project-scoped Conceptial demo source.

        Events:
        - ``phase`` (diverse_pool, synthesizing, complete)
        - ``raw_idea`` per emitted idea (with idea_id, exclusion list snapshot)
        - ``progress`` between phases
        - ``final_idea`` for each finalist
        - ``memory`` final counts on done
        - ``done`` with trajectory_ids
        """
        store = get_demo_store()
        project_id = req.project_id or _default_project_id(req.prompt)

        # When the client didn't pass a project_id, treat each call as a fresh
        # session — otherwise the anonymous project accumulates state across
        # unrelated requests and exhausts the catalogue. Explicit project_id
        # callers (the canvas) get persistent memory by default.
        if req.reset_project or not req.project_id:
            store.delete_project(project_id)

        n = max(1, min(req.n_ideas, 8))
        # phase first so consumers that expect that as the opening event still work
        yield _sse("phase", {"phase": "diverse_pool", "expected": n})

        # Emit initial memory state so the UI chip shows pre-seeded counts immediately
        initial = store.count_by_source(project_id)
        yield _sse(
            "memory",
            {
                "project_id": project_id,
                "seeded": int(initial.get(SOURCE_SEEDED, 0)),
                "generated": int(initial.get(SOURCE_GENERATED, 0)),
                "total": sum(initial.values()),
            },
        )

        # Pacing: realistic SSE cadence by default. Set CONCEPTIAL_DEMO_FAST=1
        # to fire ideas instantly (useful for dev iteration; the live demo
        # wants the pacing for visible streaming).
        fast_mode = os.environ.get("CONCEPTIAL_DEMO_FAST", "").lower() in (
            "1",
            "true",
            "yes",
        )
        source = ConceptialStreamingSource(
            store=store,
            project_id=project_id,
            pacing=not fast_mode,
        )

        emitted: list[dict] = []
        for i, idea in enumerate(source.stream(prompt=req.prompt, n=n)):
            tid = new_trajectory_id()
            raw = {
                "index": i,
                "summary": idea.summary,
                "content": idea.content,
                "trajectory_id": tid,
                "idea_id": idea.idea_id,
                "pack": idea.pack,
                "excluded_summaries": list(idea.excluded_summaries),
            }
            emitted.append(raw)
            yield _sse("raw_idea", raw)
            traj = _make_trajectory(req.prompt, idea.content, 0.8, tid)
            _persist(traj)

            # Update memory chip count after each emit
            counts = store.count_by_source(project_id)
            yield _sse(
                "memory",
                {
                    "project_id": project_id,
                    "seeded": int(counts.get(SOURCE_SEEDED, 0)),
                    "generated": int(counts.get(SOURCE_GENERATED, 0)),
                    "total": sum(counts.values()),
                },
            )

        # No GPT-4o-mini scoring round here — the curated packs are already
        # quality-graded. Synthesis phase exists for UI symmetry only.
        yield _sse("phase", {"phase": "synthesizing", "pool_size": len(emitted)})
        yield _sse("progress", {"progress": 0.5})
        yield _sse("progress", {"progress": 1.0})
        yield _sse("phase", {"phase": "complete"})

        for raw in emitted:
            yield _sse(
                "final_idea",
                {
                    "summary": raw["summary"],
                    "content": raw["content"],
                    "trajectory_id": raw["trajectory_id"],
                    "idea_id": raw["idea_id"],
                },
            )

        yield _sse(
            "done",
            {
                "project_id": project_id,
                "trajectory_ids": [r["trajectory_id"] for r in emitted],
                "idea_ids": [r["idea_id"] for r in emitted],
            },
        )

    def _default_project_id(prompt: str) -> str:
        """Hash-based fallback project_id if the client didn't pass one.

        Same prompt → same project → cumulative anti-mode memory.
        """
        import hashlib
        return "anon-" + hashlib.sha256(prompt.encode()).hexdigest()[:12]

    def _demo_stream(prompt: str, n_ideas: int) -> Iterator[str]:
        smoke = _smoke_ideas(prompt, n_ideas)
        expected = len(smoke)

        yield _sse("phase", {"phase": "diverse_pool", "expected": expected})

        for i, idea in enumerate(smoke):
            yield _sse("raw_idea", {
                "index": i,
                "summary": idea.summary,
                "content": idea.content,
                "trajectory_id": idea.trajectory_id,
            })
            traj = _make_trajectory(prompt, idea.content, 0.0, idea.trajectory_id)
            _persist(traj)

        yield _sse("phase", {"phase": "synthesizing", "pool_size": expected})
        yield _sse("progress", {"progress": 0.5})
        yield _sse("progress", {"progress": 1.0})
        yield _sse("phase", {"phase": "complete"})

        for idea in smoke:
            yield _sse("final_idea", {
                "summary": idea.summary,
                "content": idea.content,
                "trajectory_id": idea.trajectory_id,
            })

        yield _sse("done", {
            "trajectory_ids": [idea.trajectory_id for idea in smoke],
        })

    def _live_stream(req: BrainstormRequest) -> Iterator[str]:
        corpius_config = settings.corpius_config_path if settings.corpius_config_path.exists() else None
        temperatures = [0.9, 1.0, 1.1, 1.2]
        n_diverse = max(req.n_ideas, 8)

        yield _sse("phase", {"phase": "diverse_pool", "expected": n_diverse})

        raw_ideas = []
        for i in range(n_diverse):
            temp = temperatures[i % len(temperatures)]
            gen = factory(None, corpius_config_path=corpius_config)
            full_prompt = req.prompt
            if req.context:
                full_prompt = f"{req.context}\n\n{req.prompt}"
            text = gen.generate(full_prompt, temperature=temp, max_tokens=600)
            first_line = text.split("\n")[0].strip()
            summary = first_line[:60] if len(first_line) <= 60 else first_line[:57] + "..."
            tid = new_trajectory_id()

            traj = _make_trajectory(req.prompt, text, temp, tid)
            _persist(traj)

            idea_data = {"index": i, "summary": summary, "content": text, "trajectory_id": tid}
            raw_ideas.append(idea_data)
            yield _sse("raw_idea", idea_data)

        yield _sse("phase", {"phase": "synthesizing", "pool_size": len(raw_ideas)})
        yield _sse("progress", {"progress": 0.5})

        # Simple quality heuristic: take ideas with longest content (proxy for depth)
        sorted_ideas = sorted(raw_ideas, key=lambda x: len(x["content"]), reverse=True)
        final_ideas = sorted_ideas[:req.n_ideas]

        yield _sse("progress", {"progress": 1.0})
        yield _sse("phase", {"phase": "complete"})

        for idea in final_ideas:
            yield _sse("final_idea", {
                "summary": idea["summary"],
                "content": idea["content"],
                "trajectory_id": idea["trajectory_id"],
            })

        yield _sse("done", {
            "trajectory_ids": [idea["trajectory_id"] for idea in final_ideas],
        })

    return router
