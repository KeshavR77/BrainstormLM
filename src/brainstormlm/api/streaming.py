"""SSE streaming endpoint for phased brainstorm generation.

POST /brainstorm/stream returns Server-Sent Events showing the two-phase
generation process: diverse exploration followed by quality synthesis.

Default path uses the Conceptial streaming source — high-quality curated
ideas streamed at realistic pacing with project-scoped memory via
``IdeaMemoryStore``. No model load required.

Fallback (``CONCEPTIAL_STREAM_DISABLED=1``): deterministic smoke-pack
ideas from ``server._smoke_ideas``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

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
    return os.environ.get("CONCEPTIAL_STREAM_DISABLED", "").lower() not in (
        "1",
        "true",
        "yes",
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, sort_keys=True)}\n\n"


def build_streaming_router() -> APIRouter:
    router = APIRouter()
    settings = get_settings()

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

        if _conceptial_enabled():
            return StreamingResponse(
                _conceptial_stream(req),
                media_type="text/event-stream",
            )

        return StreamingResponse(
            _demo_stream(req.prompt, req.n_ideas),
            media_type="text/event-stream",
        )

    @router.get("/v1/memory/count", response_model=MemoryCountResponse)
    def memory_count(project_id: str) -> MemoryCountResponse:
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
        store = get_demo_store()
        deleted = store.delete_project(project_id)
        return {"project_id": project_id, "deleted": int(deleted)}

    def _conceptial_stream(req: BrainstormRequest) -> Iterator[str]:
        store = get_demo_store()
        project_id = req.project_id or _default_project_id(req.prompt)

        if req.reset_project or not req.project_id:
            store.delete_project(project_id)

        n = max(1, min(req.n_ideas, 8))
        yield _sse("phase", {"phase": "diverse_pool", "expected": n})

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

    return router
