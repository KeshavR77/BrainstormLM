"""SSE streaming endpoint for phased brainstorm generation.

POST /brainstorm/stream returns Server-Sent Events showing the two-phase
generation process: Corpius diverse exploration followed by quality synthesis.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from brainstormlm.api.schemas import BrainstormRequest
from brainstormlm.api.safety import validate_user_prompt
from brainstormlm.config import get_settings
from brainstormlm.feedback.human import append_human, new_trajectory_id, utc_now_iso
from brainstormlm.feedback.evaluator import RUBRIC_VERSION
from brainstormlm.feedback.trajectories import Trajectory


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

        if demo_mode or not ml_available:
            return StreamingResponse(
                _demo_stream(req.prompt, req.n_ideas),
                media_type="text/event-stream",
            )

        return StreamingResponse(
            _live_stream(req),
            media_type="text/event-stream",
        )

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
