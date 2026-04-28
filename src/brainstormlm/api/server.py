"""FastAPI server wrapping the BrainstormLM Generator protocol (plan-06).

Endpoints:
  - GET  /healthz           — liveness + promoted-adapter pointer.
  - POST /ask (SSE)         — streams meta → token(s) → done events.
  - POST /feedback          — attach a HumanRating to an existing trajectory.
  - GET  /metrics           — request counts, latency by arm, cache hit rate.

The model is never loaded at import time. ``build_app`` accepts a
``make_generator`` factory so tests can stub it; default is the real
``brainstormlm.model.factory.make_generator``.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from brainstormlm.api.logging_ import RequestLogger
from brainstormlm.api.routing import load_ab_config, resolve_arm
from brainstormlm.api.safety import (
    RateLimitMiddleware,
    TokenBucket,
    scrub_lessons,
    validate_user_prompt,
)
from brainstormlm.api.schemas import (
    AskRequest,
    FeedbackRequest,
    FeedbackResponse,
    HealthzResponse,
    MetricsResponse,
)
from brainstormlm.config import get_settings
from brainstormlm.feedback.evaluator import RUBRIC_VERSION
from brainstormlm.feedback.human import (
    append_human,
    attach_rating,
    new_trajectory_id,
    utc_now_iso,
)
from brainstormlm.feedback.trajectories import HumanRating, Trajectory
from brainstormlm.memory import LessonStore, augmented_system, retrieve
from brainstormlm.model.factory import make_generator as default_make_generator
from brainstormlm.training.promotion import read_promoted

GeneratorFactory = Callable[..., Any]

_SSE_CHUNK_TOKENS = 50  # words per simulated token event


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _chunk_response(text: str, *, chunk_size: int = _SSE_CHUNK_TOKENS) -> Iterator[str]:
    """Split a final response into roughly-token-sized windows for SSE pacing."""
    words = text.split(" ")
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i : i + chunk_size])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, sort_keys=True)}\n\n"


def build_app(
    *,
    make_generator: GeneratorFactory | None = None,
) -> FastAPI:
    """Build a FastAPI app. ``make_generator`` is overridable for tests."""
    settings = get_settings()
    factory = make_generator or default_make_generator
    app = FastAPI(title="BrainstormLM", version="0.6.0")

    bucket = TokenBucket(
        capacity=settings.rate_limit_capacity,
        refill_per_sec=settings.rate_limit_refill_per_sec,
    )
    app.add_middleware(RateLimitMiddleware, bucket=bucket)

    logger = RequestLogger(settings.server_log_path)

    @app.get("/healthz", response_model=HealthzResponse)
    def healthz() -> HealthzResponse:
        promoted = read_promoted(settings.promoted_adapter_path)
        return HealthzResponse(
            ok=True,
            promoted_adapter=promoted.adapter_path,
            promoted_steering=promoted.steering_path,
        )

    @app.post("/ask")
    def ask(req: AskRequest, request: Request) -> Any:
        try:
            validate_user_prompt(req.prompt, max_len=settings.user_prompt_max_chars)
        except ValueError as e:
            return JSONResponse({"error": "bad_prompt", "detail": str(e)}, status_code=400)

        ab_cfg = load_ab_config(settings.ab_config_path)
        resolution = resolve_arm(req.session_id, ab_cfg)

        promoted = read_promoted(settings.promoted_adapter_path)
        adapter_path = resolution.adapter_path
        steering_path = resolution.steering_path
        if adapter_path is None and promoted.adapter_path:
            adapter_path = Path(promoted.adapter_path)
        if steering_path is None and promoted.steering_path:
            steering_path = Path(promoted.steering_path)

        system: str | None = None
        memory_context_ids: list[str] = []
        if req.use_memory:
            store = LessonStore(settings.lessons_path)
            store.load()
            lessons = retrieve(
                store, req.prompt, k_success=req.k_success, k_failure=req.k_failure
            )
            scrubbed = scrub_lessons(lessons)
            system = augmented_system(None, scrubbed)
            memory_context_ids = [lesson.id for lesson in scrubbed]

        generator = factory(adapter_path, steering_path=steering_path)

        t0 = time.perf_counter()
        response_text = generator.generate(
            req.prompt,
            system=system,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        trajectory_id = new_trajectory_id()
        is_local = adapter_path is not None or steering_path is not None
        generator_model = (
            settings.hf_base_model if is_local else settings.together_base_model
        )

        trajectory = Trajectory(
            id=trajectory_id,
            timestamp=utc_now_iso(),
            domain="user",
            prompt=req.prompt,
            response=response_text,
            generator_model=generator_model,
            generator_temperature=req.temperature,
            prompter_model="(human)",
            evaluator_model=settings.evaluator_model,
            rubric_version=RUBRIC_VERSION,
            creativity=1,  # placeholder; agreement.py only counts rated rows
            logic=1,
            rationale="(awaiting human feedback)",
            failure_modes=[],
            memory_context_ids=memory_context_ids,
            adapter_path=str(adapter_path) if adapter_path else None,
            steering_path=str(steering_path) if steering_path else None,
            source="human",
            human_rating=None,
        )
        append_human(trajectory, path=settings.feedback_human_path)

        logger.log(
            {
                "route": "/ask",
                "request_id": uuid.uuid4().hex,
                "trajectory_id": trajectory_id,
                "session_id": req.session_id,
                "arm": resolution.arm,
                "adapter_path": str(adapter_path) if adapter_path else None,
                "steering_path": str(steering_path) if steering_path else None,
                "latency_ms": round(latency_ms, 2),
                "status": 200,
            }
        )

        def _stream() -> Iterator[str]:
            yield _sse(
                "meta",
                {
                    "trajectory_id": trajectory_id,
                    "arm": resolution.arm,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                },
            )
            for chunk in _chunk_response(response_text):
                yield _sse("token", {"text": chunk})
            yield _sse(
                "done",
                {
                    "trajectory_id": trajectory_id,
                    "creativity": None,
                    "logic": None,
                },
            )

        return StreamingResponse(_stream(), media_type="text/event-stream")

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest) -> FeedbackResponse:
        rating = HumanRating(
            thumbs=req.thumbs,
            rating=req.rating,
            critique=req.critique,
            rated_at=utc_now_iso(),
        )
        try:
            attach_rating(req.trajectory_id, rating, path=settings.feedback_human_path)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        logger.log(
            {
                "route": "/feedback",
                "trajectory_id": req.trajectory_id,
                "thumbs": req.thumbs,
                "rating": req.rating,
                "status": 200,
            }
        )
        return FeedbackResponse(ok=True)

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        records = logger.snapshot()
        ask_records = [r for r in records if r.get("route") == "/ask"]
        now = dt.datetime.now(dt.timezone.utc)
        cutoff = now - dt.timedelta(seconds=60)

        def _ts(r: dict) -> dt.datetime | None:
            try:
                return dt.datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                return None

        recent = [r for r in ask_records if (t := _ts(r)) is not None and t >= cutoff]

        latency_by_arm: dict[str, list[float]] = {}
        for r in ask_records:
            arm = r.get("arm", "unknown")
            latency = r.get("latency_ms")
            if latency is None:
                continue
            latency_by_arm.setdefault(arm, []).append(float(latency))

        mean_latency = {
            arm: round(sum(vs) / len(vs), 2) for arm, vs in latency_by_arm.items() if vs
        }

        cache_hit_rate = 0.0  # placeholder until we wire generator_cached into logs

        return MetricsResponse(
            total_requests=len(ask_records),
            requests_last_60s=len(recent),
            mean_latency_ms_by_arm=mean_latency,
            cache_hit_rate=cache_hit_rate,
        )

    # Mount the static frontend if present (development convenience).
    frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


# Default app for ``uvicorn brainstormlm.api.server:app`` invocations.
app = build_app()
