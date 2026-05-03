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

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
    BrainstormRequest,
    BrainstormResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthzResponse,
    IdeaBranch,
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
from brainstormlm.memory.ingest import ingest_human_feedback

# ML-heavy imports are optional — not present in demo/lightweight deployments.
try:
    from brainstormlm.model.factory import make_generator as default_make_generator
    from brainstormlm.training.promotion import read_promoted as _read_promoted
    _ML_AVAILABLE = True
except ImportError:
    default_make_generator = None  # type: ignore[assignment]
    _read_promoted = None  # type: ignore[assignment]
    _ML_AVAILABLE = False

import os as _os
_DEMO_MODE: bool = _os.environ.get("CONCEPTIAL_DEMO", "").lower() in ("1", "true", "yes")

GeneratorFactory = Callable[..., Any]

_SSE_CHUNK_TOKENS = 50  # words per simulated token event

# ---------------------------------------------------------------------------
# Smoke idea packs — returned by /brainstorm in demo mode.
# Eight packs of 4 ideas each, spanning different strategic dimensions.
# Pack is chosen deterministically by SHA256(prompt) so the same prompt
# always returns the same pack, giving a consistent demo experience.
# ---------------------------------------------------------------------------
import hashlib as _hashlib

_SMOKE_PACKS: list[list[dict]] = [
    [   # Pack 0 — Distribution / Market Entry
        dict(summary="Invert the Value Chain", content="Instead of selling through existing channels, become the infrastructure layer those enterprises depend on. Charge for API access, usage, and white-label rights — not seats. Your customers' growth becomes your growth, without any of their distribution overhead."),
        dict(summary="Niche Domination First", content="Pick the single most underserved vertical where your solution creates 10x value — not 2x. Win it completely: 80% market share, reference customers, category leadership. Only then expand. Premature horizontality is how startups die slowly."),
        dict(summary="Community as the Moat", content="Don't build software — build the network of people who need to find each other. The platform becomes defensible when users have more value in the community than in the tool itself. Your moat is social, not technical."),
        dict(summary="Freemium as Intelligence Engine", content="Give the core product away free — not as a loss leader, but as a data-collection mechanism. Free users generate the signal that makes the paid product 10x better. The free tier is R&D, not charity."),
    ],
    [   # Pack 1 — Platform / Ecosystem
        dict(summary="Open Core, Closed Leverage", content="Open-source the foundational layer to build trust, adoption, and a contributor network. Monetize the enterprise integrations, compliance controls, and managed hosting that enterprises need but can't build themselves. The model is proven: Red Hat, HashiCorp, Elastic."),
        dict(summary="Become Competitors' Infrastructure", content="Build the layer your competitors depend on. If you supply the tools, data, or APIs that other players in the space need, you're in every deal without winning any of them directly. AWS wasn't built by fighting Oracle — it was built by being indispensable."),
        dict(summary="Regulatory Alignment as Moat", content="Position now as the compliance-friendly option for an imminent regulatory shift. If you're the only player already meeting tomorrow's standards, you inherit the market the day the regulation passes — while competitors scramble to retrofit."),
        dict(summary="Partnership Ecosystem over Direct Sales", content="Don't build a sales team — build a partner ecosystem. If ten adjacent players integrate your product and resell it, you get their distribution, their customers, and their trust — without the cost of hiring the sales team to reach them."),
    ],
    [   # Pack 2 — Growth / GTM
        dict(summary="Bottoms-Up Viral Loop", content="Seed with individual contributors — the people who do the work, not the people who approve the budget. When individual users find value, they evangelize upward. The enterprise deal gets pulled by demand instead of pushed by sales. Slack, Figma, Notion all ran this playbook."),
        dict(summary="Content Moat Before Product", content="Publish the research and intellectual infrastructure of your category before most companies are paying attention. When the market wakes up to the problem, you're already the authority. The content is the lead-generation engine; the product closes."),
        dict(summary="Proof-Selling over Feature-Selling", content="Lead every sales conversation with a live benchmark — your product head-to-head against the incumbent on the customer's own data. If the product is genuinely better, the sale closes itself. If it doesn't, you need to know that before scaling GTM."),
        dict(summary="Geographic Arbitrage", content="Win in markets the incumbents are ignoring — not because they're weak markets, but because they're too operationally inconvenient for large companies to serve well. Own them completely, then use that as a stepping stone to the markets everyone else is fighting over."),
    ],
    [   # Pack 3 — Revenue / Business Model
        dict(summary="Outcome-Based Pricing", content="Charge as a percentage of measurable outcomes — not as a SaaS subscription. If your product generates $1M in innovation revenue for a customer, taking 5% is easy to justify and hard to cancel. It aligns your incentives perfectly with the customer's success."),
        dict(summary="Data Network Effect", content="Structure the product so each user's activity makes the product better for every other user. As the network grows, the value of each individual's contribution increases — creating a compounding moat that pure software products can't replicate."),
        dict(summary="Usage Inversion Pricing", content="Flip the conventional SaaS model: free at high usage, premium at low usage. Power users subsidise the base tier through their contributions to the shared model. Heavy users get more out than they put in — and they make the product indispensable."),
        dict(summary="Cooperative Ownership Model", content="Turn your best customers into co-owners. Revenue sharing for contributors, equity for early adopters, governance rights for power users. The community becomes the defensible moat — nobody leaves when they own a piece of what they helped build."),
    ],
    [   # Pack 4 — Product Strategy
        dict(summary="Anti-Features as Differentiator", content="Define the product by what it deliberately refuses to do. Every feature your competitors add that you don't makes your product cleaner, faster, and easier to advocate for. The constraint is the differentiation. Say no loudly and publicly."),
        dict(summary="Async-First Architecture", content="Remove every real-time dependency from the core product. Deep work happens asynchronously. Collaboration shouldn't require everyone online simultaneously. Build for the way high-performers actually work, not for the demo that looks impressive."),
        dict(summary="AI Co-Pilot as Core Primitive", content="Don't add AI as a feature — make it a collaborator that learns each user's style, preferences, and constraints over time. The product gets more valuable the longer you use it, creating switching costs that pure SaaS products can't match."),
        dict(summary="Physical-Digital Bridge", content="Connect the digital product to physical-world outcomes in a way software-only competitors can't replicate. When your product changes what happens in the real world — not just on-screen — it becomes genuinely indispensable."),
    ],
    [   # Pack 5 — Competitive Positioning
        dict(summary="Category Creation", content="Don't compete in an existing category — name and own a new one. If you define the category, you define the evaluation criteria. Every competitor who enters is playing on your terms, using your language, fighting for second place in a race you invented."),
        dict(summary="Radical Transparency as Moat", content="Publish everything: pricing, methodology, performance benchmarks, roadmap, even mistakes. Radical transparency is a trust signal that large incumbents can't replicate without exposing their own weaknesses. It makes you the default credible option."),
        dict(summary="Founder-Led Sales as Permanent Strategy", content="Keep founders selling even after you could afford not to. The intelligence density in those conversations — about objections, use cases, competitive framing — is irreplaceable. Sales as a permanent founder function, not a stopgap."),
        dict(summary="Developer-First Evangelism", content="Win the engineers first. They have disproportionate influence over technology decisions, share tools they love, and build integrations that expand your distribution. The CTO relationship follows the engineering relationship, not the other way around."),
    ],
    [   # Pack 6 — Innovation / R&D
        dict(summary="Constraint-Driven Innovation", content="Impose a hard constraint on the design process — a budget ceiling, a performance limit, a regulatory requirement — and let it force genuinely novel solutions. The constraint is not the enemy of creativity; it's the engine of it."),
        dict(summary="Adjacent Possible Exploration", content="Map the problem space one step beyond what's currently buildable, and build toward that frontier. The best competitive position is always slightly ahead of where the technology currently permits. You're not early — you're on time."),
        dict(summary="Failure Archaeology", content="Systematically study every previous attempt to solve this problem and why it failed. The graveyard of prior attempts is the most valuable research asset in any market. The patterns of failure predict the shape of the solution."),
        dict(summary="Cross-Domain Pattern Transfer", content="Find the domain where this problem was already solved — then translate the solution. Most hard problems are instances of general patterns. The insight rarely comes from within the domain; it comes from someone who's seen it solved elsewhere."),
    ],
    [   # Pack 7 — Organisation / Team
        dict(summary="Public Roadmap as Trust Engine", content="Build your product publicly: publish the roadmap, share the reasoning, invite public comment on priorities. The customers who care most become your most powerful advocates — and they'll recruit customers you'd never reach through conventional marketing."),
        dict(summary="Customer Advisory Council with Teeth", content="Give your ten most valuable customers genuine input over the product roadmap — not just feedback sessions, but actual veto power on key decisions. The depth of partnership you build is a retention mechanism no competitor can buy their way around."),
        dict(summary="Distributed Expertise Network", content="Replace generalist hires with domain specialists engaged on a project basis. Generalists optimise for process; specialists move the frontier. The team that wins is the one that can access the sharpest thinking in each domain at the moment it's needed."),
        dict(summary="Performance-Equity Compensation", content="Tie compensation directly to measurable customer outcomes, not to organisational metrics. When team incentives align exactly with customer value creation, every decision naturally optimises for the right thing — and you attract people who believe in what you're building."),
    ],
]


def _smoke_ideas(prompt: str, n: int) -> list[IdeaBranch]:
    """Pick a deterministic idea pack based on the prompt hash, assigning fresh trajectory IDs."""
    pack_idx = int(_hashlib.sha256(prompt.encode()).hexdigest()[:4], 16) % len(_SMOKE_PACKS)
    base = _SMOKE_PACKS[pack_idx][:n]
    return [
        IdeaBranch(summary=b["summary"], content=b["content"], trajectory_id=new_trajectory_id())
        for b in base
    ]


def _noop_read_promoted(path: Any) -> Any:
    """Minimal stand-in for read_promoted when ML deps are absent."""
    from types import SimpleNamespace
    return SimpleNamespace(adapter_path=None, steering_path=None)


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
    _promote = _read_promoted if _ML_AVAILABLE else _noop_read_promoted
    app = FastAPI(title="Conceptial", version="0.7.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:4173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    bucket = TokenBucket(
        capacity=settings.rate_limit_capacity,
        refill_per_sec=settings.rate_limit_refill_per_sec,
    )
    app.add_middleware(RateLimitMiddleware, bucket=bucket)

    logger = RequestLogger(settings.server_log_path)

    @app.get("/healthz", response_model=HealthzResponse)
    def healthz() -> HealthzResponse:
        promoted = _promote(settings.promoted_adapter_path)
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

        promoted = _promote(settings.promoted_adapter_path)
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

        generator = factory(
            adapter_path,
            steering_path=steering_path,
            corpius_config_path=resolution.corpius_config_path,
        )

        t0 = time.perf_counter()
        response_text = generator.generate(
            req.prompt,
            system=system,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        trajectory_id = new_trajectory_id()
        is_corpius = resolution.corpius_config_path is not None
        is_local = adapter_path is not None or steering_path is not None or is_corpius
        if is_corpius:
            generator_model = "Qwen/Qwen2.5-3B-Instruct (Corpius 1)"
        elif is_local:
            generator_model = settings.hf_base_model
        else:
            generator_model = settings.together_base_model

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
    def feedback(req: FeedbackRequest, background: BackgroundTasks) -> FeedbackResponse:
        rating = HumanRating(
            thumbs=req.thumbs,
            rating=req.rating,
            critique=req.critique,
            rated_at=utc_now_iso(),
        )
        try:
            rated = attach_rating(
                req.trajectory_id, rating, path=settings.feedback_human_path
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        # Fold thumbs-up rows into the ReasoningBank in the background so the
        # /feedback response stays instant. ingest_human_feedback no-ops on
        # thumbs="down" and swallows its own errors.
        background.add_task(
            ingest_human_feedback, rated, lessons_path=settings.lessons_path
        )

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

    def _persist_brainstorm_trajectory(traj: "Trajectory") -> None:
        """Attempt to write a brainstorm trajectory; silently swallow FS errors (e.g. Render ephemeral FS)."""
        try:
            append_human(traj, path=settings.feedback_human_path)
        except OSError:
            pass

    @app.post("/brainstorm", response_model=BrainstormResponse)
    def brainstorm(req: BrainstormRequest) -> BrainstormResponse:
        try:
            validate_user_prompt(req.prompt, max_len=settings.user_prompt_max_chars)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if _DEMO_MODE or not _ML_AVAILABLE:
            smoke = _smoke_ideas(req.prompt, req.n_ideas)
            # Persist stub trajectories for each smoke idea (best-effort)
            for idea in smoke:
                traj = Trajectory(
                    id=idea.trajectory_id,
                    timestamp=utc_now_iso(),
                    domain="brainstorm",
                    prompt=req.prompt,
                    response=idea.content,
                    generator_model="conceptial-demo",
                    generator_temperature=0.0,
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
                _persist_brainstorm_trajectory(traj)
            return BrainstormResponse(ideas=smoke)

        system: str | None = None
        memory_context_ids: list[str] = []
        if req.use_memory:
            store = LessonStore(settings.lessons_path)
            store.load()
            lessons = retrieve(store, req.prompt, k_success=3, k_failure=2)
            scrubbed = scrub_lessons(lessons)
            system = augmented_system(None, scrubbed)
            memory_context_ids = [lesson.id for lesson in scrubbed]

        corpius_config = settings.corpius_config_path if settings.corpius_config_path.exists() else None
        temperatures = [0.9, 1.0, 1.1, 1.2]
        ideas: list[IdeaBranch] = []
        for i in range(req.n_ideas):
            temp = temperatures[i % len(temperatures)]
            gen = factory(None, corpius_config_path=corpius_config)
            full_prompt = req.prompt
            if req.context:
                full_prompt = f"{req.context}\n\n{req.prompt}"
            text = gen.generate(full_prompt, system=system, temperature=temp, max_tokens=600)
            first_line = text.split("\n")[0].strip()
            summary = first_line[:60] if len(first_line) <= 60 else first_line[:57] + "..."

            tid = new_trajectory_id()
            traj = Trajectory(
                id=tid,
                timestamp=utc_now_iso(),
                domain="brainstorm",
                prompt=req.prompt,
                response=text,
                generator_model="Qwen/Qwen2.5-3B-Instruct (Corpius 1)" if corpius_config else settings.together_base_model,
                generator_temperature=temp,
                prompter_model="(human)",
                evaluator_model=settings.evaluator_model,
                rubric_version=RUBRIC_VERSION,
                creativity=1,
                logic=1,
                rationale="(awaiting feedback)",
                failure_modes=[],
                memory_context_ids=memory_context_ids,
                source="human",
                human_rating=None,
            )
            _persist_brainstorm_trajectory(traj)
            ideas.append(IdeaBranch(summary=summary, content=text, trajectory_id=tid))

        return BrainstormResponse(ideas=ideas)

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

    # Mount the built React app when dist/ exists; dev uses the Vite dev server.
    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


# Default app for ``uvicorn brainstormlm.api.server:app`` invocations.
app = build_app()
