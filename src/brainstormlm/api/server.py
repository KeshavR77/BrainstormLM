"""FastAPI server for Conceptial brainstorming.

Endpoints:
  - GET  /healthz           — liveness check.
  - POST /feedback          — attach a HumanRating to an existing trajectory.
  - POST /brainstorm        — non-streaming idea generation (demo packs).
  - POST /brainstorm/stream — phased SSE streaming (primary product path).
  - GET  /v1/memory/count   — per-project memory breakdown.
  - POST /v1/memory/reset   — wipe a project's idea memory.
  - GET  /metrics           — request counts, latency, cache hit rate.
"""

from __future__ import annotations

import datetime as dt
import hashlib as _hashlib
import json
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from brainstormlm.api.logging_ import RequestLogger
from brainstormlm.api.safety import (
    RateLimitMiddleware,
    TokenBucket,
    validate_user_prompt,
)
from brainstormlm.api.schemas import (
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
from brainstormlm.memory.ingest import ingest_human_feedback


# ---------------------------------------------------------------------------
# Smoke idea packs — returned by /brainstorm in demo mode and used as
# fallback by streaming.py when the Conceptial source is opt-out disabled.
# ---------------------------------------------------------------------------

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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, sort_keys=True)}\n\n"


def build_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Conceptial", version="0.8.0")

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
        return HealthzResponse(ok=True)

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

    def _persist_brainstorm_trajectory(traj: Trajectory) -> None:
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

        smoke = _smoke_ideas(req.prompt, req.n_ideas)
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

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        records = logger.snapshot()
        now = dt.datetime.now(dt.timezone.utc)
        cutoff = now - dt.timedelta(seconds=60)

        def _ts(r: dict) -> dt.datetime | None:
            try:
                return dt.datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                return None

        recent = [r for r in records if (t := _ts(r)) is not None and t >= cutoff]

        latencies = [float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]
        mean_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        return MetricsResponse(
            total_requests=len(records),
            requests_last_60s=len(recent),
            mean_latency_ms=mean_latency,
            cache_hit_rate=None,  # not instrumented; report null rather than a fake 0.0
        )

    from brainstormlm.api.streaming import build_streaming_router
    streaming_router = build_streaming_router()
    app.include_router(streaming_router)

    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


app = build_app()
