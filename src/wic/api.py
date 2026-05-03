"""
api.py — FastAPI routers for the Recursive What-If Cascade (WIC) Engine.

Endpoints
─────────
POST  /session             → Create a new session
POST  /session/{id}/step   → Advance the session state machine by one step
GET   /session/{id}/stream → SSE stream of the Simulator output
GET   /session/{id}        → Return full session state (for frontend hydration)
"""

import json
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from brainstormlm.wic.database import get_pool
from brainstormlm.wic.llm_service import call_simulator
from brainstormlm.wic.models import SessionState
from brainstormlm.wic.orchestrator import (
    _build_problem_context,
    _persist_session,
    run_wic_step,
)


router = APIRouter(prefix="/session", tags=["session"])


# ═══════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════


class CreateSessionRequest(BaseModel):
    user_id: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: UUID


class StepRequest(BaseModel):
    user_input: str | None = None
    selected_inversion_id: UUID | None = None
    action: str | None = None  # "recurse" | "synthesize"


# ═══════════════════════════════════════════════════════════════════════
# IN-MEMORY SESSION STORE  (backed by Postgres for persistence)
# ═══════════════════════════════════════════════════════════════════════

_sessions: dict[UUID, SessionState] = {}


async def _load_or_get_session(session_id: UUID) -> SessionState:
    """
    Return the in-memory session if cached, otherwise try to
    reconstruct from Postgres.  Raises 404 if not found.
    """
    if session_id in _sessions:
        return _sessions[session_id]

    # Try to load from DB
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM sessions WHERE session_id = $1", session_id
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Re-hydrate a minimal SessionState from the DB row.
    # Full node/edge re-hydration from DB can be added later; for now
    # the in-memory store is the authoritative source during a session.
    state = SessionState(
        session_id=row["session_id"],
        user_id=row["user_id"],
        status=row["status"],
        current_depth=row["current_depth"],
    )
    _sessions[session_id] = state
    return state


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════


@router.post("", response_model=CreateSessionResponse)
async def create_session(body: CreateSessionRequest):
    """Initialise a brand-new WIC Session."""
    state = SessionState(
        user_id=body.user_id,
        status="awaiting_input",
    )
    _sessions[state.session_id] = state
    await _persist_session(state)
    return CreateSessionResponse(session_id=state.session_id)


@router.post("/{session_id}/step")
async def step_session(session_id: UUID, body: StepRequest):
    """
    Advance a session by one step through the WIC state machine.

    The body fields required depend on the current session status:
    - ``awaiting_input``                → ``user_input`` (str)
    - ``awaiting_selection``            → ``selected_inversion_id`` (UUID)
    - ``awaiting_recurse_or_synthesize``→ ``action`` ("recurse" | "synthesize")
    - Other statuses advance without extra input.
    """
    state = await _load_or_get_session(session_id)

    try:
        updated = await run_wic_step(
            state,
            user_input=body.user_input,
            selected_inversion_id=body.selected_inversion_id,
            action=body.action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _sessions[session_id] = updated
    return updated.model_dump(mode="json")


@router.get("/{session_id}/stream")
async def stream_simulation(
    session_id: UUID,
    inversion_statement: str = Query(...),
    implication: str = Query(...),
):
    """
    Server-Sent Events (SSE) endpoint that streams Simulator output
    directly to the frontend.

    Query params:
    - ``inversion_statement`` — the "What if …?" string
    - ``implication``         — the one-line implication

    Each chunk is sent as ``data: <json>\n\n``.
    The final event is ``data: [DONE]\n\n``.
    """
    state = await _load_or_get_session(session_id)
    full_context = _build_problem_context(state)

    async def event_generator():
        accumulated = ""
        async for chunk in call_simulator(
            inversion_statement, implication, full_context
        ):
            accumulated += chunk
            payload = json.dumps({"chunk": chunk})
            yield f"data: {payload}\n\n"

        # Send final assembled JSON
        yield f"data: {json.dumps({'full': accumulated})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{session_id}")
async def get_session(session_id: UUID):
    """Return the full session state for frontend hydration."""
    state = await _load_or_get_session(session_id)
    return state.model_dump(mode="json")
