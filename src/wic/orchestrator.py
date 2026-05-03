"""
orchestrator.py — The WIC state machine.

Exposes ``run_wic_step`` which advances a ``SessionState`` by one
logical step depending on its current ``status``.

State transitions:

  awaiting_input  ──→  extracting
  extracting      ──→  awaiting_selection  (3 inversions generated in parallel)
  awaiting_selection ──→  simulating
  simulating      ──→  awaiting_recurse_or_synthesize
  awaiting_recurse_or_synthesize ──→ extracting  (recurse)
                                   ──→ synthesizing (harvest)
  synthesizing    ──→  complete
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from brainstormlm.wic.database import get_pool
from brainstormlm.wic.llm_service import (
    call_extractor,
    call_inversion_generator,
    call_simulator_safe,
    call_synthesizer,
)
from brainstormlm.wic.models import (
    Edge,
    InversionBranch,
    InversionType,
    Node,
    NodeType,
    SessionState,
    SimulationOutput,
    SynthesizedIdea,
)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS — context builders
# ═══════════════════════════════════════════════════════════════════════


def _build_problem_context(state: SessionState) -> str:
    """
    Build the cumulative problem context string by walking the
    ``path_taken`` through the node tree.

    At depth 0 this is just the raw prompt.  At deeper depths it
    concatenates the simulation narratives of every world visited.
    """
    parts: list[str] = []
    for nid in state.path_taken:
        node = state.nodes.get(str(nid))
        if node is None:
            continue
        if node.node_type == NodeType.ROOT and node.raw_prompt:
            parts.append(f"Original problem: {node.raw_prompt}")
        if node.simulation:
            parts.append(
                f"[Depth {node.depth}] Simulated world: "
                f"{node.simulation.simulation_narrative}"
            )
        if node.selected_inversion:
            parts.append(
                f"  ↳ Inversion chosen: {node.selected_inversion.inversion_statement}"
            )
    return "\n".join(parts) if parts else ""


def _collect_prior_inversions(state: SessionState) -> list[str]:
    """Gather all inversion statements that have been explored so far."""
    inversions: list[str] = []
    for node in state.nodes.values():
        if node.inversions:
            for branch in node.inversions:
                inversions.append(branch.inversion_statement)
    return inversions


def _build_full_cascade_context(state: SessionState) -> str:
    """
    Build a rich narrative trace of the full cascade for the synthesizer.
    Walks the ``path_taken`` in order.
    """
    lines: list[str] = []
    for nid in state.path_taken:
        node = state.nodes.get(str(nid))
        if node is None:
            continue
        if node.node_type == NodeType.ROOT:
            lines.append(f"## Root problem (depth 0)\n{node.raw_prompt}\n")
        if node.assumption_text:
            lines.append(
                f"### Assumption extracted (depth {node.depth})\n"
                f"{node.assumption_text}\n"
                f"Rationale: {node.assumption_rationale}\n"
            )
        if node.selected_inversion:
            inv = node.selected_inversion
            lines.append(
                f"### Inversion chosen ({inv.inversion_type.value})\n"
                f"{inv.inversion_statement}\n"
                f"Implication: {inv.one_line_implication}\n"
            )
        if node.simulation:
            sim = node.simulation
            lines.append(
                f"### Simulation (depth {node.depth})\n"
                f"Narrative: {sim.simulation_narrative}\n"
                f"Breaks: {', '.join(sim.breaks)}\n"
                f"Abundances: {', '.join(sim.abundances)}\n"
                f"New constraints: {', '.join(sim.new_constraints)}\n"
                f"New actors: {', '.join(sim.new_actors)}\n"
            )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# DB PERSISTENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════


async def _persist_session(state: SessionState) -> None:
    """Upsert the session row in PostgreSQL."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (session_id, user_id, status, current_depth, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (session_id) DO UPDATE
                SET status        = EXCLUDED.status,
                    current_depth = EXCLUDED.current_depth,
                    updated_at    = NOW()
            """,
            state.session_id,
            state.user_id,
            state.status,
            state.current_depth,
            state.created_at,
            datetime.utcnow(),
        )


async def _persist_node(node: Node) -> None:
    """Insert a node row (payload = full Pydantic JSON)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO nodes (node_id, session_id, parent_id, depth, node_type, payload, created_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            ON CONFLICT (node_id) DO UPDATE
                SET payload = EXCLUDED.payload
            """,
            node.node_id,
            node.session_id,
            node.parent_id,
            node.depth,
            node.node_type.value,
            node.model_dump_json(),
            node.created_at,
        )


async def _persist_edge(edge: Edge) -> None:
    """Insert an edge row."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO edges (edge_id, session_id, parent_node_id, child_node_id, edge_type, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (edge_id) DO NOTHING
            """,
            edge.edge_id,
            edge.session_id,
            edge.parent_node_id,
            edge.child_node_id,
            edge.edge_type,
            edge.created_at,
        )


# ═══════════════════════════════════════════════════════════════════════
# CORE STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════


async def run_wic_step(
    state: SessionState,
    *,
    user_input: str | None = None,
    selected_inversion_id: UUID | None = None,
    action: str | None = None,  # "recurse" | "synthesize"
) -> SessionState:
    """
    Advance the session by one logical step.

    ``user_input``            — required when status == "awaiting_input"
    ``selected_inversion_id`` — required when status == "awaiting_selection"
    ``action``                — required when status == "awaiting_recurse_or_synthesize"

    Returns the mutated ``SessionState`` (also persisted to DB).
    """
    t0 = time.monotonic()

    # ── awaiting_input ───────────────────────────────────────────────
    if state.status == "awaiting_input":
        if not user_input:
            raise ValueError("user_input is required when status is awaiting_input")

        root_node = Node(
            session_id=state.session_id,
            depth=0,
            node_type=NodeType.ROOT,
            raw_prompt=user_input,
        )
        state.nodes[str(root_node.node_id)] = root_node
        state.root_node_id = root_node.node_id
        state.active_node_id = root_node.node_id
        state.path_taken.append(root_node.node_id)
        state.status = "extracting"

        await _persist_node(root_node)
        await _persist_session(state)
        # Fall through to extracting immediately
        return await run_wic_step(state)

    # ── extracting ───────────────────────────────────────────────────
    if state.status == "extracting":
        context = _build_problem_context(state)
        prior = _collect_prior_inversions(state)

        # 1) Extract the assumption
        extraction = await call_extractor(context, prior, state.current_depth)
        state.total_llm_calls += 1

        assumption_text = extraction["assumption"]
        assumption_rationale = extraction.get("rationale", "")

        # Create assumption node
        parent_id = state.active_node_id
        assumption_node = Node(
            session_id=state.session_id,
            parent_id=parent_id,
            depth=state.current_depth,
            node_type=NodeType.ASSUMPTION,
            assumption_text=assumption_text,
            assumption_rationale=assumption_rationale,
        )
        state.nodes[str(assumption_node.node_id)] = assumption_node
        state.path_taken.append(assumption_node.node_id)

        extract_edge = Edge(
            session_id=state.session_id,
            parent_node_id=parent_id,
            child_node_id=assumption_node.node_id,
            edge_type="extracts",
        )
        state.edges.append(extract_edge)

        await _persist_node(assumption_node)
        await _persist_edge(extract_edge)

        # 2) Generate 3 inversions IN PARALLEL using asyncio.gather
        inversion_types = [
            InversionType.OPPOSITE,
            InversionType.EXTREMITY,
            InversionType.ALIEN,
        ]

        results: list[dict[str, Any]] = await asyncio.gather(
            call_inversion_generator(assumption_text, context, "opposite"),
            call_inversion_generator(assumption_text, context, "extremity"),
            call_inversion_generator(assumption_text, context, "alien"),
        )
        state.total_llm_calls += 3

        # Build inversion branches
        branches: list[InversionBranch] = []
        for inv_type, result in zip(inversion_types, results):
            branch = InversionBranch(
                inversion_type=inv_type,
                assumption_violated=assumption_text,
                inversion_statement=result["inversion_statement"],
                one_line_implication=result["one_line_implication"],
            )
            branches.append(branch)

        # Create inversion node that holds all 3 branches
        inversion_node = Node(
            session_id=state.session_id,
            parent_id=assumption_node.node_id,
            depth=state.current_depth,
            node_type=NodeType.INVERSION,
            inversions=branches,
        )
        state.nodes[str(inversion_node.node_id)] = inversion_node
        state.active_node_id = inversion_node.node_id

        gen_edge = Edge(
            session_id=state.session_id,
            parent_node_id=assumption_node.node_id,
            child_node_id=inversion_node.node_id,
            edge_type="generates",
        )
        state.edges.append(gen_edge)

        await _persist_node(inversion_node)
        await _persist_edge(gen_edge)

        state.status = "awaiting_selection"
        _record_latency(state, t0)
        await _persist_session(state)
        return state

    # ── awaiting_selection ───────────────────────────────────────────
    if state.status == "awaiting_selection":
        if not selected_inversion_id:
            raise ValueError(
                "selected_inversion_id is required when status is awaiting_selection"
            )

        # Find the inversion node
        inv_node = state.nodes.get(str(state.active_node_id))
        if not inv_node or not inv_node.inversions:
            raise ValueError("Active node has no inversions to select from")

        # Mark the selected branch
        selected: InversionBranch | None = None
        for branch in inv_node.inversions:
            if branch.inversion_id == selected_inversion_id:
                branch.selected = True
                selected = branch
                break

        if not selected:
            raise ValueError(
                f"Inversion ID {selected_inversion_id} not found in active node"
            )

        inv_node.selected_inversion = selected
        state.nodes[str(inv_node.node_id)] = inv_node
        state.path_taken.append(inv_node.node_id)

        select_edge = Edge(
            session_id=state.session_id,
            parent_node_id=inv_node.node_id,
            child_node_id=inv_node.node_id,  # self-edge: selection event
            edge_type="selects",
        )
        state.edges.append(select_edge)
        await _persist_edge(select_edge)

        state.status = "simulating"
        await _persist_node(inv_node)
        await _persist_session(state)
        # Fall through to simulating
        return await run_wic_step(state)

    # ── simulating ───────────────────────────────────────────────────
    if state.status == "simulating":
        inv_node = state.nodes.get(str(state.active_node_id))
        if not inv_node or not inv_node.selected_inversion:
            raise ValueError("No selected inversion to simulate")

        selected = inv_node.selected_inversion
        context = _build_problem_context(state)

        sim_result = await call_simulator_safe(
            selected.inversion_statement,
            selected.one_line_implication,
            context,
        )
        state.total_llm_calls += 1

        simulation = SimulationOutput(**sim_result)

        sim_node = Node(
            session_id=state.session_id,
            parent_id=inv_node.node_id,
            depth=state.current_depth,
            node_type=NodeType.SIMULATION,
            simulation=simulation,
        )
        state.nodes[str(sim_node.node_id)] = sim_node
        state.active_node_id = sim_node.node_id
        state.path_taken.append(sim_node.node_id)

        sim_edge = Edge(
            session_id=state.session_id,
            parent_node_id=inv_node.node_id,
            child_node_id=sim_node.node_id,
            edge_type="simulates",
        )
        state.edges.append(sim_edge)

        await _persist_node(sim_node)
        await _persist_edge(sim_edge)

        # Auto-synthesize if at max depth
        if state.current_depth >= state.max_depth:
            state.status = "synthesizing"
        else:
            state.status = "awaiting_recurse_or_synthesize"

        _record_latency(state, t0)
        await _persist_session(state)
        return state

    # ── awaiting_recurse_or_synthesize ───────────────────────────────
    if state.status == "awaiting_recurse_or_synthesize":
        if action not in ("recurse", "synthesize"):
            raise ValueError("action must be 'recurse' or 'synthesize'")

        if action == "recurse":
            state.current_depth += 1
            state.status = "extracting"
            await _persist_session(state)
            return await run_wic_step(state)
        else:
            state.status = "synthesizing"
            await _persist_session(state)
            return await run_wic_step(state)

    # ── synthesizing ─────────────────────────────────────────────────
    if state.status == "synthesizing":
        cascade_context = _build_full_cascade_context(state)

        synth_result = await call_synthesizer(cascade_context)
        state.total_llm_calls += 1

        idea = SynthesizedIdea(**synth_result)

        synth_node = Node(
            session_id=state.session_id,
            parent_id=state.active_node_id,
            depth=state.current_depth,
            node_type=NodeType.SYNTHESIS,
            synthesized_idea=idea,
        )
        state.nodes[str(synth_node.node_id)] = synth_node
        state.active_node_id = synth_node.node_id
        state.path_taken.append(synth_node.node_id)

        synth_edge = Edge(
            session_id=state.session_id,
            parent_node_id=state.path_taken[-2] if len(state.path_taken) >= 2 else synth_node.node_id,
            child_node_id=synth_node.node_id,
            edge_type="synthesizes",
        )
        state.edges.append(synth_edge)

        await _persist_node(synth_node)
        await _persist_edge(synth_edge)

        state.status = "complete"
        _record_latency(state, t0)
        await _persist_session(state)
        return state

    # ── fallthrough ──────────────────────────────────────────────────
    raise ValueError(f"Cannot advance session from status: {state.status}")


def _record_latency(state: SessionState, t0: float) -> None:
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    state.total_latency_ms += elapsed_ms
