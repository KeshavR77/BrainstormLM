"""
models.py — Pydantic domain models for the Recursive What-If Cascade
(WIC) Engine.

Covers the full node/edge/session graph that powers the recursive
brainstorming workflow.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────


class NodeType(str, Enum):
    ROOT = "root"  # User's original problem statement
    ASSUMPTION = "assumption"  # Extracted load-bearing assumption
    INVERSION = "inversion"  # One of 3 What-If branches
    SIMULATION = "simulation"  # Consequence map of chosen inversion
    SYNTHESIS = "synthesis"  # Final idea output (leaf node)


class InversionType(str, Enum):
    OPPOSITE = "opposite"  # Invert the assumption entirely
    EXTREMITY = "extremity"  # Push the assumption to its logical extreme
    ALIEN = "alien"  # Replace the substrate/medium entirely


# ── Value Objects ────────────────────────────────────────────────────────


class InversionBranch(BaseModel):
    inversion_id: UUID = Field(default_factory=uuid4)
    inversion_type: InversionType
    assumption_violated: str  # The exact assumption text being inverted
    inversion_statement: str  # "What if [inverted reality]?"
    one_line_implication: str  # What immediately changes in this world
    selected: bool = False  # Whether user clicked this branch


class SimulationOutput(BaseModel):
    breaks: list[str]  # Things that stop working (3-5 items)
    abundances: list[str]  # Things that become free/easy/cheap (3-5 items)
    new_constraints: list[str]  # New binding constraints that emerge (2-3 items)
    new_actors: list[str]  # New stakeholders this reality creates
    simulation_narrative: str  # 2-sentence prose summary of the new world


class SynthesizedIdea(BaseModel):
    title: str
    mechanism: str  # How it actually works (2-3 sentences)
    target_user: str
    core_value_prop: str
    assumption_chain: list[str]  # Lineage: assumption → inversion at each depth
    mvt: str  # Minimum viable test — cheapest experiment
    adjacent_analogues: list[str]  # Real things that work similarly in other domains


# ── Core Graph Objects ───────────────────────────────────────────────────


class Node(BaseModel):
    node_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    parent_id: Optional[UUID] = None  # None only for root
    depth: int = Field(ge=0, le=3)  # 0=root, 1=first simulation, max depth 3
    node_type: NodeType
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Content — populated depending on node_type
    raw_prompt: Optional[str] = None  # ROOT only
    assumption_text: Optional[str] = None  # ASSUMPTION only
    assumption_rationale: Optional[str] = None  # Why this is load-bearing
    inversions: Optional[list[InversionBranch]] = None  # INVERSION set
    selected_inversion: Optional[InversionBranch] = None
    simulation: Optional[SimulationOutput] = None  # SIMULATION only
    synthesized_idea: Optional[SynthesizedIdea] = None  # SYNTHESIS only

    # Metadata for evaluation
    novelty_score: Optional[float] = None
    user_rating: Optional[int] = None  # 1-5, collected post-synthesis


class Edge(BaseModel):
    edge_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    parent_node_id: UUID
    child_node_id: UUID
    edge_type: Literal[
        "extracts", "generates", "selects", "simulates", "synthesizes"
    ]
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Session State ────────────────────────────────────────────────────────


class SessionState(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    status: Literal[
        "awaiting_input",
        "extracting",
        "awaiting_selection",  # User must pick an inversion branch
        "simulating",
        "awaiting_recurse_or_synthesize",  # User decides: go deeper or harvest
        "synthesizing",
        "complete",
        "error",
    ]

    current_depth: int = 0
    max_depth: int = 3
    root_node_id: Optional[UUID] = None
    active_node_id: Optional[UUID] = None  # The node currently being processed
    path_taken: list[UUID] = []  # Ordered list of selected node_ids

    # Full tree stored as adjacency list for fast frontend serialization
    nodes: dict[str, Node] = {}  # node_id (str) → Node
    edges: list[Edge] = []

    # Telemetry
    total_llm_calls: int = 0
    total_latency_ms: int = 0
