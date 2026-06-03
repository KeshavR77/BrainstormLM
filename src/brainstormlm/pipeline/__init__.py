"""Selection-stage pipeline.

Two layers:
  - ``select_by_repulsion`` / ``SelectionResult`` — the pure embedding-repulsion
    primitive (schema-free); what the generation/steering worktrees import.
  - ``select_response_ideas`` / ``SelectedIdeas`` — the API-shaped step-④ adapter
    returning ``IdeaV1`` + ``DiversityScoreV1`` (drop-in for ``api/public.py``).
"""

from brainstormlm.pipeline.select import SelectionResult, select_by_repulsion
from brainstormlm.pipeline.select_api import SelectedIdeas, select_response_ideas

__all__ = [
    "SelectedIdeas",
    "SelectionResult",
    "select_by_repulsion",
    "select_response_ideas",
]
