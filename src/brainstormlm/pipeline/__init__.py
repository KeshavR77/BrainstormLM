"""Selection-stage pipeline: semantic anti-mode / repulsion selector.

Public surface the other worktrees import:

    from brainstormlm.pipeline import select_by_repulsion, SelectionResult
"""

from brainstormlm.pipeline.select import SelectionResult, select_by_repulsion

__all__ = ["SelectionResult", "select_by_repulsion"]
