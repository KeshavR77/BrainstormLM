"""Selection-stage pipeline: semantic anti-mode / repulsion selector.

Public surface the other worktrees import:

    from brainstormlm.pipeline import select_diverse, SelectionResult
"""

from brainstormlm.pipeline.select import SelectionResult, select_diverse

__all__ = ["SelectionResult", "select_diverse"]
