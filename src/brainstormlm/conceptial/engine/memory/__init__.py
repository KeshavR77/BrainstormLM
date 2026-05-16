"""Per-project native-latent idea memory for Conceptial / Corpius 1 (Day 3 / V8.2).

Sibling to ``shrooms_llm.ior`` (which holds in-process FIFO state) — this package
persists last-token hidden states to disk so a server restart, a second client, or
a multi-day project keeps its diversity memory.
"""

from brainstormlm.conceptial.engine.memory.idea_store import IdeaMemoryStore

__all__ = ["IdeaMemoryStore"]
