"""Conceptial / Corpius 1 demo surface — the LOCKED model configuration.

This package is the canonical home for the production-grade Conceptial inference path
on local 3B. It uses ``shrooms_llm`` (still editable-installed via ``[tool.uv.sources]``)
as the underlying model + steering library, but every demo-relevant convention lives here:

- :mod:`brainstormlm.conceptial.idea_store` — SQLite-backed persistent project memory
- :mod:`brainstormlm.conceptial.anti_mode` — cognitive anti-mode prompt formatter +
  brief-aware output instructions + few-shot exemplar plumbing

The runnable smoke / scoring scripts live in ``BrainstormLM/scripts/`` and import from
this package.

**Locked configuration (Day-2 final, post-autonomous-innovation run):**

```
model:        Qwen/Qwen2.5-3B-Instruct
steering:     v17 purified L14 (α=2.62 → 0 by step 168) + L18 (α=0.86, sigmoid mid=52)
sampling:     min_p=0.01 · base_temp=0.8 · per-sample temp jitter 0.05
diversity:    cognitive anti-mode at PROMPT level
              + cross-session persistence via IdeaMemoryStore
              + auto-extracted few-shot exemplars (top-K by composite score, deduped stylistically)
output mode:  image-prompt | strategy | generic (brief-aware)
```
"""

from brainstormlm.conceptial.anti_mode import (
    CAPTURE_LAYER,
    DEFAULT_PROMPT,
    OUTPUT_MODE_INSTRUCTIONS,
    exemplar_preamble,
    extract_summary,
    format_anti_mode,
    format_pristine,
)
from brainstormlm.conceptial.idea_store import (
    IdeaMemoryStore,
    IdeaRow,
    SOURCE_GENERATED,
    SOURCE_SEEDED,
)

__all__ = [
    # idea_store
    "IdeaMemoryStore",
    "IdeaRow",
    "SOURCE_SEEDED",
    "SOURCE_GENERATED",
    # anti_mode
    "CAPTURE_LAYER",
    "DEFAULT_PROMPT",
    "OUTPUT_MODE_INSTRUCTIONS",
    "exemplar_preamble",
    "format_pristine",
    "format_anti_mode",
    "extract_summary",
]
