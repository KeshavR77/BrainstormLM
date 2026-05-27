"""Score steered generations with the creativity evaluator.

A "cell" (one steering configuration) is scored by generating several samples for each
brief and combining the single-response creativity (mean) with the collection diversity.
This is the number the sweep maximises.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from creativity_steering.models.backend import Injection, ModelBackend
from creativity_steering.scoring import CreativityScorer

# Default briefs for patching: neutral topics with no creative framing, so the injected
# vector — not the prompt — is what drives any change in creativity.
DEFAULT_BRIEFS: tuple[str, ...] = (
    "Describe a chair.",
    "Suggest a name for a cafe.",
    "Write about a walk in the city.",
    "Propose a gift for a teacher.",
    "Describe the weather today.",
)


@dataclass
class CellScore:
    composite: float
    mean_single: float
    collection: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "composite": self.composite,
            "mean_single": self.mean_single,
            "collection": self.collection,
            "n_samples": self.n_samples,
        }


def score_briefs(
    backend: ModelBackend,
    scorer: CreativityScorer,
    *,
    injections: Sequence[Injection],
    briefs: Sequence[str] = DEFAULT_BRIEFS,
    system: str | None = None,
    n_samples: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    seed: int = 0,
) -> CellScore:
    """Generate ``n_samples`` steered responses per brief and score creativity."""
    all_texts: list[str] = []
    single_scores: list[float] = []
    collection_scores: list[float] = []
    for b_idx, brief in enumerate(briefs):
        brief_texts: list[str] = []
        for s in range(n_samples):
            text = backend.generate_steered(
                brief,
                injections=injections,
                system=system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed * 10_000 + b_idx * 100 + s,
            )
            brief_texts.append(text)
            single_scores.append(scorer.score_single(text, brief))
        all_texts.extend(brief_texts)
        if len(brief_texts) >= 2:
            collection_scores.append(scorer.score_collection(brief_texts, brief))

    mean_single = sum(single_scores) / len(single_scores) if single_scores else 0.0
    collection = sum(collection_scores) / len(collection_scores) if collection_scores else 0.0
    composite = 0.7 * mean_single + 0.3 * collection
    return CellScore(
        composite=composite,
        mean_single=mean_single,
        collection=collection,
        n_samples=len(all_texts),
    )
