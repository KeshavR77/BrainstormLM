"""MAUVE divergence between model outputs and a human reference set.

MAUVE (Pillutla et al., NeurIPS 2021) measures how close a model's generation
distribution is to a human-written distribution by computing a Type-1 / Type-2
mixture KL between embedding clusters. Range [0, 1]: higher = closer to human.

We use the official ``mauve-text`` PyPI package backed by sentence-transformers
embeddings (so we don't depend on a specific featurizer model). The reference
set is human-written examples in the same domain as the model outputs (e.g.
brand creative briefs).
"""

from __future__ import annotations

from typing import Sequence


def mauve_score(
    model_texts: Sequence[str],
    reference_texts: Sequence[str],
    *,
    featurize_model_name: str = "gpt2-large",
    max_text_length: int = 256,
    device_id: int = -1,
    verbose: bool = False,
) -> float:
    """Compute MAUVE divergence between model and reference text sets.

    Args:
      model_texts: outputs from the model under evaluation.
      reference_texts: human-written exemplars from the same domain.
      featurize_model_name: HF model id used to embed both sets (default
        gpt2-large — matches the paper's default).
      max_text_length: truncate texts to this many tokens before embedding.
      device_id: -1 for CPU, otherwise a CUDA device index.
      verbose: pass through to the mauve library's progress logging.

    Returns:
      A float in [0, 1]. Higher = model distribution is closer to references.
    """
    if len(model_texts) < 2 or len(reference_texts) < 2:
        return 0.0

    import mauve

    out = mauve.compute_mauve(
        p_text=list(model_texts),
        q_text=list(reference_texts),
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=verbose,
        featurize_model_name=featurize_model_name,
    )
    return float(out.mauve)
