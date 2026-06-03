"""The honest metric suite — single import surface for the other worktrees.

    from brainstormlm.eval import metric_suite_report

Assembles the diversity metrics that matter for selection, with a clear split
between *targets* and *canaries* (the Consistency-Audit lesson: never optimize
semantically-blind metrics):

  - novelty_bench    — NoveltyBench-Distinct, the north-star (always).
  - salad_rate       — degenerate-output gate (always).
  - conditional_vendi — within-prompt diversity (only when ``per_prompt`` given).
  - novelty_vs_corpus — anti-homogenization vs prior outputs (only when ``corpus`` given).
  - canaries         — Vendi / Distinct-2 / Self-BLEU: diagnostics ONLY, never targets.

MAUVE is intentionally excluded from the default report — it needs a reference
set and is heavy; pull it in explicitly where a reference distribution exists.
"""

from __future__ import annotations

from typing import Sequence

from brainstormlm.eval.conditional_vendi import conditional_vendi
from brainstormlm.eval.diversity import distinct_bigrams, vendi_score
from brainstormlm.eval.mode_collapse import DEFAULT_CLUSTER_THRESHOLD
from brainstormlm.eval.novelty_bench import novelty_bench_distinct
from brainstormlm.eval.novelty_corpus import novelty_vs_corpus
from brainstormlm.eval.salad import salad_rate
from brainstormlm.eval.self_bleu import self_bleu

_CANARY_LABEL = "semantically blind; diagnostics only, never optimization targets"


def metric_suite_report(
    texts: Sequence[str],
    *,
    corpus: Sequence[str] | None = None,
    per_prompt: Sequence[Sequence[str]] | None = None,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict:
    """Run the diversity metric suite over ``texts``.

    Always includes the north-star (``novelty_bench``) and the ``salad_rate``
    gate. ``conditional_vendi`` is added when per-prompt response groups are
    supplied; ``novelty_vs_corpus`` when a reference ``corpus`` is supplied.
    Canaries are reported but flagged as never-optimize.
    """
    report: dict = {
        "novelty_bench": novelty_bench_distinct(texts, threshold=threshold),
        "salad_rate": salad_rate(texts),
        "canaries": {
            "label": _CANARY_LABEL,
            "vendi_score": vendi_score(texts),
            "distinct2": distinct_bigrams(texts),
            "self_bleu": self_bleu(texts),
        },
    }
    if per_prompt is not None:
        report["conditional_vendi"] = conditional_vendi(per_prompt)
    if corpus is not None:
        report["novelty_vs_corpus"] = novelty_vs_corpus(texts, corpus, threshold=threshold)
    return report
