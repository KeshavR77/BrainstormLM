"""EQ-Bench Creative Writing v3 harness (subset, GPT-4o-mini judge).

This is a faithful **subset** of the EQ-Bench Creative Writing v3 benchmark
(https://eqbench.com/creative_writing.html, MIT). We bundle the first 10
prompts (out of 32) verbatim from
``EQ-bench/creative-writing-bench/data/creative_writing_prompts_v3.json``
at ``config/eq_bench_creative_v3_subset.json``.

Differences vs the official harness:
  - We score with **gpt-4o-mini**, not Claude Sonnet 4 (the recommended
    judge). This means absolute scores are not directly comparable to the
    public leaderboard, but **relative** ordering between two models judged
    with the same protocol is well-defined.
  - Subset of 10 prompts vs the full 32.
  - We use one seed modifier per prompt (the first one) for determinism.

The 11 positive + 9 negative rubric dimensions are taken verbatim from
``data/creative_writing_criteria.txt`` in the EQ-Bench repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

CRITERIA_POSITIVE = [
    "Adherence to Instructions",
    "Believable Character Actions",
    "Nuanced Characters",
    "Consistent Voice/Tone of Writing",
    "Imagery and Descriptive Quality",
    "Elegant Prose",
    "Emotionally Engaging",
    "Emotionally Complex",
    "Coherent",
    "Sentences Flow Naturally",
    "Overall Reader Engagement",
]

CRITERIA_NEGATIVE = [
    "Meandering",
    "Weak Dialogue",
    "Tell-Don't-Show",
    "Unsurprising or Uncreative",
    "Amateurish",
    "Purple Prose",
    "Overwrought",
    "Incongruent Ending Positivity",
    "Unearned Transformations",
]


def load_prompts(path: Path | None = None) -> list[dict]:
    """Load the 10-prompt EQ-Bench v3 subset, each rendered with its first seed.

    Returns a list of ``{"id": str, "category": str, "title": str, "prompt": str}``
    where ``prompt`` is the writing_prompt with ``<SEED>`` substituted by the
    first seed_modifier.
    """
    if path is None:
        root = Path(__file__).resolve().parents[3]
        path = root / "config" / "eq_bench_creative_v3_subset.json"
    raw = json.loads(path.read_text())
    out: list[dict] = []
    for key, entry in raw.items():
        seed = (entry.get("seed_modifiers") or [""])[0]
        prompt = entry["writing_prompt"].replace("<SEED>", seed)
        out.append(
            {
                "id": str(key),
                "category": entry["category"],
                "title": entry["title"],
                "prompt": prompt,
            }
        )
    return out


def _judge_prompt(writing_prompt: str, response: str) -> str:
    pos_list = "\n".join(f"  - {c}" for c in CRITERIA_POSITIVE)
    neg_list = "\n".join(f"  - {c}" for c in CRITERIA_NEGATIVE)
    return (
        "You are a literary judge evaluating a creative writing response against "
        "the EQ-Bench Creative Writing v3 rubric.\n\n"
        f"WRITING PROMPT:\n{writing_prompt}\n\n"
        f"RESPONSE TO SCORE:\n{response[:8000]}\n\n"
        "Score each criterion 0-10 (10 = strongest exemplar of this property "
        "in published fiction; 0 = absent). Be brutally honest — most LLM "
        "fiction sits in the 3-6 range. Reserve 8+ for genuinely strong "
        "writing. Return ONLY a JSON object with these exact keys:\n\n"
        "Positive criteria (higher = better):\n"
        f"{pos_list}\n\n"
        "Negative criteria (higher = MORE of this bad trait):\n"
        f"{neg_list}\n\n"
        "Use snake_case keys (e.g. 'adherence_to_instructions', "
        "'tell_dont_show'). Add a final string key 'one_line_rationale'. "
        "No markdown fences."
    )


def _to_snake(name: str) -> str:
    return (
        name.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace("'", "")
        .replace(" ", "_")
    )


def parse_judge_payload(payload: dict) -> dict:
    """Compute composite from raw judge payload.

    Each positive contributes ``score / 10``. Each negative contributes
    ``(10 - score) / 10`` (flipped). Composite is the mean across all 20
    contributions, mapped back to a 0-10 scale.
    """
    contributions: list[float] = []
    by_criterion: dict[str, float] = {}
    for c in CRITERIA_POSITIVE:
        key = _to_snake(c)
        val = float(payload.get(key, 0))
        val = max(0.0, min(10.0, val))
        by_criterion[key] = val
        contributions.append(val / 10.0)
    for c in CRITERIA_NEGATIVE:
        key = _to_snake(c)
        val = float(payload.get(key, 0))
        val = max(0.0, min(10.0, val))
        by_criterion[key] = val
        contributions.append((10.0 - val) / 10.0)
    composite = 10.0 * (sum(contributions) / len(contributions))
    return {
        "criteria": by_criterion,
        "composite": composite,
        "rationale": str(payload.get("one_line_rationale", ""))[:300],
    }


def score_response(
    writing_prompt: str,
    response: str,
    *,
    judge: Callable[[str], dict],
) -> dict:
    """Run the judge on a single response and parse the result.

    ``judge`` must take a single user-message string and return a parsed
    JSON dict. Decoupled so tests can stub it without an API call.
    """
    payload = judge(_judge_prompt(writing_prompt, response))
    return parse_judge_payload(payload)


def aggregate(scored: Sequence[dict]) -> dict:
    """Mean composite + per-criterion means across N scored responses."""
    valid = [s for s in scored if "composite" in s]
    if not valid:
        return {"mean_composite": 0.0, "n": 0, "per_criterion_mean": {}}
    mean_comp = sum(s["composite"] for s in valid) / len(valid)
    all_criteria = sorted({k for s in valid for k in s["criteria"].keys()})
    per_crit = {
        k: sum(s["criteria"].get(k, 0.0) for s in valid) / len(valid) for k in all_criteria
    }
    return {
        "mean_composite": mean_comp,
        "n": len(valid),
        "per_criterion_mean": per_crit,
    }
