"""Cloud evaluation utilities: aggressive boilerplate stripping, taxonomy judge, PRISM floor."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from creativity_evaluator.prism.judge import OpenAIJudge

_STRIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?is)^(?:\s*(?:Certainly|Sure|Sure thing|Of course|Absolutely|Great|I'd be happy to|"
        r"Happy to help|You got it|No problem|Alright|Okay|OK|Yes|Here you go|Let's dive in)[^\n]{0,200}?\n)+"
    ),
    re.compile(
        r"(?is)^\s*(?:Here(?:'s| are| is)|Below (?:are|is)|Below (?:you'll|you will) find)[^\n]+\n"
    ),
    re.compile(r"(?is)^\s*(?:I've compiled|I have compiled|I've prepared|Let me (?:share|provide))[^\n]+\n"),
    # Enumerated intros ("Here are 12 innovative features:")
    re.compile(
        r"(?is)^\s*(?:Here (?:is|are)|Below (?:is|are))\s+\d+\s+[^\n]+\n"
    ),
    re.compile(
        r"(?is)^\s*(?:I'll|I will|Allow me to)\s+(?:provide|give|share|present|offer)[^\n]+\n"
    ),
]


def strip_boilerplate(text: str) -> str:
    """Strip conversational filler before structural / embedding metrics."""
    t = (text or "").strip()
    if not t:
        return t
    prev = None
    safety = 0
    while prev != t and safety < 8:
        prev = t
        safety += 1
        for pat in _STRIP_PATTERNS:
            t = pat.sub("", t, count=1)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t.strip()


def taxonomy_judge_system_prompt(num_categories: int) -> str:
    n = int(num_categories)
    return f"""You are mapping highly specific, novel AI concepts to broad thematic buckets.

You MUST map this sample to exactly ONE category index from 1 through {n} (matching the numbered list order).

Rules:
- Prefer Semantic Affinity: match the underlying mechanism, intent, or thematic role—not surface wording.
- Force a best-fit mapping even when the sample uses unusual or jargon-heavy language.
- ONLY return category index 0 if the text is complete gibberish: unreadable, incoherent, or not natural language.

Respond with JSON only, no markdown fences:
{{"category": <int 0-{n}>}}"""


def taxonomy_map_category(
    sample_text: str,
    categories: list[str],
    *,
    client: Any,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_chars: int = 12000,
) -> int:
    """
    Phase 3-style taxonomy assignment (1..len(categories)), or 0 only for gibberish.

    ``client`` must implement ``chat.completions.create`` like ``openai.OpenAI``.
    """
    cats = [str(c).strip() for c in categories if str(c).strip()]
    if not cats:
        raise ValueError("categories must be non-empty")
    n = len(cats)
    body = strip_boilerplate(sample_text)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[truncated]"
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cats))
    user = f"CATEGORIES (N={n}):\n{numbered}\n\nSAMPLE:\n{body}\n"
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": taxonomy_judge_system_prompt(n)},
            {"role": "user", "content": user},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:].lstrip()
    data = json.loads(raw)
    c = int(data.get("category", -1))
    if c < 0 or c > n:
        raise ValueError(f"invalid category {c!r} (expected 0..{n})")
    return c


def mean_prism_triplet_score(
    prompt: str,
    texts: list[str],
    judge: "OpenAIJudge | None" = None,
) -> float:
    """
    Mean CreativityPrism single-response score: average of (originality+elaboration+usefulness)/3
    on 1–5 scale, one call per text. Used as the quality floor (e.g. >= 3.0).
    """
    from creativity_evaluator.prism.judge import OpenAIJudge

    j = judge or OpenAIJudge()
    if not texts:
        return 0.0
    total = 0.0
    for t in texts:
        s = j.score_single(strip_boilerplate(t), prompt)
        total += (float(s.originality) + float(s.elaboration) + float(s.usefulness)) / 3.0
    return total / float(len(texts))
