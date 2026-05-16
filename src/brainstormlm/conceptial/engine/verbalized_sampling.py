"""Verbalized Sampling helpers — structure generation so the model assigns probabilities over modes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class VerbalizedParse:
    """Parsed VS completion."""

    candidates: list[tuple[str, float]]
    answer_text: str
    raw_completion: str


def verbalized_instruction_suffix(k: int) -> str:
    """Append to the user task before chat templating."""
    return (
        "\n\n### Verbalized sampling (required)\n"
        f"First think of exactly {int(k)} distinct candidate responses to the task above.\n"
        "Assign each candidate a positive probability such that probabilities sum to 1.0.\n"
        'Output a JSON object only (no markdown fences) with this shape:\n'
        '{"candidates":[{"text":"...","p":0.2},...],"chosen_index": <0-based index you will elaborate>}\n'
        "Then on new lines after the JSON, write ANSWER_START followed by the full text for "
        "chosen_index only (complete answer to the original task).\n"
    )


_JSON_OBJ = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_verbalized_completion(raw: str, *, expected_k: int) -> VerbalizedParse:
    """Extract candidates / probabilities / chosen elaboration; robust to minor formatting issues."""
    text = (raw or "").strip()
    candidates: list[tuple[str, float]] = []
    blob: str | None = None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.I)
    if fenced:
        blob = fenced.group(1).strip()
    if blob is None:
        m = _JSON_OBJ.search(text)
        if m:
            blob = m.group(0)
    answer_text = text
    if blob:
        try:
            payload = json.loads(blob)
            clist = payload.get("candidates") or []
            for c in clist:
                if isinstance(c, dict) and "text" in c:
                    p = float(c.get("p", 1.0 / max(1, len(clist))))
                    candidates.append((str(c["text"]).strip(), max(1e-9, p)))
        except (json.JSONDecodeError, TypeError, ValueError):
            candidates = []
    if candidates:
        s = sum(p for _, p in candidates)
        if s > 0:
            candidates = [(t, p / s) for t, p in candidates]
    if "ANSWER_START" in text:
        answer_text = text.split("ANSWER_START", 1)[-1].strip()
    if not candidates:
        candidates = [(text, 1.0)]
    return VerbalizedParse(candidates=candidates, answer_text=answer_text.strip() or text, raw_completion=text)


def pick_tail_candidate(
    parsed: VerbalizedParse,
    *,
    sample_index: int,
    k_hint: int,
) -> str:
    """Prefer lower-probability modes for diversity across repeated samples."""
    cands = parsed.candidates
    if not cands:
        return parsed.answer_text
    ranked = sorted(cands, key=lambda x: x[1])
    idx = int(sample_index) % len(ranked)
    tail_first = ranked[idx][0].strip()
    if tail_first and len(tail_first) >= 3:
        return tail_first
    return parsed.answer_text
