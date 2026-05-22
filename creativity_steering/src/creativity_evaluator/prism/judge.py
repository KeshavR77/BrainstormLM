"""
LLM judges for CreativityPrism scoring.

Two implementations:
  - ClaudeJudge  — uses claude-sonnet-4-6 via Anthropic SDK
  - OpenAIJudge  — uses gpt-4o via OpenAI SDK

Each judge has two methods:
  score()        — evaluates a COLLECTION of texts (5 dimensions)
  score_single() — evaluates a SINGLE response vs its prompt (3 dimensions)
"""

from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..models import PrismScores, SinglePrismScores
from .rubrics import (
    PRISM_SYSTEM_PROMPT,
    PRISM_SINGLE_SYSTEM_PROMPT,
    CREATIVITY_PRISM_SYSTEM,
    CREATIVITY_PRISM_SYSTEM_IDEA,
    _ARM_FOOTER,
    _ARM_FOOTER_IDEA,
)


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def _format_collection_message(texts: list[str], context_prompt: str | None) -> str:
    lines = []
    if context_prompt:
        lines.append(f"TASK: {context_prompt}\n")
    lines.append(f"COLLECTION ({len(texts)} outputs):\n")
    for i, t in enumerate(texts, 1):
        lines.append(f"--- Output {i} ---\n{t.strip()}\n")
    lines.append("\nScore this collection using the rubric provided.")
    return "\n".join(lines)


def _format_single_message(text: str, prompt: str) -> str:
    return f"PROMPT: {prompt}\n\nRESPONSE:\n{text.strip()}\n\nScore this response using the rubric provided."


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _parse_collection_scores(raw: str) -> PrismScores:
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned non-JSON response: {raw!r}") from exc

    def _get(key: str) -> float:
        val = data.get(key)
        if val is None:
            raise ValueError(f"Missing key '{key}' in judge response: {data}")
        return float(val)

    return PrismScores(
        originality=_get("originality"),
        fluency=_get("fluency"),
        flexibility=_get("flexibility"),
        elaboration=_get("elaboration"),
        usefulness=_get("usefulness"),
        raw_response=raw,
    )


def _parse_single_scores(raw: str) -> SinglePrismScores:
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned non-JSON response: {raw!r}") from exc

    def _get(key: str) -> float:
        val = data.get(key)
        if val is None:
            raise ValueError(f"Missing key '{key}' in judge response: {data}")
        return float(val)

    return SinglePrismScores(
        originality=_get("originality"),
        elaboration=_get("elaboration"),
        usefulness=_get("usefulness"),
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Arm comparison helpers (ported from shrooms-llm creativity_prism.py)
# ---------------------------------------------------------------------------

def parse_winner_arm_line(verdict: str, valid: set[str]) -> str | None:
    """
    Extract the arm name from the last ``WINNER_ARM: <name>`` line in a verdict.
    Returns None if the line is missing or the name is not in ``valid``.
    """
    if not verdict or not valid:
        return None
    for line in reversed(verdict.strip().splitlines()):
        line = line.strip()
        m = re.match(r"^WINNER_ARM:\s*(.+?)\s*$", line, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name in valid:
                return name
            name = name.strip("`\"'")
            if name in valid:
                return name
    return None


def _format_arm_message(
    prompt: str,
    arm_completions: Sequence[tuple[str, str]],
    max_chars_per: int = 8000,
) -> str:
    parts = [
        "Original user prompt:",
        prompt.strip(),
        "",
        "You are comparing the following **named arms**. Each block is one model/configuration.",
        "",
    ]
    for arm, text in arm_completions:
        t = (text or "").strip()
        if max_chars_per > 0 and len(t) > max_chars_per:
            t = t[:max_chars_per] + "\n[...truncated for context limit...]"
        parts.extend([f"### ARM: {arm}", t, ""])
    if parts[-1] == "":
        parts.pop()
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Judge interface
# ---------------------------------------------------------------------------

class BaseJudge(ABC):
    @abstractmethod
    def score(self, texts: list[str], context_prompt: str | None = None) -> PrismScores:
        """Score a COLLECTION of texts on 5 CreativityPrism dimensions."""

    @abstractmethod
    def score_single(self, text: str, prompt: str) -> SinglePrismScores:
        """Score a SINGLE response on 3 dimensions: originality, elaboration, usefulness."""

    @abstractmethod
    def compare_arms(
        self,
        prompt: str,
        arm_completions: Sequence[tuple[str, str]],
        *,
        rubric: str = "default",
        temperature: float = 0.2,
        max_chars_per_completion: int = 8000,
    ) -> tuple[str, str | None]:
        """
        Compare named (arm_label, completion_text) pairs and pick a winner.
        Returns (verdict_text, winner_arm_label | None).

        rubric: "default" (novelty + quality) or "idea" (idea creativity + logic only).
        """


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

class ClaudeJudge(BaseJudge):
    """
    Judges using claude-sonnet-4-6 (Anthropic SDK).
    Requires ANTHROPIC_API_KEY in environment or passed explicitly.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 256,
    ):
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens

    def _call(self, system: str, user: str) -> str:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def score(self, texts: list[str], context_prompt: str | None = None) -> PrismScores:
        raw = self._call(PRISM_SYSTEM_PROMPT, _format_collection_message(texts, context_prompt))
        return _parse_collection_scores(raw)

    def score_single(self, text: str, prompt: str) -> SinglePrismScores:
        raw = self._call(PRISM_SINGLE_SYSTEM_PROMPT, _format_single_message(text, prompt))
        return _parse_single_scores(raw)

    def compare_arms(
        self,
        prompt: str,
        arm_completions: Sequence[tuple[str, str]],
        *,
        rubric: str = "default",
        temperature: float = 0.2,
        max_chars_per_completion: int = 8000,
    ) -> tuple[str, str | None]:
        names = [a[0] for a in arm_completions]
        arm_names_str = ", ".join(f"`{n}`" for n in names)
        footer = _ARM_FOOTER_IDEA if rubric == "idea" else _ARM_FOOTER
        system = (
            (CREATIVITY_PRISM_SYSTEM_IDEA if rubric == "idea" else CREATIVITY_PRISM_SYSTEM)
            + footer.format(arm_names=arm_names_str)
        )
        user = _format_arm_message(prompt, arm_completions, max_chars_per_completion)
        verdict = self._call(system, user)
        return verdict, parse_winner_arm_line(verdict, set(names))


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIJudge(BaseJudge):
    """
    Judges using gpt-4o (OpenAI SDK).
    Requires OPENAI_API_KEY in environment or passed explicitly.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 256,
    ):
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens

    def _call(self, system: str, user: str) -> str:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def score(self, texts: list[str], context_prompt: str | None = None) -> PrismScores:
        raw = self._call(PRISM_SYSTEM_PROMPT, _format_collection_message(texts, context_prompt))
        return _parse_collection_scores(raw)

    def score_single(self, text: str, prompt: str) -> SinglePrismScores:
        raw = self._call(PRISM_SINGLE_SYSTEM_PROMPT, _format_single_message(text, prompt))
        return _parse_single_scores(raw)

    def compare_arms(
        self,
        prompt: str,
        arm_completions: Sequence[tuple[str, str]],
        *,
        rubric: str = "default",
        temperature: float = 0.2,
        max_chars_per_completion: int = 8000,
    ) -> tuple[str, str | None]:
        names = [a[0] for a in arm_completions]
        arm_names_str = ", ".join(f"`{n}`" for n in names)
        footer = _ARM_FOOTER_IDEA if rubric == "idea" else _ARM_FOOTER
        system = (
            (CREATIVITY_PRISM_SYSTEM_IDEA if rubric == "idea" else CREATIVITY_PRISM_SYSTEM)
            + footer.format(arm_names=arm_names_str)
        )
        user = _format_arm_message(prompt, arm_completions, max_chars_per_completion)
        verdict = self._call(system, user)
        return verdict, parse_winner_arm_line(verdict, set(names))
