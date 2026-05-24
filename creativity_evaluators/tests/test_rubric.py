"""Guards that the bias-mitigation rules from the research synthesis are actually present
in the rubric text (the checklist demands they be visible in code)."""

from __future__ import annotations

from creativity_evaluators.stage2.rubric import (
    RESPONSE_JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_user_message,
)
from creativity_evaluators.types import DIMENSIONS


def test_rubric_mentions_all_five_dimensions():
    for dim in DIMENSIONS:
        assert dim in SYSTEM_PROMPT


def test_rubric_uses_coarse_1_to_5_scale():
    assert "1-5" in SYSTEM_PROMPT


def test_rubric_forbids_rewarding_length():
    assert "do not reward longer" in SYSTEM_PROMPT.lower()


def test_rubric_forbids_rewarding_flowery_vocab():
    assert "flowery" in SYSTEM_PROMPT.lower()


def test_schema_requires_all_dimensions_and_reasoning():
    assert set(DIMENSIONS).issubset(set(RESPONSE_JSON_SCHEMA["required"]))
    assert "reasoning" in RESPONSE_JSON_SCHEMA["required"]


def test_user_message_contains_prompt_and_response():
    msg = build_user_message("MY_PROMPT", "MY_RESPONSE")
    assert "MY_PROMPT" in msg and "MY_RESPONSE" in msg
