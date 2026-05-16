"""Shared OpenAI chat baselines (control + prompt-engineered) and prompt helpers for benchmarks / UI."""

from __future__ import annotations

import os
import time
from typing import Any

SYSTEM_STANDARD = "You are a helpful assistant."


def _usage_dict(usage: Any) -> dict[str, int]:
    """Normalize OpenAI ``response.usage`` into JSON-serializable ints."""
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    tt = getattr(usage, "total_tokens", None)
    if tt is None:
        tt = pt + ct
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": int(tt or 0)}

OPENAI_ENGINEERED_SYSTEM = (
    "You are a creative ideation and research copilot. Deliver high-signal, original, practical ideas.\n"
    "Use clear structure, avoid filler, avoid repeating points, and include concrete examples.\n"
    "Prioritize non-obvious but actionable angles and brief rationale for each recommendation."
)

CREATIVITY_TASK_NUDGE = (
    "\n\nBefore you answer: prioritize original angles, non-obvious connections, and ideas a careful "
    "planner might skip—still stay on-topic and coherent."
)


def model_uses_default_sampling_controls(model: str) -> bool:
    """Return true for OpenAI models that should not be sent temperature/top-p controls."""
    mid = (model or "").strip().lower()
    return mid.startswith(("gpt-5", "o3", "o4"))


def format_user_block(raw: str, *, creativity_nudge: bool) -> str:
    text = raw.rstrip()
    if not creativity_nudge or not text:
        return text
    return text + CREATIVITY_TASK_NUDGE


def messages_to_prompt(system: str, user: str, tokenizer: Any) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system.strip()}\n\n{user.strip()}"


def openai_chat_single_turn(
    *,
    prompt: str,
    model: str,
    temperature: float,
    system: str,
    seed: int | None = None,
    return_usage: bool = False,
) -> str | tuple[str, dict[str, int]]:
    """One system + one user message; optional seed for reproducibility.

    If ``return_usage`` is true, returns ``(text, usage_dict)`` with token counts from the API.
    """
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is not set.")
    from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, OpenAI

    # Longer client timeout + small retry loop helps flaky TLS / spotty Wi‑Fi during long benchmarks.
    client = OpenAI(timeout=120.0, max_retries=0)
    kwargs: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if not model_uses_default_sampling_controls(model):
        kwargs["temperature"] = float(temperature)
    if seed is not None:
        kwargs["seed"] = int(seed)
    last_err: BaseException | None = None
    for attempt in range(8):
        try:
            r = client.chat.completions.create(**kwargs)
            text = (r.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("OpenAI returned an empty completion.")
            if return_usage:
                return text, _usage_dict(getattr(r, "usage", None))
            return text
        except BadRequestError as ex:
            msg = str(ex).lower()
            changed = False
            if "temperature" in kwargs and "temperature" in msg:
                kwargs.pop("temperature", None)
                changed = True
            if "seed" in kwargs and "seed" in msg:
                kwargs.pop("seed", None)
                changed = True
            if changed:
                continue
            raise
        except APIStatusError as ex:
            status_code = int(getattr(ex, "status_code", 0) or 0)
            if status_code == 429 or status_code >= 500:
                retry_after = 0.0
                response = getattr(ex, "response", None)
                if response is not None:
                    try:
                        retry_after = float(response.headers.get("retry-after", "0") or 0)
                    except (TypeError, ValueError):
                        retry_after = 0.0
                time.sleep(max(retry_after, min(90.0, 2.0 * (2**attempt))))
                last_err = ex
                continue
            raise
        except (APITimeoutError, APIConnectionError) as ex:
            last_err = ex
            time.sleep(min(90.0, 2.0 * (2**attempt)))
    assert last_err is not None
    raise last_err


def openai_chat_multiturn(
    *,
    turns: list[dict[str, str]],
    model: str,
    temperature: float,
    system: str = SYSTEM_STANDARD,
) -> str:
    """Chat Completions with system + user/assistant history (no tools)."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is not set.")
    from openai import OpenAI

    client = OpenAI()
    api_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    api_messages.extend(turns)
    r = client.chat.completions.create(
        model=model,
        temperature=float(temperature),
        messages=api_messages,
    )
    text = (r.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty completion.")
    return text
