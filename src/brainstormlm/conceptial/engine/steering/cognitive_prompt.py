"""
Conceptial V6 — native cognitive steering: verbalized orthogonality planning for solution-set batching.

Appends a strict chain-of-thought requirement to the user brief; server strips ``<thinking>...</thinking>``
from returned text while optionally re-encoding **final-idea** completions into IOR memory.
"""

from __future__ import annotations

import re

# Flexible open tag (models sometimes garble attributes); standard or missing close handled in strip.
_THINKING_PAIR = re.compile(
    r"<\s*thinking\b[^>]*>.*?</\s*thinking\s*>",
    re.DOTALL | re.IGNORECASE,
)
_ORPHAN_THINKING_OPEN = re.compile(r"<\s*thinking\b[^>]*>", re.IGNORECASE)
_STANDALONE_CLOSE = re.compile(r"</\s*thinking\s*>", re.IGNORECASE)


def format_cognitive_steering_prompt(base_prompt: str, previous_ideas: list[str]) -> str:
    """
    Augment ``base_prompt`` with cognitive-steering instructions for sequential diverse generation.

    When ``previous_ideas`` is empty, require a ``<thinking>`` brainstorm block before the answer.
    Otherwise require explicit analysis of overlap / cliché latent space and an orthogonal plan.
    """
    base = (base_prompt or "").strip()
    tag_note = " Use the exact tags `<thinking>` and `</thinking>` on their own lines; do not alter the spelling."
    if not previous_ideas:
        suffix = (
            "\n\nBefore answering, open a <thinking> block: decide which high-level **solution category** "
            "you will occupy, then brainstorm an unconventional approach within it."
            + tag_note
        )
        return f"{base}{suffix}"

    listed = ", ".join(str(x).strip() for x in previous_ideas if str(x).strip())
    if not listed:
        listed = "(none)"
    suffix = (
        "\n\n"
        f"You previously generated these concepts: {listed}. You must NOT overlap with these. Open a "
        "<thinking> block. Explicitly analyze the 'cliché latent space' of your past answers, and "
        "mathematically plan a radically orthogonal creative direction. "
        "To guarantee orthogonality, you MUST shift the fundamental category of the solution. "
        "If the previous ideas were digital campaigns, make this a physical product change. "
        "If they were stunts, make this a B2B partnership. Force a categorical leap. "
        "Close the block, then output your final idea."
        + tag_note
    )
    return f"{base}{suffix}"


def strip_thinking_block(text: str) -> str:
    """
    Remove ``<thinking>...`` regions: well-formed pairs, flexible open tags, unclosed preambles
    (through first ``\\n\\n``), and stray closes. Case-insensitive.

    Also drops whole lines that *begin* with a ``<thinking`` token (covers truncated / degenerate
    pseudo-tags like ``<thinking and`` that never close with ``>``).
    """
    t = text or ""
    while True:
        nxt = _THINKING_PAIR.sub("", t)
        if nxt == t:
            break
        t = nxt
    m = _ORPHAN_THINKING_OPEN.search(t)
    if m:
        rest = t[m.end() :]
        dbl = rest.find("\n\n")
        if dbl != -1:
            t = (t[: m.start()] + rest[dbl + 2 :]).strip()
        else:
            line_end = rest.find("\n")
            if line_end != -1:
                t = (t[: m.start()] + rest[line_end + 1 :]).strip()
            else:
                t = t[: m.start()].strip()
    t = _STANDALONE_CLOSE.sub("", t)
    # Degenerate: line starts with "<thinking" but never formed a valid `...>` opener.
    t = "\n".join(
        ln for ln in t.split("\n") if not re.match(r"\s*<\s*thinking\b", ln, flags=re.IGNORECASE)
    )
    t = re.sub(r"<\s*thinking\b[^>]*$", "", t, flags=re.MULTILINE | re.IGNORECASE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
