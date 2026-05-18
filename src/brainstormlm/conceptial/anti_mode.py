"""Cognitive anti-mode prompt formatter — the LOCKED Conceptial demo logic.

This module is the prompt-engineering layer that turns local Qwen2.5-3B (with v17 purified
steering) into a state-of-the-art diverse ideation engine. Three orthogonal pushes stack:

1. **Cognitive anti-mode at PROMPT level**: each new sample's prompt explicitly lists prior
   concept names as exclusions and asks the model to generate something fundamentally
   different. Breaks the "independent samples from the same prior" mode-collapse problem.

2. **Few-shot exemplars**: optional in-context demos (top-scoring concepts from prior runs)
   are injected as prior user→assistant turns. Pulls the model toward the exemplars' visual
   specificity and creativity. Exemplar names are auto-added to the exclusion list to
   prevent verbatim copying.

3. **Brief-aware output mode**: ``image-prompt`` for visual briefs (logos, design),
   ``strategy`` for business briefs (GTM, R&D), ``generic`` minimal. Without this, the
   default image-prompt instruction would override GTM-style briefs and produce visual
   concepts in response to strategic prompts.

The runtime side (model loading, SteeringEngine, IdeaMemoryStore persistence) lives in the
``BrainstormLM/scripts/smoke_cognitive_anti_mode.py`` CLI; this file is pure stdlib + re.
"""

from __future__ import annotations

import re

DEFAULT_PROMPT = "Give me a creative logo for an engineering club at Stanford."
CAPTURE_LAYER = 14  # L14 hidden captured (matches v17 creative-vector layer) for project memory parity


_CONCEPT_TAG_INSTRUCTION_IMAGE = (
    "Start with a memorable 2-6 word concept name as the very first line (any natural style is "
    "fine: title case, single coined word, or short phrase). Then on the following lines, write "
    "the concept as a natural-language image-generation prompt suitable for DALL-E, Midjourney, "
    "or SDXL — describe what the viewer would see in concrete visual detail. "
    "Do NOT include code, SVG, HTML, XML, or any markup. No <svg> tags, no path data, no XML "
    "attributes. Just plain English describing the visual."
)

_CONCEPT_TAG_INSTRUCTION_STRATEGY = (
    "Start with a memorable 2-6 word concept name as the very first line (any natural style is "
    "fine: title case, single coined word, or short phrase). Then on the following lines, "
    "elaborate the strategic concept with concrete structural detail. If the brief asks for "
    "specific sections (customer wedge, proof metric, pricing, major risk, etc.), include all "
    "of them as labeled subsections. Be concrete: specific numbers, specific personas, specific "
    "mechanisms. Avoid hand-wavy phrases like 'leverages AI to drive innovation'. Do NOT include "
    "code, markup, or visual-prompt-style imagery."
)

_CONCEPT_TAG_INSTRUCTION_GENERIC = (
    "Start with a memorable 2-6 word concept name as the very first line. Then provide the "
    "concept body with concrete detail. Stay tightly on-brief — answer the question being asked. "
    "Do NOT include code, SVG, HTML, XML, or any markup."
)

OUTPUT_MODE_INSTRUCTIONS: dict[str, str] = {
    "image-prompt": _CONCEPT_TAG_INSTRUCTION_IMAGE,
    "strategy": _CONCEPT_TAG_INSTRUCTION_STRATEGY,
    "generic": _CONCEPT_TAG_INSTRUCTION_GENERIC,
}


def exemplar_preamble(brief: str, exemplars: list[str]) -> str:
    """Render exemplars as prior user→assistant turns so the model treats them as in-context demos.

    The model is instruction-tuned on multi-turn chat, so a sequence of
    ``user: <brief>\\nassistant: <example>`` exchanges is parsed as "these are the kinds of
    responses I just produced." Anti-mode constraints in the final user turn keep the model
    from copying the exemplars verbatim.
    """
    if not exemplars:
        return ""
    parts: list[str] = []
    for ex in exemplars:
        parts.append(f"user: {brief.strip()}\nassistant: {ex.strip()}")
    return "\n\n".join(parts) + "\n\n"


def format_pristine(user_text: str, exemplars: list[str] | None = None, output_mode: str = "image-prompt") -> str:
    """Format a single-shot prompt (no anti-mode exclusions). Used by the pristine arm."""
    instr = OUTPUT_MODE_INSTRUCTIONS.get(output_mode, _CONCEPT_TAG_INSTRUCTION_IMAGE)
    preamble = exemplar_preamble(user_text, exemplars or [])
    return f"{preamble}user: {user_text.strip()}\n\n{instr}\nassistant: "


def format_anti_mode(
    user_text: str,
    prior_summaries: list[str],
    exemplars: list[str] | None = None,
    output_mode: str = "image-prompt",
    motif_excludes: list[str] | None = None,
    archetype_seed: str | None = None,
) -> str:
    """Format a sample N's prompt with anti-mode exclusions and optional few-shot exemplars.

    Exemplar concept names are auto-appended to the exclusion list to prevent verbatim copy.
    Args:
        user_text: the original brief.
        prior_summaries: concept names extracted from prior samples (this session + persisted
            across sessions via IdeaMemoryStore).
        exemplars: optional curated high-quality demo bodies.
        output_mode: one of ``image-prompt`` (default), ``strategy``, ``generic``.
        motif_excludes: optional brief-level cliché-motif ban (e.g. ``["gears", "cogs"]``
            for engineering briefs). Renders as a separate labeled block; fires from k=0
            (unlike ``prior_summaries`` which only fires k>=1 by construction). Used to
            break motif-level mode collapse that concept-name exclusions can't catch.
        archetype_seed: optional per-sample positive visual-vocabulary anchor (e.g.
            ``"biology — cellular structures"``). Pushes each sample into a different
            latent basin instead of just banning the current one. Imagery-scoped, not
            topic-scoped: the seed steers the visual language, not the subject matter.
    """
    instr = OUTPUT_MODE_INSTRUCTIONS.get(output_mode, _CONCEPT_TAG_INSTRUCTION_IMAGE)
    preamble = exemplar_preamble(user_text, exemplars or [])

    # Pull a name out of each exemplar (first non-empty line) so anti-mode excludes it.
    exemplar_names: list[str] = []
    for ex in exemplars or []:
        first_line = next((ln.strip() for ln in ex.splitlines() if ln.strip()), "")
        if first_line:
            first_line = re.sub(r"\s*[:\-—]\s*$", "", first_line)[:140]
            exemplar_names.append(first_line)
    concept_exclusions = exemplar_names + [s for s in prior_summaries if s not in exemplar_names]

    seen: set[str] = set()
    motif_list: list[str] = []
    for m in motif_excludes or []:
        s = (m or "").strip()
        if s and s not in seen:
            seen.add(s)
            motif_list.append(s)

    seed = (archetype_seed or "").strip()

    style_note = (
        " The example assistant responses above are STYLE references only — match their visual "
        "specificity, vivid concrete imagery, and ~150-250 word length, but produce a totally "
        "different concept. Do NOT reuse the example concept names ('"
        + "', '".join(exemplar_names) + "') or their core imagery."
        if exemplar_names
        else ""
    )

    seed_block = (
        f"Ground your visual vocabulary in: {seed}. Borrow imagery, structure, and "
        "metaphors from this domain while still solving the brief — the brief defines "
        "the subject; this seed defines the visual language."
        if seed
        else ""
    )

    if not concept_exclusions and not motif_list:
        parts = [user_text.strip()]
        if seed_block:
            parts.append(seed_block)
        parts.append(f"{instr}{style_note}")
        augmented = "\n\n".join(parts)
        return f"{preamble}user: {augmented}\nassistant: "

    sections: list[str] = [user_text.strip()]
    if concept_exclusions:
        bullet_lines = "\n".join(f"  - {s}" for s in concept_exclusions)
        sections.append(
            "Important constraint: the following concepts have already been used OR are example "
            "references — you MUST NOT repeat their names, core archetype, or rephrasings:\n"
            f"{bullet_lines}"
        )
    if motif_list:
        motif_lines = "\n".join(f"  - {m}" for m in motif_list)
        sections.append(
            "Avoid the following stock visual motifs entirely — they are clichés for this "
            "kind of brief and strong work omits them rather than reframing them:\n"
            f"{motif_lines}"
        )
    if seed_block:
        sections.append(seed_block)

    closing = (
        "Generate ONE completely different concept that does not share imagery, "
        "metaphor, or theme with the list above. Be specific and visual."
    )
    if motif_list:
        closing += (
            " The cliché motifs above are off-limits even when reframed or renamed — do not "
            "include them in any form, no synonyms, no metonymies. Choose a fundamentally "
            "different visual archetype than these defaults."
        )
    closing += style_note
    sections.append(closing)
    sections.append(instr)

    augmented = "\n\n".join(sections)
    return f"{preamble}user: {augmented}\nassistant: "


# --- Concept-name extraction ---------------------------------------------------------
# Used to populate the next sample's exclusion list and (post-hoc) to identify exemplars
# for the auto-ReasoningBank loop. The extractor handles 7 fallback patterns.

GENERIC_SECTION_NAMES = frozenset(
    s.lower()
    for s in (
        "customer wedge", "proof metric", "pricing", "major risk", "major risks",
        "go-to-market strategy", "go to market strategy", "concept overview",
        "introduction", "summary", "overview", "background", "approach",
        "implementation", "next steps", "conclusion", "considerations",
        # Literal artefacts from earlier prompt-instruction quirks (kept as a guardrail).
        "blank line", "concept body", "full concept body", "concept name",
    )
)

_SUMMARY_PATTERNS = (
    re.compile(r"^\s*CONCEPT_NAME\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*##+\s*(.+?)\s*$", re.MULTILINE),
    re.compile(r"^\s*([A-Z0-9][A-Z0-9_\-]{2,40})\s*:\s", re.MULTILINE),
    re.compile(r"\*\*([^*\n]{4,80})\*\*"),
    re.compile(
        r"^\s*((?:\d+[A-Za-z]+|[A-Z][A-Za-z0-9]+)(?:\s+(?:\d+[A-Za-z]+|[A-Z][A-Za-z0-9]+)){0,5})\s*[:—]\s",
        re.MULTILINE,
    ),
    re.compile(r"^\s*\d+[\.\)]\s*\*?\*?([^\.\n]{6,140})", re.MULTILINE),
    re.compile(r"^\s*([A-Z][A-Za-z][^\.\n]{6,120}?)[\.:!\n]", re.MULTILINE),
)


def extract_summary(text: str) -> str:
    """Best-effort 1-line concept name from a completion. Used for the next-sample exclusion list."""
    text = (text or "").strip()
    if not text:
        return ""
    for pat in _SUMMARY_PATTERNS:
        for m in pat.finditer(text):
            s = m.group(1).strip().rstrip(".,;:!*")
            s = re.sub(r"\s+", " ", s)
            if not (4 <= len(s) <= 140):
                continue
            if s.lower() in GENERIC_SECTION_NAMES:
                continue
            return s
    for line in text.splitlines():
        s = line.strip().rstrip(".,;:!*")
        s = re.sub(r"\s+", " ", s)
        if not s or not (4 <= len(s) <= 140):
            continue
        if s.lower() in GENERIC_SECTION_NAMES:
            continue
        return s
    return ""


__all__ = [
    "DEFAULT_PROMPT",
    "CAPTURE_LAYER",
    "OUTPUT_MODE_INSTRUCTIONS",
    "GENERIC_SECTION_NAMES",
    "exemplar_preamble",
    "format_pristine",
    "format_anti_mode",
    "extract_summary",
]
