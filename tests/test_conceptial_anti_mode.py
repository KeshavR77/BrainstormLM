"""Unit tests for brainstormlm.conceptial.anti_mode.

Covers prompt formatting (format_pristine, format_anti_mode, exemplar_preamble),
brief-aware output-mode selection, exemplar→exclusion auto-dedup, and the
extract_summary regex chain.
"""

from __future__ import annotations

import pytest

from brainstormlm.conceptial.anti_mode import (
    GENERIC_SECTION_NAMES,
    OUTPUT_MODE_INSTRUCTIONS,
    exemplar_preamble,
    extract_summary,
    format_anti_mode,
    format_pristine,
)


# ---------- OUTPUT_MODE_INSTRUCTIONS ----------------------------------------------


def test_output_mode_instructions_has_three_modes():
    assert set(OUTPUT_MODE_INSTRUCTIONS) == {"image-prompt", "strategy", "generic"}


def test_image_prompt_instruction_forbids_markup():
    assert "SVG" in OUTPUT_MODE_INSTRUCTIONS["image-prompt"]
    assert "markup" in OUTPUT_MODE_INSTRUCTIONS["image-prompt"].lower()


def test_strategy_instruction_mentions_required_sections():
    instr = OUTPUT_MODE_INSTRUCTIONS["strategy"]
    for section in ("customer wedge", "proof metric", "pricing", "major risk"):
        assert section in instr.lower()


# ---------- exemplar_preamble ------------------------------------------------------


def test_exemplar_preamble_empty_list_returns_empty_string():
    assert exemplar_preamble("brief", []) == ""


def test_exemplar_preamble_renders_user_assistant_turns():
    out = exemplar_preamble("Make a logo.", ["Gearheads\nA crimson gear.", "Spark\nA single bolt."])
    assert "user: Make a logo." in out
    assert "assistant: Gearheads" in out
    assert "assistant: Spark" in out
    # Two examples → two user/assistant pairs.
    assert out.count("user:") == 2
    assert out.count("assistant:") == 2


# ---------- format_pristine --------------------------------------------------------


def test_format_pristine_includes_brief_and_image_instruction_by_default():
    out = format_pristine("Make a logo for a club.")
    assert "Make a logo for a club." in out
    assert "user:" in out and "assistant:" in out
    assert OUTPUT_MODE_INSTRUCTIONS["image-prompt"] in out


def test_format_pristine_unknown_mode_falls_back_to_image():
    fallback = format_pristine("brief", output_mode="not-a-real-mode")
    default = format_pristine("brief", output_mode="image-prompt")
    assert OUTPUT_MODE_INSTRUCTIONS["image-prompt"] in fallback
    # Modulo the mode arg, both should be byte-identical.
    assert fallback == default


def test_format_pristine_strategy_mode_uses_strategy_instruction():
    out = format_pristine("Generate a GTM plan.", output_mode="strategy")
    assert OUTPUT_MODE_INSTRUCTIONS["strategy"] in out
    assert "customer wedge" in out.lower()


def test_format_pristine_with_exemplars_includes_preamble():
    out = format_pristine("Make a logo.", exemplars=["EXAMPLE_NAME\nbody"])
    assert "EXAMPLE_NAME" in out


# ---------- format_anti_mode -------------------------------------------------------


def test_format_anti_mode_no_priors_no_exemplars_omits_exclusion_block():
    out = format_anti_mode("Make a logo.", prior_summaries=[])
    assert "MUST NOT repeat" not in out
    assert "Make a logo." in out
    assert OUTPUT_MODE_INSTRUCTIONS["image-prompt"] in out


def test_format_anti_mode_with_priors_lists_exclusions_as_bullets():
    out = format_anti_mode("Make a logo.", prior_summaries=["Gearheads", "Spark"])
    assert "MUST NOT repeat" in out
    assert "- Gearheads" in out
    assert "- Spark" in out


def test_format_anti_mode_strategy_mode_uses_strategy_instr():
    out = format_anti_mode("Generate a GTM plan.", prior_summaries=["Foo"], output_mode="strategy")
    assert OUTPUT_MODE_INSTRUCTIONS["strategy"] in out


def test_format_anti_mode_exemplar_names_added_to_exclusions():
    out = format_anti_mode(
        "Make a logo.",
        prior_summaries=["Prior Idea"],
        exemplars=["Gearheads\nA gear logo.", "Spark\nA bolt."],
    )
    assert "- Gearheads" in out
    assert "- Spark" in out
    assert "- Prior Idea" in out
    assert "STYLE references only" in out


def test_format_anti_mode_exemplar_dedupe_against_priors():
    # If a prior summary equals an exemplar name, it should appear exactly once.
    out = format_anti_mode(
        "Make a logo.",
        prior_summaries=["Gearheads"],
        exemplars=["Gearheads\nbody"],
    )
    assert out.count("- Gearheads") == 1


def test_format_anti_mode_strips_trailing_colon_from_exemplar_name_in_exclusion_bullet():
    # The exemplar's first line "NAME:" should appear in the exclusion bullet WITHOUT
    # the trailing colon (the strip only applies to the bullet form — the raw exemplar
    # body still renders in the preamble for the model to read).
    out = format_anti_mode(
        "Make a logo.",
        prior_summaries=[],
        exemplars=["GEARHEADS:\nA crimson gear."],
    )
    assert "- GEARHEADS\n" in out
    assert "- GEARHEADS:" not in out


def test_format_anti_mode_exemplar_only_still_emits_style_note():
    out = format_anti_mode("brief", prior_summaries=[], exemplars=["FOO\nbody"])
    assert "STYLE references only" in out


# ---------- format_anti_mode: motif_excludes (cliché-motif ban) -------------------


def test_format_anti_mode_motif_excludes_render_in_own_labeled_block():
    out = format_anti_mode(
        "Make a logo.",
        prior_summaries=[],
        motif_excludes=["gears", "cogs"],
    )
    # Motifs get their own block distinct from the concept-name block.
    assert "stock visual motifs" in out.lower()
    assert "- gears" in out
    assert "- cogs" in out


def test_format_anti_mode_motifs_fire_at_k_zero_with_empty_priors():
    # Even with no prior_summaries (i.e. k=0, first sample of a batch), motifs
    # must still render — they're a brief-level ban, not a per-sample exclusion.
    out = format_anti_mode(
        "Make a logo.",
        prior_summaries=[],
        motif_excludes=["gears"],
    )
    assert "- gears" in out
    assert "stock visual motifs" in out.lower()


def test_format_anti_mode_motifs_include_anti_cliche_closing_note():
    # When motifs are present, the formatter must add the anti-cliché closing
    # phrasing that frames them as off-limits even when reframed.
    out = format_anti_mode(
        "Make a logo.",
        prior_summaries=[],
        motif_excludes=["gears", "cogs"],
    )
    assert "cliché" in out.lower() or "cliche" in out.lower()
    # The closing should mention that reframing them isn't allowed.
    assert "reframed" in out.lower() or "reframing" in out.lower() or "synonym" in out.lower()


def test_format_anti_mode_no_motifs_back_compat_none():
    # Default (no kwarg) == explicit None == [] — all behave the same.
    a = format_anti_mode("Make a logo.", prior_summaries=["Foo"])
    b = format_anti_mode("Make a logo.", prior_summaries=["Foo"], motif_excludes=None)
    c = format_anti_mode("Make a logo.", prior_summaries=["Foo"], motif_excludes=[])
    assert a == b == c
    assert "stock visual motifs" not in a.lower()


def test_format_anti_mode_priors_and_motifs_render_together():
    out = format_anti_mode(
        "Make a logo.",
        prior_summaries=["3D Blueprint Vision"],
        motif_excludes=["gears", "cogs"],
    )
    # Both blocks present, separately labeled.
    assert "MUST NOT repeat" in out
    assert "- 3D Blueprint Vision" in out
    assert "stock visual motifs" in out.lower()
    assert "- gears" in out
    assert "- cogs" in out


def test_format_anti_mode_motifs_dedup_and_filter_empty_strings():
    out = format_anti_mode(
        "Make a logo.",
        prior_summaries=[],
        motif_excludes=["gears", "", "  ", "gears"],
    )
    # Empty/whitespace entries dropped; duplicates collapsed.
    assert out.count("- gears") == 1
    assert "-  \n" not in out and "- \n" not in out


def test_format_anti_mode_empty_priors_and_empty_motifs_uses_short_path():
    # With nothing to exclude, the no-exclusion early-return path runs as before.
    out = format_anti_mode("Make a logo.", prior_summaries=[], motif_excludes=[])
    assert "MUST NOT repeat" not in out
    assert "stock visual motifs" not in out.lower()


# ---------- extract_summary --------------------------------------------------------


def test_extract_summary_returns_concept_name_tag():
    assert extract_summary("CONCEPT_NAME: 24V Power Surge\nBody...") == "24V Power Surge"


def test_extract_summary_returns_markdown_header():
    assert extract_summary("## Gearheads\nFirst line of body.") == "Gearheads"


def test_extract_summary_returns_uppercase_id_with_colon():
    assert extract_summary("21ST_EYE: a stylized eye logo\nBody.") == "21ST_EYE"


def test_extract_summary_returns_bold_text():
    assert extract_summary("Some preamble. **Tone Catalyst** is the concept.") == "Tone Catalyst"


def test_extract_summary_numbered_list_picks_first_item():
    result = extract_summary("1. Crimson Gear\nA logo featuring a red gear.")
    assert "Crimson Gear" in result


def test_extract_summary_first_line_fallback_when_no_pattern_matches():
    # No CONCEPT_NAME, no header, no all-caps id, no bold, no numbered list,
    # but the first line is still a reasonable title.
    result = extract_summary("Pacific Compass\nThe second line of body content.")
    assert result == "Pacific Compass"


def test_extract_summary_skips_generic_section_names():
    # "Customer Wedge" is a generic section label, not a concept name.
    out = extract_summary("CONCEPT_NAME: Customer Wedge\nReal Concept Name\nbody")
    assert out != "Customer Wedge"


def test_extract_summary_skips_blank_line_literal():
    # Model occasionally emits the instruction "Blank line" as literal text. Filter.
    out = extract_summary("Blank line\nReal Concept Name\nbody.")
    assert out != "Blank line"


def test_extract_summary_empty_input_returns_empty():
    assert extract_summary("") == ""
    assert extract_summary("   \n  \n  ") == ""


def test_extract_summary_strips_trailing_punctuation():
    out = extract_summary("CONCEPT_NAME: Spark Engine.\n")
    assert out == "Spark Engine"


def test_extract_summary_length_lower_bound_filters_too_short_input():
    # All candidate lines are below the 4-char floor → returns "".
    assert extract_summary("Hi\nNo\nx") == ""


def test_extract_summary_length_upper_bound_caps_at_140():
    # Massive header line (200 chars) is filtered out, but the next line ("body")
    # falls through the line-by-line fallback. Either way: never longer than 140.
    long_line = "x" * 200
    out = extract_summary(f"## {long_line}\nshorter follow-up line that fits")
    assert len(out) <= 140


@pytest.mark.parametrize(
    "raw",
    [
        "customer wedge",
        "proof metric",
        "concept name",
        "blank line",
    ],
)
def test_generic_section_names_membership(raw):
    assert raw in GENERIC_SECTION_NAMES
