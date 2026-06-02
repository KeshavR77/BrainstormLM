"""Tests for the salad detector — pure stdlib, no network, no embeddings.

Mode-specific fixtures are padded past the 20-token micro_output floor so that
micro_output does not co-fire and muddy the targeted assertion.
"""

from __future__ import annotations

from brainstormlm.eval.salad import salad_detector, salad_rate

# 25 tokens, one comma, all-ASCII, no repeated run — a clean idea.
CLEAN = (
    "The bioluminescent garden concept pairs living algae walls with a quiet "
    "reading nook, letting visitors watch soft blue light ripple across the "
    "ceiling each evening."
)


def test_keys_present() -> None:
    assert set(salad_detector(CLEAN).keys()) == {
        "token_loop",
        "comma_bleed",
        "micro_output",
        "non_ascii",
        "is_salad",
    }


def test_clean_prose_passes_all_modes() -> None:
    result = salad_detector(CLEAN)
    assert result == {
        "token_loop": False,
        "comma_bleed": False,
        "micro_output": False,
        "non_ascii": False,
        "is_salad": False,
    }


def test_token_loop_consecutive_identical_tokens() -> None:
    text = (
        "The garden glows softly and then the light goes loop loop loop loop "
        "across the long mirrored ceiling above the visitors tonight"
    )
    result = salad_detector(text)
    assert result["token_loop"] is True
    assert result["is_salad"] is True
    # isolation: only token_loop fires
    assert result["micro_output"] is False
    assert result["comma_bleed"] is False
    assert result["non_ascii"] is False


def test_token_loop_repeated_phrase_block() -> None:
    text = (
        "After the long meeting everyone agreed to just go home go home go home "
        "and forget the entire stressful situation until the next morning arrives"
    )
    result = salad_detector(text)
    assert result["token_loop"] is True
    assert result["micro_output"] is False


def test_comma_bleed_runaway_list() -> None:
    text = (
        "red, blue, green, gold, teal, pink, lime, gray, black, white, brown, "
        "beige, cyan, plum, rose, sand, mint, navy, ruby, jade"
    )
    result = salad_detector(text)
    assert result["comma_bleed"] is True
    assert result["is_salad"] is True
    assert result["micro_output"] is False
    assert result["token_loop"] is False


def test_micro_output_short_text() -> None:
    result = salad_detector("too short to count as an idea")
    assert result["micro_output"] is True
    assert result["is_salad"] is True


def test_non_ascii_heavy_text() -> None:
    text = (
        "café naïve résumé Zürich señor jalapeño piñata fiancé déjà vu "
        "smörgåsbord façade crème brûlée Malmö Tromsø Köln Düsseldorf Bjørn Þór Æsir"
    )
    result = salad_detector(text)
    assert result["non_ascii"] is True
    assert result["is_salad"] is True
    assert result["micro_output"] is False


def test_empty_string_is_micro_salad() -> None:
    result = salad_detector("")
    assert result["micro_output"] is True
    assert result["is_salad"] is True


def test_salad_rate_mix() -> None:
    loop = "loop loop loop loop is the only thing this broken decoder ever wants to say now"
    assert salad_rate([CLEAN, loop]) == 0.5


def test_salad_rate_empty_list() -> None:
    assert salad_rate([]) == 0.0


def test_salad_rate_all_clean() -> None:
    assert salad_rate([CLEAN, CLEAN]) == 0.0
