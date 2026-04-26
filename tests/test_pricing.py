import pytest

from brainstormlm.feedback import pricing


def test_cost_of_gpt_4o_mini_one_million_tokens_each_way():
    # gpt-4o-mini: $0.15 / Mtok in, $0.60 / Mtok out → 0.15 + 0.60 = 0.75
    assert pricing.cost_of("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_cost_of_zero_tokens_is_zero():
    assert pricing.cost_of("gpt-4o-mini", 0, 0) == 0.0


def test_cost_of_unknown_model_raises():
    with pytest.raises(KeyError):
        pricing.cost_of("not-a-real-model", 1, 1)


def test_estimate_run_cost_for_200_under_15_cents():
    # Phase-1 budget guard: a 200-trajectory run should land well under $0.15.
    assert pricing.estimate_run_cost(200) < 0.15


def test_estimate_run_cost_scales_linearly():
    one = pricing.estimate_run_cost(1)
    hundred = pricing.estimate_run_cost(100)
    assert hundred == pytest.approx(one * 100)


def test_prices_table_covers_required_models():
    for model in ("gpt-4o-mini", "gpt-4o", "Qwen/Qwen2.5-7B-Instruct-Turbo"):
        assert model in pricing.PRICES


def test_estimate_extraction_cost_for_200_under_25_cents():
    # Plan-03 budget guard: re-extracting lessons from 200 trajectories should
    # land well under $0.25 with gpt-4o-mini.
    assert pricing.estimate_extraction_cost(200) < 0.25


def test_estimate_extraction_cost_scales_linearly():
    one = pricing.estimate_extraction_cost(1)
    fifty = pricing.estimate_extraction_cost(50)
    assert fifty == pytest.approx(one * 50)
