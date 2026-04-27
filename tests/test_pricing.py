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


def test_estimate_sibling_run_cost_under_budget():
    # Plan-04 budget guard: 300 prompts × 6 siblings should land under $5.
    assert pricing.estimate_sibling_run_cost(300, siblings=6) < 5.0


def test_estimate_sibling_run_cost_scales_with_siblings():
    one = pricing.estimate_sibling_run_cost(100, siblings=1)
    six = pricing.estimate_sibling_run_cost(100, siblings=6)
    # 1 prompter + 1 generator + 1 evaluator vs 1 prompter + 6 generator + 6 evaluator,
    # so 6× is strictly less than the 6-sibling estimate (prompter cost shared).
    assert six > one
    assert six < one * 6


def test_estimate_modal_cost_a100_80gb_two_hours():
    cost = pricing.estimate_modal_cost(2.0, gpu="A100-80GB")
    assert cost == pytest.approx(pricing.MODAL_GPU_HOURLY["A100-80GB"] * 2.0)


def test_estimate_modal_cost_unknown_gpu_raises():
    with pytest.raises(KeyError):
        pricing.estimate_modal_cost(1.0, gpu="not-a-real-gpu")
