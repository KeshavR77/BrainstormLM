"""Per-model pricing table and cost arithmetic for the feedback loop."""

PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
}

_TOKEN_BUDGET = {
    "prompter": (300, 80),
    "generator": (100, 500),
    "evaluator": (1200, 250),
}


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_per_m, out_per_m = PRICES[model]
    return (prompt_tokens / 1_000_000) * in_per_m + (completion_tokens / 1_000_000) * out_per_m


def estimate_run_cost(
    n: int,
    *,
    prompter_model: str = "gpt-4o-mini",
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
    evaluator_model: str = "gpt-4o-mini",
) -> float:
    p_in, p_out = _TOKEN_BUDGET["prompter"]
    g_in, g_out = _TOKEN_BUDGET["generator"]
    e_in, e_out = _TOKEN_BUDGET["evaluator"]
    per_trial = (
        cost_of(prompter_model, p_in, p_out)
        + cost_of(generator_model, g_in, g_out)
        + cost_of(evaluator_model, e_in, e_out)
    )
    return per_trial * n
