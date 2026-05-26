"""Quality reward-model scorer.

Catches the *appropriateness* half of "novelty x appropriateness" — without it, jumbled
gibberish would look maximally novel and score as "creative". We use NoveltyBench's choice,
``Skywork/Skywork-Reward-Gemma-2-27B-v0.2`` (a top RewardBench performer), with an 8B
fallback for environments where the 27B is impractical.

The 27B is slow, so every score is cached to disk keyed on (model, prompt, response). The
model is loaded lazily via the standard ``AutoModelForSequenceClassification`` reward API;
tests inject ``score_fn`` instead.
"""

from __future__ import annotations

from collections.abc import Callable

from ..utils.caching import DiskCache

SKYWORK_27B = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
SKYWORK_8B = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

ScoreFn = Callable[[str, str], float]


class RewardModelScorer:
    """Scalar quality reward for a (prompt, response), with on-disk memoization.

    Parameters
    ----------
    model_name, fallback_model, use_fallback:
        ``use_fallback=True`` swaps the default 27B for the 8B model.
    device, dtype:
        Forwarded to the lazily-loaded HF model.
    cache:
        Optional ``DiskCache``; when present, scores are read/written keyed on
        (active_model, prompt, response).
    score_fn:
        Inject a ``(prompt, response) -> float`` to bypass the HF model (testing).
    """

    def __init__(
        self,
        model_name: str = SKYWORK_27B,
        fallback_model: str = SKYWORK_8B,
        use_fallback: bool = False,
        device: str | None = None,
        dtype: str = "bfloat16",
        cache: DiskCache | None = None,
        score_fn: ScoreFn | None = None,
    ):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.use_fallback = use_fallback
        self.device = device
        self.dtype = dtype
        self.cache = cache
        self._score_fn = score_fn
        self._model = None  # lazily-built (tokenizer, model)

    @property
    def active_model(self) -> str:
        return self.fallback_model if self.use_fallback else self.model_name

    def score(self, prompt: str, response: str) -> float:
        if self.cache is not None:
            key = DiskCache.make_key(self.active_model, prompt, response)
            cached = self.cache.get(key)
            if cached is not None:
                return float(cached["score"])
        value = self._raw_score(prompt, response)
        if self.cache is not None:
            self.cache.set(key, {"score": value})
        return value

    def _raw_score(self, prompt: str, response: str) -> float:
        if self._score_fn is not None:
            return float(self._score_fn(prompt, response))
        return self._skywork_score(prompt, response)

    def _skywork_score(self, prompt: str, response: str) -> float:
        """Real path: Skywork reward via the standard sequence-classification API (lazy)."""
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if self._model is None:
            tokenizer = AutoTokenizer.from_pretrained(self.active_model)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.active_model,
                torch_dtype=getattr(torch, self.dtype, None),
                num_labels=1,
                device_map=self.device or "auto",
            )
            model.eval()
            self._model = (tokenizer, model)
        tokenizer, model = self._model

        convo = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        inputs = tokenizer.apply_chat_template(
            convo, tokenize=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            logits = model(inputs).logits
        return float(logits[0][0].item())
