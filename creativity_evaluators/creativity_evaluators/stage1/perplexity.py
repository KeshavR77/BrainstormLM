"""Length-normalized perplexity under a fixed reference LM (``gpt2-medium``).

We report ``nll / n_tokens`` (mean negative log-likelihood per token) over the *response*
only — never the prompt. High PPL under a small, version-stable LM means unusual phrasing,
a mild originality signal: per Saakyan et al. (ICLR 2026) it correlates with originality but
*not* pragmaticality, so the composite treats it as one input among several, not a verdict.

gpt2-medium is loaded lazily; tests inject ``nll_fn`` (``text -> (total_nll, n_tokens)``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ..types import PPLResult

# text -> (summed token NLL, number of scored tokens)
NLLFn = Callable[[str], "tuple[float, int]"]


class PerplexityScorer:
    """Length-normalized NLL under a fixed reference LM."""

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        device: str | None = None,
        max_length: int = 1024,
        nll_fn: NLLFn | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._nll_fn = nll_fn
        self._lm = None  # lazily-built (tokenizer, model)

    def score(self, text: str) -> PPLResult:
        total_nll, n_tokens = self._nll(text)
        nll_per_token = total_nll / n_tokens if n_tokens > 0 else math.nan
        return PPLResult(nll_per_token=nll_per_token, n_tokens=n_tokens, model=self.model_name)

    def _nll(self, text: str) -> tuple[float, int]:
        if self._nll_fn is not None:
            return self._nll_fn(text)
        return self._gpt2_nll(text)

    def _gpt2_nll(self, text: str) -> tuple[float, int]:
        """Real path: summed cross-entropy NLL over the response tokens. Loaded lazily."""
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        if self._lm is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.eval()
            if self.device:
                model.to(self.device)
            self._lm = (tokenizer, model)
        tokenizer, model = self._lm

        ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).input_ids
        if self.device:
            ids = ids.to(self.device)
        n_scored = ids.shape[1] - 1  # next-token prediction shifts labels by one
        if n_scored <= 0:
            return 0.0, 0
        with torch.no_grad():
            loss = model(ids, labels=ids).loss  # mean NLL over the n_scored shifted tokens
        return float(loss.item()) * n_scored, n_scored
