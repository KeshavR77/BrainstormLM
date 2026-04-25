"""TogetherGenerator — hosted OSS base model behind the Generator protocol."""

from brainstormlm.clients import together as together_client
from brainstormlm.config import get_settings


class TogetherGenerator:
    def __init__(self, model: str | None = None, top_p: float | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.together_base_model
        self.top_p = top_p if top_p is not None else settings.default_top_p

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        return together_client.complete(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.top_p,
            use_cache=use_cache,
        )
