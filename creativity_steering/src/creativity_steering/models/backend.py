"""Backend abstraction shared by the mock and HF (Qwen) implementations.

A backend exposes just enough of a causal LM for the three phases:

  * ``generate`` / ``generate_and_capture`` — produce a response and cache its per-layer
    residual activations (derivation phase).
  * ``capture_activations`` — re-read activations for a fixed (prompt, response) pair at
    one or more token positions, so the same generation can yield several capture variants.
  * ``generate_steered`` — generate with ``alpha * vector`` added at chosen layers
    (activation-patching phase).

Activations cross the boundary as **numpy float32** of shape ``(n_layers, hidden_dim)``
per capture position, so the derivation/analysis core never imports torch. The layer
axis is indexed ``0 .. n_layers-1`` where layer ``L`` is the *output* of decoder block
``L``; that is exactly the residual stream a forward hook on block ``L`` modifies, which
keeps extraction and injection matched (the conformity paper's design).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class ActivationRecord:
    """A generated response plus its cached per-layer residuals, keyed by capture position."""

    text: str
    activations: dict[str, np.ndarray]  # position -> (n_layers, hidden_dim) float32

    def at(self, position: str) -> np.ndarray:
        if position not in self.activations:
            raise KeyError(
                f"position {position!r} not captured; have {sorted(self.activations)}"
            )
        return self.activations[position]


@dataclass
class Injection:
    """One steering intervention: add ``alpha * vector`` at decoder-block ``layer``'s output."""

    layer: int
    vector: np.ndarray  # (hidden_dim,)
    alpha: float
    rescale: bool = True  # scale to the per-token activation norm so alpha transfers across sizes


class ModelBackend(ABC):
    """Minimal causal-LM surface used by all three phases."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of decoder blocks (steerable layers 0 .. num_layers-1)."""

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> str:
        """Sample a single response to ``prompt`` under an optional ``system`` framing."""

    @abstractmethod
    def capture_activations(
        self,
        prompt: str,
        response: str,
        *,
        system: str | None = None,
        positions: Sequence[str],
    ) -> dict[str, np.ndarray]:
        """Forward ``prompt+response`` once; return ``{position: (n_layers, hidden_dim)}``."""

    @abstractmethod
    def generate_steered(
        self,
        prompt: str,
        *,
        injections: Sequence[Injection],
        system: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> str:
        """Generate with the given steering injections active for the whole decode."""

    # ------------------------------------------------------------------
    # Shared convenience — generate, then cache activations for that response.

    def generate_and_capture(
        self,
        prompt: str,
        *,
        system: str | None = None,
        positions: Sequence[str],
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        seed: int = 0,
    ) -> ActivationRecord:
        text = self.generate(
            prompt,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
        acts = self.capture_activations(prompt, text, system=system, positions=positions)
        return ActivationRecord(text=text, activations=acts)
