"""Deterministic, offline mock backend with a *planted* creativity direction.

This is a synthetic stand-in for the real Qwen so the whole derive → analyze → patch
pipeline runs with no model download and produces verifiable, non-trivial results:

  * Each layer ``L`` has a fixed unit "creativity axis" ``u_L`` and a salience ``w_L``
    that peaks at ``creative_layer``. A response's realised creativity ``c`` (a cheap
    lexical proxy of the generated text) is written into the residual as
    ``(2c - 1) * w_L * u_L`` plus content-independent noise. So DiffMeans over
    creative-vs-neutral responses recovers ``u_L`` with a per-layer norm peaking at
    ``creative_layer`` — exactly what the norm/cosine/oracle analysis should find.
  * Steered generation raises the text's creativity in proportion to how well the
    injected vector aligns with ``u_L`` at salient layers, so the activation-patching
    sweep recovers the planted layer / sign as the winner.

None of this is real semantics; it exists to exercise the machinery. Swap in
``HFBackend`` (config: ``backend="hf"``) for the real model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from creativity_steering.config import (
    CAPTURE_ASSISTANT_START,
    CAPTURE_LAST_RESPONSE,
    CAPTURE_MEAN_RESPONSE,
)
from creativity_steering.models.backend import Injection, ModelBackend

# A single vocabulary. The realised creativity of a response is controlled by how much
# of this vocabulary it is allowed to draw from: a low level uses only the first few
# words (highly repetitive → low distinct-n), a high level uses the whole list (→ high
# distinct-n). This gives a smooth, monotone creativity proxy with a wide range.
_VOCAB = (
    "the thing is good lighthouse origami quasar marmalade tessellate velvet glacier "
    "syncopate ember labyrinth cartographer zephyr kaleidoscope driftwood lantern paradox "
    "nebula saffron filigree monsoon obsidian trellis cobalt mirage cathedral plume "
    "fathom thistle archipelago vellum cinder harpsichord meadow alchemy aurora cipher"
).split()

# How much of the planted signal each capture position carries (the response hasn't been
# written yet at assistant_start, so it sees a weaker signal).
_POSITION_GAIN = {
    CAPTURE_LAST_RESPONSE: 1.0,
    CAPTURE_MEAN_RESPONSE: 0.9,
    CAPTURE_ASSISTANT_START: 0.3,
}

_CREATIVE_WORDS = {"creative", "inventive", "imaginative", "original", "unconventional",
                   "wildly", "surprising", "novel", "brainstorm"}
_NEUTRAL_WORDS = {"conventional", "standard", "ordinary", "plain", "factual",
                  "straightforward", "careful", "expected", "boring"}

_SIGNAL = 1.0
_NOISE = 0.30


def _seed_from(*parts: str) -> int:
    h = hashlib.sha256("␟".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def text_creativity(text: str) -> float:
    """Cheap [0,1] proxy: distinct-bigram ratio of the text (the mock's 'realised creativity')."""
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words[:-1], words[1:]))
    return len(set(bigrams)) / len(bigrams)


def _framing_level(system: str | None, prompt: str) -> float:
    """Map a prompt/system framing to a target creativity level in [0,1]."""
    blob = f"{system or ''} {prompt}".lower()
    if _CREATIVE_WORDS & set(re.findall(r"[a-z]+", blob)):
        return 0.9
    if _NEUTRAL_WORDS & set(re.findall(r"[a-z]+", blob)):
        return 0.1
    return 0.5


class MockBackend(ModelBackend):
    def __init__(
        self,
        *,
        num_layers: int = 24,
        hidden_dim: int = 128,
        creative_layer: int = 12,
        seed: int = 0,
    ) -> None:
        self._num_layers = int(num_layers)
        self._hidden_dim = int(hidden_dim)
        self._creative_layer = int(creative_layer)
        self._seed = int(seed)
        rng = np.random.default_rng(seed)
        # Per-layer salience, a Gaussian bump peaked at creative_layer.
        layers = np.arange(self._num_layers)
        sigma = max(self._num_layers / 8.0, 1.0)
        self.layer_salience = np.exp(
            -((layers - self._creative_layer) ** 2) / (2 * sigma**2)
        ).astype(np.float32)
        # A single global creativity direction shared across the salient band, blended with
        # per-layer noise. Band layers (high salience) point ~along `global_axis` and so are
        # mutually aligned (a high-cosine band, like the paper's layers 23-33); off-band
        # layers are ~random. This makes both the cosine-band analysis and the all-layer
        # injection (Experiment 2) behave like the real model.
        g = rng.normal(size=self._hidden_dim)
        self.global_axis = (g / np.linalg.norm(g)).astype(np.float32)
        r = rng.normal(size=(self._num_layers, self._hidden_dim))
        r = r / np.linalg.norm(r, axis=-1, keepdims=True)
        s = self.layer_salience[:, None]
        raw = s * self.global_axis[None, :] + (1.0 - s) * r
        self.creativity_axis = (raw / np.linalg.norm(raw, axis=-1, keepdims=True)).astype(np.float32)

    # ------------------------------------------------------------------ metadata
    @property
    def model_id(self) -> str:
        return f"mock-qwen-{self._num_layers}L-{self._hidden_dim}d"

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    # ------------------------------------------------------------------ generation
    def _render(self, level: float, salt: str, n_words: int = 60) -> str:
        rng = np.random.default_rng(_seed_from(salt, f"{level:.3f}"))
        level = float(np.clip(level, 0.0, 1.0))
        # Vocabulary size scales with level: more level → wider vocab → higher distinct-n.
        k = max(1, int(round(level * len(_VOCAB))))
        vocab = _VOCAB[:k]
        words = [str(rng.choice(vocab)) for _ in range(n_words)]
        # Group into ~10-word "sentences".
        sentences = [" ".join(words[i : i + 10]).capitalize() + "." for i in range(0, n_words, 10)]
        return " ".join(sentences)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        seed: int = 0,
        sampling=None,  # noqa: ANN001 — SamplingPreset, accepted for interface parity
    ) -> str:
        # The mock derives creativity from the framing words, not from decoding, so the
        # sampling preset is accepted (interface parity with HFBackend) but not applied.
        base = _framing_level(system, prompt)
        jitter = (np.random.default_rng(_seed_from(prompt, system or "", str(seed))).random() - 0.5) * 0.2
        n_words = max(20, min(max_new_tokens // 4, 80))
        return self._render(base + jitter, salt=f"{prompt}|{system}|{seed}", n_words=n_words)

    # ------------------------------------------------------------------ activations
    def _residual(self, c: float, salt: str, gain: float) -> np.ndarray:
        signed = (2.0 * float(np.clip(c, 0, 1)) - 1.0) * _SIGNAL * gain
        signal = signed * self.layer_salience[:, None] * self.creativity_axis  # (L, H)
        noise = np.random.default_rng(_seed_from(salt)).normal(
            scale=_NOISE, size=(self._num_layers, self._hidden_dim)
        ).astype(np.float32)
        return (signal + noise).astype(np.float32)

    def capture_activations(
        self,
        prompt: str,
        response: str,
        *,
        system: str | None = None,
        positions: Sequence[str],
    ) -> dict[str, np.ndarray]:
        c = text_creativity(response)
        out: dict[str, np.ndarray] = {}
        for pos in positions:
            gain = _POSITION_GAIN.get(pos, 1.0)
            out[pos] = self._residual(c, salt=f"{prompt}|{response}|{pos}", gain=gain)
        return out

    # ------------------------------------------------------------------ steering
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
        base = _framing_level(system, prompt)
        boost = 0.0
        for inj in injections:
            v = np.asarray(inj.vector, dtype=np.float32).reshape(-1)
            # Match HFBackend: an out-of-range layer is a caller error, not a silent no-op,
            # so a layer-index bug surfaces here in tests rather than on a paid GPU pod.
            if not 0 <= inj.layer < self._num_layers:
                raise ValueError(
                    f"injection layer {inj.layer} out of range [0, {self._num_layers})"
                )
            vn = np.linalg.norm(v)
            if vn < 1e-12:  # a zero direction (e.g. a padding row in a frame) is a valid no-op
                continue
            cos = float(np.dot(v / vn, self.creativity_axis[inj.layer]))
            # Alignment with the planted axis, weighted by where the layer matters.
            boost += inj.alpha * cos * float(self.layer_salience[inj.layer]) * 0.15
        n_words = max(20, min(max_new_tokens // 4, 80))
        salt = f"{prompt}|{system}|{seed}|{boost:.4f}"
        return self._render(base + boost, salt=salt, n_words=n_words)
