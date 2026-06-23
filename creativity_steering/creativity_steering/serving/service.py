"""SteeringService — loads a package's vector set + a backend once, then applies the package
to each generation. ``steered=False`` runs the identical prompt with no injection, which is
the baseline arm for A/B formal testing.
"""

from __future__ import annotations

from creativity_steering.artifacts import SteeringVectorSet
from creativity_steering.serving.package import SteeringPackage


class SteeringService:
    def __init__(
        self,
        backend,
        vectorset: SteeringVectorSet,
        package: SteeringPackage,
        *,
        refiner=None,
    ):
        bad = sorted(L for L in package.all_layers() if not (0 <= L < vectorset.n_layers))
        if bad:
            raise ValueError(
                f"package layers {bad} out of range for a {vectorset.n_layers}-layer vector set"
            )
        self.backend = backend
        self.vs = vectorset
        self.package = package
        self.refiner = refiner  # optional concept-frozen post-step (steer -> refine)

    @classmethod
    def from_package(
        cls, package: SteeringPackage, *, backend=None, vectorset=None, refiner=None
    ) -> "SteeringService":
        """Build a live service: load the vector set from disk and (lazily) the HF model.
        Inject ``backend`` / ``vectorset`` to run offline (tests, custom backends). Pass
        ``refiner`` to enable the optional steer -> refine post-step."""
        if vectorset is None:
            vectorset = SteeringVectorSet.load(package.vectors_path)
        if backend is None:
            from creativity_steering.models.hf import HFBackend

            backend = HFBackend(
                model_id=package.model_id, device=package.device, dtype=package.dtype
            )
        return cls(backend, vectorset, package, refiner=refiner)

    def generate(
        self,
        prompt: str,
        *,
        variant: str | None = None,
        system: str | None = None,
        steered: bool = True,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        p = self.package
        injections = p.injections(self.vs, variant) if steered else []
        return self.backend.generate_steered(
            prompt,
            injections=injections,
            system=p.system if system is None else system,
            max_new_tokens=int(max_new_tokens) if max_new_tokens else p.max_new_tokens,
            temperature=float(temperature) if temperature is not None else p.temperature,
            seed=int(seed) if seed is not None else p.seed,
        )

    @property
    def has_refiner(self) -> bool:
        return self.refiner is not None

    def refine_text(self, prompt: str, text: str) -> str:
        """Polish steered candidates via the configured refiner (concept-frozen). Raises if no
        refiner was wired in."""
        if self.refiner is None:
            raise RuntimeError("no refiner configured for this service")
        return self.refiner.refine(prompt, text)

    def info(self) -> dict:
        d = self.package.public()
        d["capture_position"] = self.vs.capture_position
        d["n_layers"] = self.vs.n_layers
        d["refiner"] = self.refiner.info() if self.refiner is not None else None
        return d
