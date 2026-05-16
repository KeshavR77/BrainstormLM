"""Iterative Orthogonal Rejection (QD / latent-space diversity)."""

from brainstormlm.conceptial.engine.ior.engine import IOREngine, IORRunState, attach_iorengine
from brainstormlm.conceptial.engine.ior.rejection import orthogonal_reject

__all__ = ["IORRunState", "IOREngine", "attach_iorengine", "orthogonal_reject"]
