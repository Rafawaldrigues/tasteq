"""Preference models and a registry to build them by name.

The torch-based models are imported lazily so the DSP and simulation code
works without the ``ml`` extra installed.
"""

from __future__ import annotations

from typing import Any

from tasteq.acquisition import TrustRegion
from tasteq.models.base import PreferenceModel
from tasteq.models.random import RandomTrustRegion
from tasteq.space import EQSpace

MODEL_KINDS = ("random", "gp", "neural")


def make_model(
    kind: str,
    space: EQSpace,
    context_dim: int = 0,
    seed: int = 0,
    trust_region: TrustRegion | None = None,
    **params: Any,
) -> PreferenceModel:
    """Build a model by kind name with optional model-specific parameters."""
    cls: type[PreferenceModel]
    if kind == "random":
        cls = RandomTrustRegion
    elif kind == "gp":
        from tasteq.models.gp import PreferenceGP

        cls = PreferenceGP
    elif kind == "neural":
        from tasteq.models.neural import NeuralBradleyTerry

        cls = NeuralBradleyTerry
    else:
        raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")
    return cls(space, context_dim, seed, trust_region, **params)


__all__ = ["MODEL_KINDS", "PreferenceModel", "make_model"]
