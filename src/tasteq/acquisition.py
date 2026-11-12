"""Where to look next: the adaptive trust region and candidate generation.

A ±10 dB range does not mean A and B should differ by 20 dB. Every model
proposes its challenger inside a box around the current best (the trust
region). The box is wide while the model is unsure about its best point and
narrows as that uncertainty falls, so early questions explore boldly and later
ones fine-tune without jarring jumps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from tasteq.dsp import FloatArray


@dataclass(frozen=True)
class TrustRegion:
    """Half-width of the search box around the incumbent, in unit-cube units.

    In the raw space the cube spans 20 dB per band, so the defaults allow
    steps from ±1 dB (confident; about the smallest EQ change people reliably
    hear) up to ±5 dB (exploring).
    """

    min_radius: float = 0.05
    max_radius: float = 0.25
    failure_shrink: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 < self.min_radius <= self.max_radius <= 1.0:
            raise ValueError("need 0 < min_radius <= max_radius <= 1")
        if not 0.0 < self.failure_shrink <= 1.0:
            raise ValueError("failure_shrink must be in (0, 1]")

    def radius_for(self, uncertainty_ratio: float) -> float:
        """Map ``std(best) / std(prior)`` in [0, 1] to a radius.

        A ratio of 1 means the model knows nothing about its best point yet
        (widest box); near 0 it is confident (narrowest box).
        """
        ratio = float(np.clip(uncertainty_ratio, 0.0, 1.0))
        return self.min_radius + (self.max_radius - self.min_radius) * ratio

    @staticmethod
    def bounds(center: FloatArray, radius: float) -> tuple[FloatArray, FloatArray]:
        """Box ``center ± radius`` clipped to the unit cube."""
        return np.clip(center - radius, 0.0, 1.0), np.clip(center + radius, 0.0, 1.0)

    def sample(
        self, center: FloatArray, radius: float, n: int, rng: np.random.Generator
    ) -> FloatArray:
        """Uniform samples from the box around ``center``."""
        low, high = self.bounds(center, radius)
        return rng.uniform(low, high, size=(n, center.shape[0]))


def candidate_points(
    dim: int,
    rng: np.random.Generator,
    anchors: FloatArray | None = None,
    n_global: int = 1024,
    n_local_per_anchor: int = 64,
    local_scale: float = 0.08,
) -> FloatArray:
    """Candidate set for maximizing a model over the unit cube.

    Sobol points cover the cube; Gaussian perturbations around ``anchors``
    (typically observed winners) refine near where the data is. Used by
    every model so that "best" is searched the same way for all of them.
    """
    parts = [qmc.Sobol(dim, scramble=True, seed=rng).random(n_global)] if n_global else []
    if anchors is not None and len(anchors):
        local = anchors[:, None, :] + local_scale * rng.standard_normal(
            (len(anchors), n_local_per_anchor, dim)
        )
        parts.extend([anchors, local.reshape(-1, dim)])
    return np.asarray(np.clip(np.vstack(parts), 0.0, 1.0), dtype=np.float64)
