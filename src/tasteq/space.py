"""Search spaces: how a point in the unit cube becomes an EQ gain vector.

Models always work in ``[0, 1]^dim``. The space decides what that means:

- ``raw``: one coordinate per band, mapped linearly to the gain range.
- ``dct``: a few smooth basis functions (DCT-II over band index, which is
  log-frequency because the bands are octaves) whose sum is the curve.
  Fewer dimensions and only smooth curves, at the cost of not being able to
  express jagged ones. Which is better is an empirical question for the
  benchmark, not an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from tasteq.dsp import EQSpec, FloatArray

SpaceKind = Literal["raw", "dct"]


@dataclass(frozen=True)
class EQSpace:
    """Map between the models' unit cube and EQ gains in dB."""

    kind: SpaceKind = "raw"
    n_components: int = 5
    spec: EQSpec = field(default_factory=EQSpec)

    def __post_init__(self) -> None:
        if self.kind == "dct" and not 1 <= self.n_components <= self.spec.n_bands:
            raise ValueError(f"n_components must be in [1, {self.spec.n_bands}]")

    @property
    def dim(self) -> int:
        """Dimension of the unit cube the models search."""
        return self.spec.n_bands if self.kind == "raw" else self.n_components

    @property
    def flat(self) -> FloatArray:
        """Unit-cube point that maps to a flat EQ."""
        return np.full(self.dim, 0.5)

    def to_gains(self, u: ArrayLike) -> FloatArray:
        """Map points of shape ``(..., dim)`` in ``[0, 1]`` to gains ``(..., n_bands)``."""
        points = np.asarray(u, dtype=np.float64)
        if points.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {points.shape}")
        centered = 2.0 * np.clip(points, 0.0, 1.0) - 1.0  # [-1, 1]
        half_range = (self.spec.max_gain_db - self.spec.min_gain_db) / 2.0
        midpoint = (self.spec.max_gain_db + self.spec.min_gain_db) / 2.0
        if self.kind == "raw":
            return midpoint + half_range * centered
        return self.spec.clip(midpoint + half_range * centered @ self.basis.T)

    @property
    def basis(self) -> FloatArray:
        """DCT-II basis, shape ``(n_bands, n_components)``, each column peaking at ±1."""
        n = self.spec.n_bands
        band = np.arange(n)[:, None] + 0.5
        order = np.arange(self.n_components)[None, :]
        return np.asarray(np.cos(np.pi * band * order / n), dtype=np.float64)
