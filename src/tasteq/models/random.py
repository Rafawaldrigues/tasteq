"""Model-free baseline: a dueling hill climber with a trust region."""

from __future__ import annotations

import numpy as np

from tasteq.acquisition import TrustRegion
from tasteq.dsp import FloatArray
from tasteq.models.base import PreferenceModel
from tasteq.preference import as_context
from tasteq.space import EQSpace


class RandomTrustRegion(PreferenceModel):
    """Challenge the incumbent with a random point from the trust region.

    A winning challenger becomes the incumbent; a losing one shrinks the box.
    There is no model, so it ignores context and its "prediction" is simply
    closeness to the incumbent. It answers the question every learned model
    must beat: is the learning doing anything a random walk would not?
    """

    name = "random"

    def __init__(
        self,
        space: EQSpace,
        context_dim: int = 0,
        seed: int = 0,
        trust_region: TrustRegion | None = None,
        shrink: float = 0.8,
    ) -> None:
        super().__init__(space, context_dim, seed, trust_region)
        self.shrink = shrink
        self.incumbent = space.flat
        self.radius = self.trust_region.max_radius
        self._n_seen = 0

    def _fit(self) -> None:
        for comparison in self.comparisons[self._n_seen :]:
            winner = comparison.winner
            if winner is not None and not np.array_equal(winner, self.incumbent):
                self.incumbent = winner.copy()
            else:
                self.radius = max(self.trust_region.min_radius, self.radius * self.shrink)
        self._n_seen = len(self.comparisons)

    def predict(
        self, x: FloatArray, context: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Negative distance to the incumbent, with unit uncertainty everywhere."""
        as_context(context, self.context_dim)
        distance = np.linalg.norm(np.atleast_2d(x) - self.incumbent, axis=-1)
        return -distance, np.ones_like(distance)

    def best(self, context: FloatArray | None = None) -> FloatArray:
        """Return the incumbent (the same for every context)."""
        as_context(context, self.context_dim)
        return self.incumbent.copy()

    def suggest_pair(self, context: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
        """Incumbent vs. a uniform sample from the current trust region."""
        as_context(context, self.context_dim)
        challenger = self.trust_region.sample(self.incumbent, self.radius, 1, self.rng)[0]
        return self.incumbent.copy(), challenger
