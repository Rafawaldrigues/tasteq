"""Common interface for preference models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from tasteq.acquisition import TrustRegion, candidate_points
from tasteq.dsp import FloatArray
from tasteq.preference import Comparison, Outcome, as_context
from tasteq.space import EQSpace


class PreferenceModel(ABC):
    """Learns a utility over EQ settings from pairwise comparisons.

    All points are in the unit cube of ``space``; ``context`` is a vector in
    ``[0, 1]^context_dim`` describing the track (empty when ``context_dim`` is 0).

    The model keeps the full comparison history and refits on every update,
    so any model can be rebuilt exactly from stored comparisons.
    """

    name: str = "base"
    recommend_beta: float = 1.0
    recommend_radius: float = 0.05  # ±1 dB around heard settings in the raw space

    def __init__(
        self,
        space: EQSpace,
        context_dim: int = 0,
        seed: int = 0,
        trust_region: TrustRegion | None = None,
    ) -> None:
        self.space = space
        self.context_dim = context_dim
        self.trust_region = trust_region or TrustRegion()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.comparisons: list[Comparison] = []
        self._best_cache: dict[bytes, FloatArray] = {}

    @property
    def dim(self) -> int:
        """Dimension of the EQ search space."""
        return self.space.dim

    def update(self, comparisons: Sequence[Comparison]) -> None:
        """Add new comparisons and refit."""
        for comparison in comparisons:
            if comparison.a.shape != (self.dim,):
                raise ValueError(f"expected points of dimension {self.dim}")
            as_context(comparison.context, self.context_dim)
        self.comparisons.extend(comparisons)
        self._best_cache.clear()
        self._fit()

    @abstractmethod
    def _fit(self) -> None:
        """Refit the model on ``self.comparisons``."""

    @abstractmethod
    def predict(
        self, x: FloatArray, context: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Return utility mean and standard deviation for points ``x`` of shape ``(m, dim)``."""

    @abstractmethod
    def suggest_pair(self, context: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
        """Next pair to ask about: ``(incumbent, challenger)``."""

    def best(self, context: FloatArray | None = None) -> FloatArray:
        """Point with the highest predicted utility for this context (flat with no data)."""
        c = as_context(context, self.context_dim)
        key = c.tobytes()
        if key not in self._best_cache:
            self._best_cache[key] = self._search_best(c)
        return self._best_cache[key].copy()

    def best_gains(self, context: FloatArray | None = None) -> FloatArray:
        """:meth:`best` expressed as EQ gains in dB."""
        return self.space.to_gains(self.best(context))

    def _search_best(self, context: FloatArray) -> FloatArray:
        if not self.comparisons:
            return self.space.flat
        # Search only near settings that won at least once (or flat): a
        # recommendation should never be an extrapolation into a region with
        # no data, nor sit next to something the listener rejected.
        anchors = np.vstack([self.space.flat, self._observed_winners()])
        candidates = candidate_points(
            self.dim,
            self.rng,
            anchors=anchors,
            n_global=0,
            n_local_per_anchor=max(16, 2048 // len(anchors)),
            local_scale=self.recommend_radius,
        )
        score = self._recommendation_score(candidates, context)
        # Second pass: refine around the most promising candidates.
        top = candidates[np.argsort(score)[-8:]]
        refined = candidate_points(self.dim, self.rng, anchors=top, n_global=0, local_scale=0.03)
        refined_score = self._recommendation_score(refined, context)
        if np.max(refined_score) > np.max(score):
            return np.asarray(refined[int(np.argmax(refined_score))], dtype=np.float64)
        return np.asarray(candidates[int(np.argmax(score))], dtype=np.float64)

    def _recommendation_score(self, x: FloatArray, context: FloatArray) -> FloatArray:
        """Lower confidence bound: what we recommend should be good *and* known.

        Exploration is the acquisition's job. The recommendation (what plays
        in listen mode, and what the benchmark scores) must not land on a
        region the model has never seen just because it extrapolates high
        there, which flexible models such as MLPs readily do.
        """
        mean, std = self.predict(x, context)
        return mean - self.recommend_beta * std

    def _observed_winners(self) -> FloatArray:
        winners = [c.winner for c in self.comparisons if c.winner is not None]
        if not winners:
            return np.empty((0, self.dim))
        return np.unique(np.round(np.array(winners), 9), axis=0)

    def _radius(self, incumbent: FloatArray, context: FloatArray) -> float:
        """Trust-region radius: uncertainty-driven, shrunk by a streak of failures.

        The uncertainty term widens the box where the model knows little about
        its best guess. The failure term (as in TuRBO) shrinks it each time in
        a row the challenger loses, since that means steps of this size
        overshoot; any challenger win resets it.
        """
        radius = self.trust_region.radius_for(self._uncertainty_ratio(incumbent, context))
        shrunk = radius * self.trust_region.failure_shrink ** self._failure_streak()
        return max(self.trust_region.min_radius, shrunk)

    def _failure_streak(self) -> int:
        """Consecutive latest comparisons where the incumbent (``a``) beat the challenger.

        Ties end the streak instead of extending it: "no difference" means the
        step was too small to hear, so shrinking further would only produce
        more inaudible pairs.
        """
        streak = 0
        for comparison in reversed(self.comparisons):
            if comparison.outcome is not Outcome.A:
                break
            streak += 1
        return streak

    def _uncertainty_ratio(self, point: FloatArray, context: FloatArray) -> float:
        """``std(point) / std(far from data)``: how unsure the model is about ``point``."""
        reference = self.rng.uniform(size=(256, self.dim))
        _, std = self.predict(np.vstack([point[None, :], reference]), context)
        return float(std[0] / max(float(np.max(std[1:])), 1e-12))
