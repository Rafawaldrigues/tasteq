"""Preference Gaussian process (BoTorch ``PairwiseGP``) with EUBO acquisition."""

from __future__ import annotations

import contextlib
import warnings
from typing import Any

import numpy as np
import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.priors import GammaPrior

from tasteq.acquisition import TrustRegion
from tasteq.dsp import FloatArray
from tasteq.models.base import PreferenceModel
from tasteq.preference import as_context
from tasteq.space import EQSpace

DTYPE = torch.float64


def additive_context_kernel(dim: int, context_dim: int) -> ScaleKernel:
    """``k_x + k_x · k_c``: a shared taste plus a context-dependent correction.

    The first term is learned from every comparison regardless of track; the
    second only grows where the data shows the taste changes with context.
    Without context it reduces to a single ARD RBF kernel over the EQ.
    """
    eq_dims = tuple(range(dim))

    def eq_kernel() -> RBFKernel:
        return RBFKernel(
            ard_num_dims=dim, active_dims=eq_dims, lengthscale_prior=GammaPrior(2.4, 2.7)
        )

    if context_dim == 0:
        return ScaleKernel(eq_kernel(), outputscale_prior=GammaPrior(2.0, 1.0))
    context_kernel = RBFKernel(
        ard_num_dims=context_dim,
        active_dims=tuple(range(dim, dim + context_dim)),
        lengthscale_prior=GammaPrior(2.4, 2.7),
    )
    interaction: Kernel = ScaleKernel(
        eq_kernel() * context_kernel, outputscale_prior=GammaPrior(2.0, 4.0)
    )
    return ScaleKernel(eq_kernel() + interaction, outputscale_prior=GammaPrior(2.0, 1.0))


class PreferenceGP(PreferenceModel):
    """Gaussian process over ``[eq | context]`` fit with a Laplace approximation.

    Strict preferences only: ties carry no ordering for ``PairwiseGP``'s probit
    likelihood, so they are kept in the history but not used for fitting.
    Weights are likewise not supported by the likelihood and are ignored.

    The next challenger maximizes EUBO against the incumbent (BoTorch's
    ``previous_winner`` form), restricted to the trust region and with the
    context held fixed, so one of A/B is always the current best guess.
    """

    name = "gp"

    def __init__(
        self,
        space: EQSpace,
        context_dim: int = 0,
        seed: int = 0,
        trust_region: TrustRegion | None = None,
        num_restarts: int = 4,
        raw_samples: int = 128,
    ) -> None:
        super().__init__(space, context_dim, seed, trust_region)
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.model: PairwiseGP | None = None

    def _inputs(self, x: FloatArray, context: FloatArray) -> torch.Tensor:
        points = np.atleast_2d(x)
        tiled = np.broadcast_to(context, (points.shape[0], self.context_dim))
        return torch.as_tensor(np.hstack([points, tiled]), dtype=DTYPE)

    def _fit(self) -> None:
        index: dict[bytes, int] = {}
        rows: list[FloatArray] = []
        pairs: list[tuple[int, int]] = []

        def row_id(point: FloatArray, context: FloatArray) -> int:
            row = np.concatenate([point, context])
            key = np.round(row, 9).tobytes()
            if key not in index:
                index[key] = len(rows)
                rows.append(row)
            return index[key]

        for c in self.comparisons:
            if c.winner is None:
                continue
            loser = c.b if c.winner is c.a else c.a
            pairs.append((row_id(c.winner, c.context), row_id(loser, c.context)))

        if not pairs:
            self.model = None
            return
        torch.manual_seed(self.seed + len(self.comparisons))
        datapoints = torch.as_tensor(np.array(rows), dtype=DTYPE)
        comparisons = torch.as_tensor(pairs, dtype=torch.long)
        model = PairwiseGP(
            datapoints,
            comparisons,
            covar_module=additive_context_kernel(self.dim, self.context_dim),
        )
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Keep the prior hyperparameters if fitting fails.
            with contextlib.suppress(Exception):
                fit_gpytorch_mll(mll)
        self.model = model.eval()

    def predict(
        self, x: FloatArray, context: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Posterior mean and standard deviation of the latent utility."""
        c = as_context(context, self.context_dim)
        points = np.atleast_2d(x)
        if self.model is None:
            return np.zeros(points.shape[0]), np.ones(points.shape[0])
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            posterior: Any = self.model.posterior(self._inputs(points, c))
            mean = posterior.mean.squeeze(-1).numpy()
            std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
        return mean.astype(np.float64), std.astype(np.float64)

    def suggest_pair(self, context: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
        """Incumbent vs. the EUBO-maximizing challenger inside the trust region."""
        c = as_context(context, self.context_dim)
        incumbent = self.best(c)
        if self.model is None:
            radius = self.trust_region.max_radius
            challenger = self.trust_region.sample(incumbent, radius, 1, self.rng)[0]
            return incumbent, challenger

        radius = self._radius(incumbent, c)
        low, high = self.trust_region.bounds(incumbent, radius)
        bounds = torch.as_tensor(
            np.vstack([np.concatenate([low, c]), np.concatenate([high, c])]), dtype=DTYPE
        )
        fixed = {self.dim + i: float(v) for i, v in enumerate(c)}
        acquisition = AnalyticExpectedUtilityOfBestOption(
            pref_model=self.model, previous_winner=self._inputs(incumbent, c)
        )
        torch.manual_seed(self.seed + 7919 * len(self.comparisons))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate, _ = optimize_acqf(
                acquisition,
                bounds=bounds,
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                fixed_features=fixed or None,
            )
        challenger = candidate.detach().numpy()[0, : self.dim].astype(np.float64)
        if np.linalg.norm(challenger - incumbent) < 1e-3:
            challenger = self.trust_region.sample(incumbent, radius, 1, self.rng)[0]
        return incumbent, challenger
