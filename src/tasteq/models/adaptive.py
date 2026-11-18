"""Neural model that separates a listener's overall taste from per-track adjustments."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from tasteq.acquisition import TrustRegion
from tasteq.models.neural import NeuralBradleyTerry, _mlp
from tasteq.space import EQSpace


class AdditiveUtility(nn.Module):
    """``f(x, c) = g(x) + h(x, c)``: a global taste plus a track-dependent deviation."""

    def __init__(self, dim: int, context_dim: int, hidden: int) -> None:
        super().__init__()
        self.dim = dim
        self.taste = _mlp(dim, hidden)
        self.deviation_net = _mlp(dim + context_dim, hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the utility of ``[point, context]`` rows, shape ``(m, 1)``."""
        utility: torch.Tensor = self.taste(inputs[:, : self.dim]) + self.deviation_net(inputs)
        return utility

    def deviation(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the track-dependent part alone, shape ``(m, 1)``."""
        deviation: torch.Tensor = self.deviation_net(inputs)
        return deviation


class TargetCurveUtility(nn.Module):
    """Distance to a preferred EQ curve: ``f(x, c) = -½ Σ ((x − t₀ − Δ(c)) / s)²``.

    ``t₀`` is the listener's overall target curve, ``Δ(c)`` a track-dependent
    shift of it and ``s`` a per-band tolerance. This is the shape of taste
    the EQ setting suggests (one best curve, worse the further away), and it
    turns a dozen answers into ten target gains to estimate instead of a
    whole function of ten inputs. The target starts at flat and ``Δ`` at
    zero (its last layer is zero-initialized).
    """

    def __init__(self, dim: int, context_dim: int, hidden: int) -> None:
        super().__init__()
        self.dim = dim
        self.target = nn.Parameter(torch.zeros(dim))
        self.log_tolerance = nn.Parameter(torch.full((dim,), -1.0))
        self.shift = _mlp(context_dim, hidden) if context_dim else None
        if self.shift is not None:
            last = nn.Linear(hidden, dim)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.shift[-1] = last

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the utility of ``[point, context]`` rows, shape ``(m, 1)``."""
        target = self.target + self.deviation(inputs)
        z = (inputs[:, : self.dim] - target) / self.log_tolerance.exp()
        return -0.5 * z.square().sum(dim=1, keepdim=True)

    def deviation(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the track-dependent shift of the target, shape ``(m, dim)``."""
        if self.shift is None:
            return torch.zeros(len(inputs), self.dim)
        shift: torch.Tensor = self.shift(inputs[:, self.dim :])
        return shift

    def audible_deviation(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the shift in units of the listener's tolerance, which is what they hear."""
        return self.deviation(inputs) / self.log_tolerance.exp()


Head = Literal["curve", "mlp"]


class AdaptiveNeuralBradleyTerry(NeuralBradleyTerry):
    """Bradley–Terry ensemble whose utility is global taste plus a per-track deviation.

    A listener's answers on any sound, including the synthetic sounds of a
    screening phase that look nothing like real songs, inform the global
    term ``g``. The deviation ``h`` is shrunk towards zero, both on the
    answered pairs and on the same EQ points under random contexts, so it
    only moves where comparisons on similar tracks disagree with the global
    taste. Without that evidence the model falls back to one EQ for every
    track instead of extrapolating a per-track curve from nothing.

    ``deviation_weight`` sets how much evidence a per-track adjustment
    needs: large values approach a single global EQ, small ones an
    unconstrained context model like :class:`NeuralBradleyTerry`.

    ``head`` picks the utility: ``"curve"`` (:class:`TargetCurveUtility`,
    a preferred curve per track) or ``"mlp"`` (:class:`AdditiveUtility`, any
    smooth function).
    """

    name = "neural_adaptive"

    def __init__(
        self,
        space: EQSpace,
        context_dim: int = 0,
        seed: int = 0,
        trust_region: TrustRegion | None = None,
        deviation_weight: float = 2.0,
        head: Head = "curve",
        **params: Any,
    ) -> None:
        super().__init__(space, context_dim, seed, trust_region, **params)
        self.deviation_weight = deviation_weight
        self.head = head

    def _make_member(self) -> nn.Module:
        if self.head == "curve":
            return TargetCurveUtility(self.dim, self.context_dim, self.hidden)
        return AdditiveUtility(self.dim, self.context_dim, self.hidden)

    def _extra_loss(
        self, member: nn.Module, a: torch.Tensor, b: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        assert isinstance(member, AdditiveUtility | TargetCurveUtility)
        observed = torch.cat([a, b])
        if self.context_dim == 0:
            return torch.zeros(())
        # Same EQ points under contexts drawn uniformly from [-1, 1] (inputs are centered).
        shuffled = observed.clone()
        shuffled[:, self.dim :] = (
            2.0 * torch.rand(len(observed), self.context_dim, generator=generator) - 1.0
        )
        # A target shift is penalized by how much it changes what is heard; in raw
        # input units it is tiny and would barely be held back.
        deviation = (
            member.audible_deviation if isinstance(member, TargetCurveUtility) else member.deviation
        )
        deviations = torch.cat([deviation(observed), deviation(shuffled)])
        return self.deviation_weight * deviations.square().mean()
