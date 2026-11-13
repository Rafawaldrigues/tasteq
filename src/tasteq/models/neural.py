"""Neural Bradley–Terry model: MLP utilities, a deep ensemble and Thompson sampling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tasteq.acquisition import TrustRegion
from tasteq.dsp import FloatArray
from tasteq.models.base import PreferenceModel
from tasteq.preference import as_context
from tasteq.space import EQSpace


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 1),
    )


class NeuralBradleyTerry(PreferenceModel):
    """Ensemble of MLP utilities trained with the Bradley–Terry likelihood.

    ``P(a ≻ b) = sigmoid(f(a) − f(b))``. Each member is trained on a bootstrap
    resample of the comparisons from its own initialization; the spread across
    members is the uncertainty estimate. Ties use a soft label of 0.5 and
    weights scale each comparison's loss, so weak feedback counts less.

    Regularization matters a lot with this little data. Nearly separable
    comparisons push Bradley–Terry logits towards infinity, so a Gaussian
    prior on utilities (``prior_weight``) keeps them on a plausible scale.
    Weight decay is decoupled (AdamW): with plain Adam, L2 dominates the
    normalized step once the data gradient is small and collapses the network
    to a constant.

    The challenger is chosen by Thompson sampling: draw one member at random
    and take its argmax inside the trust region.
    """

    name = "neural"

    def __init__(
        self,
        space: EQSpace,
        context_dim: int = 0,
        seed: int = 0,
        trust_region: TrustRegion | None = None,
        n_members: int = 5,
        hidden: int = 64,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-4,
        n_challenger_candidates: int = 2048,
        prior_weight: float = 0.05,
    ) -> None:
        super().__init__(space, context_dim, seed, trust_region)
        self.n_members = n_members
        self.hidden = hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_challenger_candidates = n_challenger_candidates
        self.prior_weight = prior_weight
        self.members: list[nn.Sequential] = []
        self._offsets = np.zeros(0)

    def _inputs(self, x: FloatArray, context: FloatArray) -> torch.Tensor:
        points = np.atleast_2d(x)
        tiled = np.broadcast_to(context, (points.shape[0], self.context_dim))
        # Centered inputs train faster than [0, 1] ones.
        return torch.as_tensor(2.0 * np.hstack([points, tiled]) - 1.0, dtype=torch.float32)

    def _fit(self) -> None:
        a = torch.cat([self._inputs(c.a, c.context) for c in self.comparisons])
        b = torch.cat([self._inputs(c.b, c.context) for c in self.comparisons])
        labels = torch.tensor([c.label for c in self.comparisons], dtype=torch.float32)
        weights = torch.tensor([c.weight for c in self.comparisons], dtype=torch.float32)

        generator = torch.Generator().manual_seed(self.seed + len(self.comparisons))
        n = len(self.comparisons)
        self.members = []
        for _ in range(self.n_members):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
            member = _mlp(self.dim + self.context_dim, self.hidden)
            sample = torch.randint(0, n, (n,), generator=generator)  # bootstrap
            optimizer = torch.optim.AdamW(
                member.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            for _ in range(self.epochs):
                optimizer.zero_grad()
                fa, fb = member(a[sample]).squeeze(-1), member(b[sample]).squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    fa - fb, labels[sample], weight=weights[sample]
                )
                # Gaussian prior on utilities, f ~ N(0, 1 / (2 * prior_weight)).
                loss = loss + self.prior_weight * (fa.square().mean() + fb.square().mean())
                loss.backward()
                optimizer.step()
            member.eval()
            self.members.append(member)

        # Bradley–Terry only sees differences, so each member's utility is
        # defined up to a constant. Centering every member on the data it was
        # fit to removes that offset; otherwise the ensemble spread would
        # measure arbitrary constants instead of real disagreement. (Anchoring
        # at a single point such as flat would instead fake certainty there.)
        observed = torch.cat([a, b])
        with torch.no_grad():
            self._offsets = np.array([float(m(observed).mean()) for m in self.members])

    def _member_utilities(self, x: FloatArray, context: FloatArray) -> FloatArray:
        """Centered utilities per member, shape ``(n_members, m)``."""
        inputs = self._inputs(x, context)
        with torch.no_grad():
            raw = np.stack([m(inputs).squeeze(-1).numpy() for m in self.members])
        return (raw - self._offsets[:, None]).astype(np.float64)

    def predict(
        self, x: FloatArray, context: FloatArray | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Ensemble mean and standard deviation of the utility."""
        c = as_context(context, self.context_dim)
        points = np.atleast_2d(x)
        if not self.members:
            return np.zeros(points.shape[0]), np.ones(points.shape[0])
        utilities = self._member_utilities(points, c)
        return utilities.mean(axis=0), utilities.std(axis=0)

    def suggest_pair(self, context: FloatArray | None = None) -> tuple[FloatArray, FloatArray]:
        """Incumbent vs. the argmax of one randomly drawn ensemble member."""
        c = as_context(context, self.context_dim)
        incumbent = self.best(c)
        if not self.members:
            radius = self.trust_region.max_radius
            return incumbent, self.trust_region.sample(incumbent, radius, 1, self.rng)[0]

        radius = self._radius(incumbent, c)
        candidates = self.trust_region.sample(
            incumbent, radius, self.n_challenger_candidates, self.rng
        )
        sampled = self._member_utilities(candidates, c)[int(self.rng.integers(len(self.members)))]
        return incumbent, candidates[int(np.argmax(sampled))]
