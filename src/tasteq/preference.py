"""The unit of feedback: one pairwise comparison between two EQ settings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tasteq.dsp import FloatArray


class Outcome(Enum):
    """Answer to "which one do you prefer?"."""

    A = "a"
    B = "b"
    TIE = "tie"


@dataclass(frozen=True)
class Comparison:
    """A judged pair.

    ``a`` and ``b`` are points in the model's unit cube (see
    :class:`tasteq.space.EQSpace`); ``context`` describes the track they were
    played on. ``weight`` lets weaker evidence, such as a thumbs-down during
    playback, count less than a deliberate A/B answer.
    """

    a: FloatArray
    b: FloatArray
    context: FloatArray
    outcome: Outcome
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.a.shape != self.b.shape:
            raise ValueError("a and b must have the same shape")
        if not 0.0 < self.weight <= 1.0:
            raise ValueError("weight must be in (0, 1]")

    @property
    def winner(self) -> FloatArray | None:
        """The preferred point, or ``None`` for a tie."""
        if self.outcome is Outcome.TIE:
            return None
        return self.a if self.outcome is Outcome.A else self.b

    @property
    def label(self) -> float:
        """Probability-style target: 1 if A won, 0 if B won, 0.5 for a tie."""
        return {Outcome.A: 1.0, Outcome.B: 0.0, Outcome.TIE: 0.5}[self.outcome]


def as_context(context: FloatArray | None, dim: int) -> FloatArray:
    """Validate a context vector, treating ``None`` as the empty context."""
    c = np.zeros(0) if context is None else np.asarray(context, dtype=np.float64)
    if c.shape != (dim,):
        raise ValueError(f"expected a context of dimension {dim}, got shape {c.shape}")
    return c
