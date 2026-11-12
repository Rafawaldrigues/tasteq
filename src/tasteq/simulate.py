"""Simulated listeners with hidden preferences.

A persona is a hidden utility over EQ gains (optionally depending on the
track context) plus a noisy choice model. The benchmark only lets models see
the persona's answers, then scores them against the hidden utility.

To keep the benchmark from being circular, the suite mixes
*well-specified* personas (smooth, unimodal: what a GP assumes) with
*misspecified* ones (asymmetric, ridge-shaped, multimodal, discontinuous,
noisy and inconsistent listeners).

Utilities are in logit units: with ``temperature = 1`` a utility gap of 1
means the better option is chosen about 73% of the time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import optimize

from tasteq.dsp import EQSpec, FloatArray
from tasteq.preference import Outcome

UtilityFn = Callable[[FloatArray, FloatArray], FloatArray]
"""``utility(gains (m, n_bands), context (k,)) -> (m,)``."""

SPEC = EQSpec()

# Target curves over the bands 31, 62, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz.
BASS_BOOST = np.array([7.0, 6.0, 4.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
V_SHAPE = np.array([5.0, 4.0, 1.0, -1.0, -3.0, -3.0, -1.0, 2.0, 4.0, 5.0])
DE_ESS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -5.0, -6.0, -3.0])
VOCAL_FORWARD = np.array([-2.0, -2.0, -1.0, 0.0, 2.0, 4.0, 4.0, 2.0, 0.0, 0.0])
BRIGHT = np.array([-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 2.0, 5.0, 7.0, 7.0])
GENTLE_WARMTH = np.array([4.0, 4.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0])


def _quadratic(target: FloatArray, sensitivity_db: float) -> UtilityFn:
    def utility(gains: FloatArray, context: FloatArray) -> FloatArray:
        del context
        return np.asarray(-0.5 * np.sum(((gains - target) / sensitivity_db) ** 2, axis=-1))

    return utility


def _asymmetric_de_ess(gains: FloatArray, context: FloatArray) -> FloatArray:
    # Boosting the sibilance region past the target hurts twice as much as
    # cutting it too much.
    error = gains - DE_ESS
    harsh = np.zeros(SPEC.n_bands, dtype=bool)
    harsh[[7, 8]] = True
    scale = np.where(harsh & (error > 0), 2.0, 1.0)
    return _quadratic(np.zeros(SPEC.n_bands), 4.5)(error * scale, context)


def _genre_dependent(gains: FloatArray, context: FloatArray) -> FloatArray:
    # Context[0] in [0, 1] moves the taste from bass-heavy to vocal-forward.
    mix = float(context[0]) if context.size else 0.0
    target = (1.0 - mix) * BASS_BOOST + mix * VOCAL_FORWARD
    return _quadratic(target, 3.0)(gains, context)


def _tonal_balance(gains: FloatArray, context: FloatArray) -> FloatArray:
    # Only the bass-minus-treble tilt matters, plus a dislike of jagged curves:
    # a ridge of equally good settings rather than a single optimum.
    del context
    tilt = gains[..., :4].mean(axis=-1) - gains[..., 6:].mean(axis=-1)
    roughness = np.sum(np.diff(gains, axis=-1) ** 2, axis=-1)
    return np.asarray(-0.5 * ((tilt - 8.0) / 2.0) ** 2 - 0.01 * roughness)


def _bimodal(gains: FloatArray, context: FloatArray) -> FloatArray:
    # Likes bass-heavy or bright, but not something in between.
    bass = _quadratic(BASS_BOOST, 3.0)(gains, context)
    bright = _quadratic(BRIGHT, 3.0)(gains, context) - 0.5
    return np.maximum(bass, bright)


def _harshness_cliff(gains: FloatArray, context: FloatArray) -> FloatArray:
    # Mild preferences, but any presence-region boost above +3 dB is suddenly
    # unbearable: a discontinuity no smooth model expects.
    base = _quadratic(GENTLE_WARMTH, 3.0)(gains, context)
    harsh = np.any(gains[..., 6:9] > 3.0, axis=-1)
    return np.asarray(base - 4.0 * harsh)


@dataclass(frozen=True)
class Persona:
    """A simulated listener: hidden utility plus a noisy way of answering."""

    name: str
    description: str
    utility: UtilityFn
    well_specified: bool
    temperature: float = 1.0
    lapse: float = 0.0
    tie_margin: float = 0.15

    def utility_of(self, gains: FloatArray, context: FloatArray) -> FloatArray:
        """Hidden utility of one or many gain vectors."""
        return np.asarray(self.utility(np.atleast_2d(gains), context), dtype=np.float64)

    def choose(
        self, a: FloatArray, b: FloatArray, context: FloatArray, rng: np.random.Generator
    ) -> Outcome:
        """Answer an A/B question about two gain vectors.

        With probability ``lapse`` the answer is a coin flip (inattention).
        Otherwise, near-identical options get "no difference" and the rest
        follow a Bradley–Terry model: ``P(A) = sigmoid(Δu / temperature)``.
        """
        if rng.random() < self.lapse:
            return Outcome.A if rng.random() < 0.5 else Outcome.B
        u = self.utility_of(np.stack([a, b]), context)
        gap = float(u[0] - u[1])
        if abs(gap) < self.tie_margin:
            return Outcome.TIE
        p_a = 1.0 / (1.0 + np.exp(-gap / self.temperature))
        return Outcome.A if rng.random() < p_a else Outcome.B


PERSONAS: dict[str, Persona] = {
    p.name: p
    for p in [
        Persona(
            "bass_lover",
            "Wants a strong low-end boost, neutral elsewhere.",
            _quadratic(BASS_BOOST, 3.5),
            well_specified=True,
        ),
        Persona(
            "v_shape",
            "The classic smile curve: more bass and treble, recessed mids.",
            _quadratic(V_SHAPE, 3.5),
            well_specified=True,
        ),
        Persona(
            "sibilance_hater",
            "Cuts 4-8 kHz and punishes any harsh boost there twice as hard.",
            _asymmetric_de_ess,
            well_specified=False,
        ),
        Persona(
            "genre_dependent",
            "Bass-heavy on one kind of track, vocal-forward on another (uses context).",
            _genre_dependent,
            well_specified=True,
        ),
        Persona(
            "tonal_balance",
            "Only cares about the bass/treble tilt: a ridge of equally good curves.",
            _tonal_balance,
            well_specified=False,
        ),
        Persona(
            "bimodal",
            "Likes bass-heavy or bright, dislikes anything in between.",
            _bimodal,
            well_specified=False,
        ),
        Persona(
            "harshness_cliff",
            "Mild taste with a hard cliff: any presence boost above +3 dB is rejected.",
            _harshness_cliff,
            well_specified=False,
        ),
        Persona(
            "noisy_listener",
            "Smile-curve taste, but answers with a lot of noise.",
            _quadratic(V_SHAPE, 3.5),
            well_specified=True,
            temperature=3.0,
        ),
        Persona(
            "inconsistent",
            "Bass lover who answers at random a quarter of the time.",
            _quadratic(BASS_BOOST, 3.5),
            well_specified=True,
            lapse=0.25,
        ),
    ]
}


def optimum(persona: Persona, context: FloatArray, spec: EQSpec = SPEC) -> tuple[FloatArray, float]:
    """Best gains inside the allowed range and their utility.

    Multi-start bounded search; exact enough to normalize regret. Cached per
    persona and context because the benchmark asks for it many times.
    """
    return _optimum_cached(persona.name, tuple(np.round(context, 9)), spec)


@lru_cache(maxsize=4096)
def _optimum_cached(
    name: str, context_key: tuple[float, ...], spec: EQSpec
) -> tuple[FloatArray, float]:
    persona = PERSONAS[name]
    context = np.array(context_key, dtype=np.float64)
    rng = np.random.default_rng(0)
    low, high = spec.min_gain_db, spec.max_gain_db

    starts = rng.uniform(low, high, size=(20_000, spec.n_bands))
    starts = np.vstack([starts, np.zeros(spec.n_bands)])
    for curve in (BASS_BOOST, V_SHAPE, DE_ESS, VOCAL_FORWARD, BRIGHT, GENTLE_WARMTH):
        starts = np.vstack([starts, curve])
    values = persona.utility_of(starts, context)

    best_x, best_u = starts[np.argmax(values)], float(np.max(values))
    bounds = [(low, high)] * spec.n_bands

    def loss(g: FloatArray) -> float:
        return -float(persona.utility_of(g, context)[0])

    for start in starts[np.argsort(values)[-8:]]:
        result = optimize.minimize(
            loss,
            start,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if -result.fun > best_u:
            best_x, best_u = np.asarray(result.x, dtype=np.float64), float(-result.fun)
    return best_x, best_u
