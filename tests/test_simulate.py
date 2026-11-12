import numpy as np
import pytest

from tasteq.preference import Comparison, Outcome
from tasteq.simulate import BASS_BOOST, PERSONAS, Persona, optimum

CONTEXT = np.array([0.5, 0.5])


@pytest.mark.parametrize("name", PERSONAS)
def test_optimum_beats_flat_and_random_points(name: str) -> None:
    persona = PERSONAS[name]
    best_gains, best_u = optimum(persona, CONTEXT)
    assert best_gains.min() >= -10.0
    assert best_gains.max() <= 10.0
    flat_u = persona.utility_of(np.zeros(10), CONTEXT)[0]
    assert best_u - flat_u > 0.5, "flat must not be (nearly) optimal, or regret is ill-defined"
    others = np.random.default_rng(0).uniform(-10, 10, size=(2000, 10))
    assert persona.utility_of(others, CONTEXT).max() <= best_u + 1e-9


def test_quadratic_persona_optimum_is_its_target() -> None:
    gains, _ = optimum(PERSONAS["bass_lover"], CONTEXT)
    np.testing.assert_allclose(gains, BASS_BOOST, atol=0.05)


def test_genre_dependent_persona_uses_context() -> None:
    persona = PERSONAS["genre_dependent"]
    rock, _ = optimum(persona, np.array([0.0, 0.5]))
    vocal, _ = optimum(persona, np.array([1.0, 0.5]))
    assert rock[0] > vocal[0] + 5  # far more sub-bass on context 0


def test_choice_frequencies_follow_bradley_terry() -> None:
    persona = PERSONAS["bass_lover"]
    a, b = BASS_BOOST, np.zeros(10)
    gap = persona.utility_of(a, CONTEXT)[0] - persona.utility_of(b, CONTEXT)[0]
    rng = np.random.default_rng(1)
    wins = np.mean([persona.choose(a, b, CONTEXT, rng) is Outcome.A for _ in range(4000)])
    assert wins == pytest.approx(1 / (1 + np.exp(-gap)), abs=0.02)


def test_lapse_makes_answers_random() -> None:
    base = PERSONAS["bass_lover"]
    careless = Persona("careless", "", base.utility, True, lapse=1.0)
    rng = np.random.default_rng(2)
    answers = [careless.choose(BASS_BOOST, np.zeros(10), CONTEXT, rng) for _ in range(2000)]
    assert np.mean([o is Outcome.A for o in answers]) == pytest.approx(0.5, abs=0.04)


def test_near_identical_options_are_ties() -> None:
    persona = PERSONAS["bass_lover"]
    rng = np.random.default_rng(3)
    nudged = BASS_BOOST + 0.01
    assert persona.choose(BASS_BOOST, nudged, CONTEXT, rng) is Outcome.TIE


def test_comparison_labels_and_validation() -> None:
    a, b = np.zeros(3), np.ones(3)
    assert Comparison(a, b, CONTEXT, Outcome.A).winner is a
    assert Comparison(a, b, CONTEXT, Outcome.TIE).winner is None
    assert Comparison(a, b, CONTEXT, Outcome.TIE).label == 0.5
    with pytest.raises(ValueError, match="weight"):
        Comparison(a, b, CONTEXT, Outcome.A, weight=0.0)
    with pytest.raises(ValueError, match="same shape"):
        Comparison(a, np.ones(4), CONTEXT, Outcome.A)
