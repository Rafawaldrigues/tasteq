"""Behavioral tests shared by every model, plus model-specific checks.

These use small settings so they stay fast; convergence quality is the
benchmark's job, not the unit tests'.
"""

from typing import Any

import numpy as np
import pytest

from tasteq.acquisition import TrustRegion, candidate_points
from tasteq.models import MODEL_KINDS, PreferenceModel, make_model
from tasteq.preference import Comparison, Outcome
from tasteq.simulate import PERSONAS
from tasteq.space import EQSpace

pytest.importorskip("torch")

SPACE = EQSpace()
CONTEXT = np.array([0.3, 0.7])
FAST: dict[str, dict[str, Any]] = {
    "random": {},
    "gp": {"num_restarts": 2, "raw_samples": 32},
    "neural": {"n_members": 3, "epochs": 60},
}


def build(kind: str, seed: int = 0) -> PreferenceModel:
    return make_model(kind, SPACE, context_dim=2, seed=seed, **FAST[kind])


def play(model: PreferenceModel, n: int, seed: int = 0) -> None:
    persona = PERSONAS["bass_lover"]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, b = model.suggest_pair(CONTEXT)
        outcome = persona.choose(SPACE.to_gains(a), SPACE.to_gains(b), CONTEXT, rng)
        model.update([Comparison(a, b, CONTEXT, outcome)])


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_cold_start_recommends_flat_and_proposes_inside_cube(kind: str) -> None:
    model = build(kind)
    np.testing.assert_array_equal(model.best(CONTEXT), SPACE.flat)
    a, b = model.suggest_pair(CONTEXT)
    assert a.shape == b.shape == (SPACE.dim,)
    assert np.all((b >= 0) & (b <= 1))
    assert np.abs(b - a).max() <= model.trust_region.max_radius + 1e-9


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_interface_contract_after_updates(kind: str) -> None:
    model = build(kind)
    play(model, 6)
    mean, std = model.predict(np.random.default_rng(0).uniform(size=(7, SPACE.dim)), CONTEXT)
    assert mean.shape == std.shape == (7,)
    assert np.all(std >= 0)
    best = model.best(CONTEXT)
    assert best.shape == (SPACE.dim,)
    assert np.all((best >= 0) & (best <= 1))
    assert model.best_gains(CONTEXT).shape == (10,)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_rebuilding_from_history_is_deterministic(kind: str) -> None:
    original = build(kind, seed=5)
    play(original, 5)
    rebuilt = build(kind, seed=5)
    rebuilt.update(original.comparisons)
    x = np.random.default_rng(1).uniform(size=(5, SPACE.dim))
    np.testing.assert_allclose(rebuilt.predict(x, CONTEXT)[0], original.predict(x, CONTEXT)[0])


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_rejects_wrong_dimensions(kind: str) -> None:
    model = build(kind)
    with pytest.raises(ValueError, match="dimension"):
        model.update([Comparison(np.zeros(3), np.ones(3), CONTEXT, Outcome.A)])
    with pytest.raises(ValueError, match="context"):
        model.best(np.zeros(5))


@pytest.mark.parametrize("kind", ["gp", "neural"])
def test_learned_models_rank_a_clear_preference(kind: str) -> None:
    # Every comparison says "more bass is better": the model must agree.
    model = build(kind)
    rng = np.random.default_rng(0)
    comparisons = []
    for _ in range(15):
        low = rng.uniform(0.3, 0.7, SPACE.dim)
        high = low.copy()
        high[:3] = np.clip(high[:3] + 0.15, 0, 1)
        comparisons.append(Comparison(high, low, CONTEXT, Outcome.A))
    model.update(comparisons)
    probe = np.full((2, SPACE.dim), 0.5)
    probe[0, :3] += 0.15
    mean, _ = model.predict(probe, CONTEXT)
    assert mean[0] > mean[1]


def test_gp_ignores_ties_without_failing() -> None:
    model = build("gp")
    model.update([Comparison(SPACE.flat, SPACE.flat + 0.01, CONTEXT, Outcome.TIE)])
    assert model.model is None  # type: ignore[attr-defined]
    np.testing.assert_array_equal(model.best(CONTEXT), SPACE.flat)


def test_random_baseline_moves_to_winner_and_shrinks_on_loss() -> None:
    model = build("random")
    start = model.radius  # type: ignore[attr-defined]
    challenger = SPACE.flat + 0.1
    model.update([Comparison(SPACE.flat, challenger, CONTEXT, Outcome.A)])
    assert model.radius < start  # type: ignore[attr-defined]
    model.update([Comparison(SPACE.flat, challenger, CONTEXT, Outcome.B)])
    np.testing.assert_array_equal(model.best(CONTEXT), challenger)


def test_failure_streak_shrinks_radius_and_ties_reset_it() -> None:
    model = build("neural")
    a, b = SPACE.flat, SPACE.flat + 0.1
    model.update([Comparison(a, b, CONTEXT, Outcome.A)] * 3)
    assert model._failure_streak() == 3
    model.update([Comparison(a, b, CONTEXT, Outcome.TIE)])
    assert model._failure_streak() == 0


def test_trust_region_maps_uncertainty_to_radius() -> None:
    region = TrustRegion(min_radius=0.05, max_radius=0.25)
    assert region.radius_for(1.0) == pytest.approx(0.25)
    assert region.radius_for(0.0) == pytest.approx(0.05)
    assert region.radius_for(7.0) == pytest.approx(0.25)
    low, high = region.bounds(np.array([0.02, 0.98]), 0.1)
    np.testing.assert_allclose(low, [0.0, 0.88])
    np.testing.assert_allclose(high, [0.12, 1.0])
    with pytest.raises(ValueError, match="radius"):
        TrustRegion(min_radius=0.3, max_radius=0.2)


def test_candidate_points_cover_cube_and_anchors() -> None:
    rng = np.random.default_rng(0)
    anchors = np.full((2, 4), 0.5)
    points = candidate_points(4, rng, anchors, n_global=64, n_local_per_anchor=8)
    assert points.shape == (64 + 2 + 16, 4)
    assert points.min() >= 0
    assert points.max() <= 1


def test_unknown_model_kind() -> None:
    with pytest.raises(ValueError, match="unknown model kind"):
        make_model("svm", SPACE)
