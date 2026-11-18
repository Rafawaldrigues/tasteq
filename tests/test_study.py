from pathlib import Path

import numpy as np
import pytest

from tasteq.bench import ModelEntry
from tasteq.simulate import optimum
from tasteq.study import (
    StudyConfig,
    load_tracks,
    make_participant,
    run_participant,
    study_personas,
    summarize,
)

TRACKS_FILE = Path(__file__).parents[1] / "configs" / "study_tracks.yaml"


def config(**overrides: object) -> StudyConfig:
    data: dict[str, object] = {
        "name": "test",
        "tracks_file": TRACKS_FILE,
        "seeds": 1,
        "modes": {"quick": {"screening": 4, "music": 8}},
        "models": [{"name": "random", "kind": "random"}],
    }
    data.update(overrides)
    return StudyConfig.model_validate(data)


def test_shipped_study_config_is_valid() -> None:
    for name in ("study.yaml", "study_audible.yaml"):
        cfg = StudyConfig.from_yaml(TRACKS_FILE.parent / name)
        assert set(cfg.modes) == {"rapida", "media", "longa"}


def test_study_tracks_cover_screening_sounds_and_styles() -> None:
    tracks = load_tracks(TRACKS_FILE)
    assert sum(t.synthetic for t in tracks) == 8
    assert len({t.style for t in tracks if not t.synthetic}) == 6
    assert all(t.context.shape == (4,) and t.band_db.shape == (10,) for t in tracks)


def test_participant_hears_chosen_styles_and_keeps_some_songs_back() -> None:
    cfg = config()
    tracks = load_tracks(TRACKS_FILE)
    who = make_participant(cfg, tracks, "study_by_style", "quick", 0)
    assert 2 <= len(who.styles) <= 3
    assert all(tracks[i].synthetic for i in who.screening)
    assert all(tracks[i].style in who.styles for i in who.heard + who.unheard)
    assert len(who.heard) == 2  # 8 music questions, 4 per song
    assert len(who.unheard) >= cfg.min_unheard
    assert not set(who.heard) & set(who.unheard)


def test_steady_listener_wants_the_same_eq_everywhere_others_do_not() -> None:
    tracks = load_tracks(TRACKS_FILE)
    personas = study_personas(tracks)
    songs = [t for t in tracks if not t.synthetic][:6]

    def spread(name: str) -> float:
        best = np.array([optimum(personas[name], t.context)[0] for t in songs])
        return float(best.std(axis=0).mean())

    assert spread("study_steady_bass") < 0.1
    assert spread("study_by_style") > 1.0
    assert spread("study_per_song") > 1.0


def test_run_participant_scores_after_screening_and_along_the_music_phase() -> None:
    cfg = config(eval_every=4)
    rows = run_participant(
        cfg, ModelEntry(name="random", kind="random"), "study_balance", "quick", 0
    )
    assert [r.n_music for r in rows] == [0, 4, 8]
    assert all(r.n_screening == 4 for r in rows)
    assert all(np.isfinite(r.heard_regret) and np.isfinite(r.unheard_regret) for r in rows)
    assert len({r.single_eq_regret for r in rows}) == 1
    assert all(0.0 <= r.tie_rate <= 1.0 for r in rows)


def test_summary_lists_every_mode_and_model() -> None:
    cfg = config()
    rows = [
        vars(r)
        for r in run_participant(
            cfg, ModelEntry(name="random", kind="random"), "study_steady_vocal", "quick", 0
        )
    ]
    text = summarize(rows)
    assert "| quick | random |" in text
    assert "study_steady_vocal" in text


def test_curve_head_starts_flat_and_learns_a_target() -> None:
    pytest.importorskip("torch")
    from tasteq.models import make_model
    from tasteq.preference import Comparison, Outcome
    from tasteq.space import EQSpace

    space = EQSpace()
    model = make_model("neural_adaptive", space, 2, n_members=3, epochs=80)
    context = np.array([0.5, 0.5])
    rng = np.random.default_rng(0)
    comparisons = []
    for _ in range(15):
        low = rng.uniform(0.4, 0.6, space.dim)
        high = low.copy()
        high[:3] += 0.1
        comparisons.append(Comparison(high, low, context, Outcome.A))
    model.update(comparisons)
    mean, _ = model.predict(
        np.vstack([space.flat, space.flat + np.r_[[0.1] * 3, [0.0] * 7]]), context
    )
    assert mean[1] > mean[0]
