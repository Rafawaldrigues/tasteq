"""Analysis, storage, sessions and the CLI on a tiny synthetic library."""

import itertools
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from tasteq.analysis import (
    ContextProjector,
    band_energy_db,
    diverse_subset,
    find_audio_files,
    representative_excerpt,
)
from tasteq.dsp import integrated_loudness
from tasteq.preference import Outcome
from tasteq.storage import Storage
from tasteq.synth import pink_noise, write_demo_library

pytest.importorskip("librosa")
pytest.importorskip("torch")

SR = 44_100


class SilentPlayer:
    def __init__(self) -> None:
        self.played: list[int] = []

    def start(self, audio: np.ndarray, sample_rate: int) -> None:
        self.played.append(audio.shape[0])

    def wait(self) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture(scope="module")
def library_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("music")
    write_demo_library(folder, n_tracks=4, duration_s=16.0)
    return folder


@pytest.fixture
def db(tmp_path: Path, library_dir: Path) -> Path:
    from tasteq.library import analyze_folder

    path = tmp_path / "test.sqlite"
    with Storage(path) as storage:
        report = analyze_folder(library_dir, storage)
    assert report.analyzed == 4
    return path


# --- analysis ---------------------------------------------------------------


def test_band_energy_follows_spectral_tilt() -> None:
    noise = pink_noise(4.0, SR, np.random.default_rng(0))
    bright = np.diff(noise, prepend=0.0)  # first difference tilts energy upwards
    assert band_energy_db(bright, SR)[-1] > band_energy_db(noise, SR)[-1] + 10


def test_excerpt_skips_quiet_intro() -> None:
    rng = np.random.default_rng(1)
    loud = pink_noise(20.0, SR, rng)
    track = np.concatenate([0.01 * pink_noise(10.0, SR, rng), loud])
    start = representative_excerpt(track, SR, duration_s=8.0)
    # At most one 1 s hop of the intro may leak in (it barely changes the level).
    assert start >= 9.0


def test_short_track_excerpt_starts_at_zero() -> None:
    assert representative_excerpt(np.ones(SR * 5), SR, duration_s=12.0) == 0.0


def test_context_projector_round_trip_and_range() -> None:
    features = np.random.default_rng(2).normal(size=(12, 14))
    projector = ContextProjector.fit(features, dim=4)
    contexts = projector.transform(features)
    assert contexts.shape == (12, 4)
    assert contexts.min() >= 0.0
    assert contexts.max() <= 1.0
    restored = ContextProjector.from_dict(projector.to_dict())
    np.testing.assert_allclose(restored.transform(features[0]), contexts[0])


def test_context_dim_is_capped_by_library_size() -> None:
    projector = ContextProjector.fit(np.random.default_rng(3).normal(size=(3, 14)), dim=6)
    assert projector.dim == 2


def test_diverse_subset_spreads_out() -> None:
    points = np.array([[0.5, 0.5], [0.51, 0.5], [0.0, 0.0], [1.0, 1.0], [0.52, 0.49]])
    chosen = diverse_subset(points, 3)
    assert set(chosen[1:]) == {2, 3}
    assert diverse_subset(points, 10) == diverse_subset(points, 5)


# --- storage and library --------------------------------------------------------


def test_analysis_is_cached_and_stored(db: Path, library_dir: Path) -> None:
    from tasteq.library import Library, analyze_folder

    with Storage(db) as storage:
        assert analyze_folder(library_dir, storage).cached == 4
        library = Library(storage)
        assert len(library.tracks) == len(find_audio_files(library_dir))
        assert library.contexts.shape == (4, library.context_dim)
        track = library.tracks[0]
        assert 0 <= track.analysis.excerpt_start_s <= track.analysis.duration_s


def test_library_requires_analysis(tmp_path: Path) -> None:
    from tasteq.library import Library

    with Storage(tmp_path / "empty.sqlite") as storage, pytest.raises(LookupError):
        Library(storage)


def test_renderer_matches_loudness(db: Path) -> None:
    from tasteq.library import Library, Renderer

    with Storage(db) as storage:
        track = Library(storage).tracks[0]
    renderer = Renderer()
    excerpt = renderer.excerpt(track, np.linspace(-10, 10, 10))
    assert excerpt.shape[0] == int(track.analysis.excerpt_duration_s * SR)
    assert abs(integrated_loudness(excerpt, SR) - renderer.target.integrated_lufs) <= 0.5


def test_meta_round_trip(tmp_path: Path) -> None:
    with Storage(tmp_path / "m.sqlite") as storage:
        storage.set_meta("k", {"a": [1, 2]})
        storage.set_meta("k", {"a": [3]})
        assert storage.get_meta("k") == {"a": [3]}
        assert storage.get_meta("missing", 7) == 7


# --- sessions -------------------------------------------------------------------


def test_answers_map_listening_order_to_outcomes(db: Path) -> None:
    from tasteq.session import Tuner

    with Storage(db) as storage:
        tuner = Tuner(storage, "random", seed=1)
        track = tuner.library.tracks[0]
        question = tuner.ask(track)
        first, second = question.presented
        expected = Outcome.A if question.a_first else Outcome.B
        assert question.outcome("first") is expected
        assert question.outcome("tie") is Outcome.TIE
        np.testing.assert_array_equal(first, question.a if question.a_first else question.b)
        assert second is not first


def test_calibration_logs_and_rebuilds_model(db: Path) -> None:
    from tasteq.calibrate import run_calibration
    from tasteq.session import Tuner

    replies = itertools.cycle(["1", "r", "2", "="])
    player = SilentPlayer()
    with Storage(db) as storage:
        tuner = Tuner(storage, "gp", seed=0)
        answered = run_calibration(tuner, player, lambda _: next(replies), n_tracks=2, per_track=2)
        tuner.close()
        assert answered == 4
        assert len(player.played) == 4 * 2 + 1 * 2  # one replay ('r') plays both again
        rows = storage.comparison_rows()
        assert len(rows) == 4
        assert {r["presented_first"] for r in rows} <= {"a", "b"}
        assert all(r["excerpt_start_s"] is not None for r in rows)

        rebuilt = Tuner(storage, "gp", seed=0)
        assert rebuilt.n_comparisons == 4
        track = rebuilt.library.tracks[0]
        np.testing.assert_allclose(rebuilt.best_gains(track), tuner.best_gains(track))


def test_calibration_quit(db: Path) -> None:
    from tasteq.calibrate import run_calibration
    from tasteq.session import Tuner

    with Storage(db) as storage:
        tuner = Tuner(storage, "random")
        assert run_calibration(tuner, SilentPlayer(), lambda _: "q", n_tracks=2) == 0
        assert storage.comparison_rows() == []


def test_listening_turns_thumbs_into_weak_comparisons(db: Path) -> None:
    from tasteq.listen import run_listening
    from tasteq.session import THUMB_WEIGHT, Tuner

    replies = iter(["+", "-", "", "1"])  # like, dislike, skip, then one A/B
    with Storage(db) as storage:
        tuner = Tuner(storage, "random", session_kind="listen")
        played = run_listening(
            tuner, SilentPlayer(), lambda _: next(replies), explore_every=3, max_tracks=3
        )
        assert played == 3
        rows = storage.comparison_rows()
        assert [r["source"] for r in rows] == ["thumb", "thumb", "ab"]
        assert [r["outcome"] for r in rows[:2]] == ["a", "b"]
        assert rows[0]["weight"] == THUMB_WEIGHT


# --- CLI ---------------------------------------------------------------------------


def test_cli_analyze_and_show(tmp_path: Path, library_dir: Path) -> None:
    from tasteq.cli import app

    db = tmp_path / "cli.sqlite"
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(library_dir), "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "4 analyzed" in result.output
    result = runner.invoke(app, ["show", "--db", str(db), "--model", "random"])
    assert result.exit_code == 0, result.output
    assert "synthetic_00" in result.output


def test_cli_demo_library(tmp_path: Path) -> None:
    from tasteq.cli import app

    result = CliRunner().invoke(app, ["demo-library", str(tmp_path / "lib"), "--tracks", "1"])
    assert result.exit_code == 0
    assert len(find_audio_files(tmp_path / "lib")) == 1
