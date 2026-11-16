"""Busy-moment selection and the per-track refinement tournament."""

import itertools
import shutil
from pathlib import Path

import numpy as np
import pytest

from tasteq.analysis import activity_moments
from tasteq.storage import Storage
from tasteq.synth import drum_loop, pink_noise, write_demo_library

pytest.importorskip("librosa")
pytest.importorskip("torch")

SR = 44_100
DURATION = 8.0


class SilentPlayer:
    def __init__(self) -> None:
        self.played = 0

    def start(self, audio: np.ndarray, sample_rate: int) -> None:
        self.played += 1

    def wait(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _song() -> np.ndarray:
    """Quiet pad, a drum section, a long soft stretch, then a second drum section."""
    rng = np.random.default_rng(0)
    pad = 0.05 * pink_noise(20.0, SR, rng)
    drums = drum_loop(20.0, SR, rng, bpm=140.0)
    parts = [pad, drums, pad, 0.6 * drums]
    return np.concatenate(parts)


def test_moments_are_busy_distinct_and_do_not_overlap() -> None:
    song = _song()
    starts = activity_moments(song, SR, n_moments=2, duration_s=DURATION)
    assert len(starts) == 2
    assert starts == sorted(starts)
    assert starts[1] - starts[0] >= DURATION
    for start in starts:  # mostly drums (20-40 s and 60-80 s), not the soft pad
        in_drums = sum(
            max(0.0, min(start + DURATION, end) - max(start, begin))
            for begin, end in ((20.0, 40.0), (60.0, 80.0))
        )
        assert in_drums >= 0.75 * DURATION


def test_moments_fall_back_for_short_tracks() -> None:
    assert activity_moments(np.ones(SR * 5), SR, n_moments=3, duration_s=12.0) == [0.0]
    assert activity_moments(np.ones(SR * 30), SR, n_moments=0) == []


def test_moments_never_exceed_what_fits() -> None:
    rng = np.random.default_rng(1)
    track = pink_noise(30.0, SR, rng)
    starts = activity_moments(track, SR, n_moments=10, duration_s=12.0)
    assert 1 <= len(starts) <= 2  # only two non-overlapping 12 s windows fit in 30 s
    assert all(0.0 <= s <= 30.0 - 12.0 for s in starts)


@pytest.fixture(scope="module")
def analyzed_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tasteq.library import analyze_folder

    folder = tmp_path_factory.mktemp("music")
    write_demo_library(folder, n_tracks=4, duration_s=40.0)
    path = tmp_path_factory.mktemp("db") / "refine.sqlite"
    with Storage(path) as storage:
        analyze_folder(folder, storage)
    return path


@pytest.fixture
def db(analyzed_db: Path, tmp_path: Path) -> Path:
    """A private copy, so tests that log comparisons do not see each other's."""
    copy = tmp_path / "copy.sqlite"
    shutil.copy(analyzed_db, copy)
    return copy


@pytest.mark.parametrize("model", ["random", "gp"])
def test_candidates_are_distinct_and_start_from_the_best_guess(db: Path, model: str) -> None:
    from tasteq.refine import candidate_points
    from tasteq.session import Tuner

    with Storage(db) as storage:
        tuner = Tuner(storage, model, seed=0)
        track = tuner.library.tracks[0]
        points = candidate_points(tuner, track, 6)
    assert points.shape == (6, tuner.space.dim)
    np.testing.assert_array_equal(points[0], tuner.best_point(track))
    assert points.min() >= 0.0
    assert points.max() <= 1.0
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    assert distances[np.triu_indices(6, k=1)].min() > 1e-6


def test_candidates_need_two_versions(db: Path) -> None:
    from tasteq.refine import candidate_points
    from tasteq.session import Tuner

    with Storage(db) as storage:
        tuner = Tuner(storage, "random")
        with pytest.raises(ValueError, match="two versions"):
            candidate_points(tuner, tuner.library.tracks[0], 1)


def test_tournament_logs_each_answer_and_returns_the_winner(db: Path) -> None:
    from tasteq.refine import run_refinement_for_track
    from tasteq.session import Tuner

    replies = itertools.cycle(["1", "=", "2", "r", "1"])
    player = SilentPlayer()
    with Storage(db) as storage:
        tuner = Tuner(storage, "gp", seed=0)
        track = tuner.library.tracks[0]
        result = run_refinement_for_track(
            tuner, track, player, lambda _: next(replies), n_versions=5, n_moments=2
        )
        tuner.close()
        assert result.answered == 4  # champion meets each of the 4 challengers once
        assert not result.quit_early
        rows = storage.comparison_rows()
        assert len(rows) == 4
        starts = {r["excerpt_start_s"] for r in rows}
        assert len(starts) == 2  # rotated through two moments
        assert all(0.0 <= s <= track.analysis.duration_s for s in starts)
        assert result.winner.shape == (tuner.space.dim,)


def test_tournament_quit_keeps_what_was_answered(db: Path) -> None:
    from tasteq.refine import run_refinement
    from tasteq.session import Tuner

    replies = iter(["1", "q"])
    with Storage(db) as storage:
        tuner = Tuner(storage, "random", seed=0)
        results = list(
            run_refinement(
                tuner, SilentPlayer(), lambda _: next(replies), tuner.library.tracks, n_versions=4
            )
        )
        assert len(results) == 1  # quitting ends the whole run
        assert results[0].quit_early
        assert results[0].answered == 1
        assert len(storage.comparison_rows()) == 1


def test_refine_command(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from tasteq import playback
    from tasteq.cli import app

    monkeypatch.setattr(playback, "default_player", lambda: SilentPlayer())
    result = CliRunner().invoke(
        app,
        ["refine", "--db", str(db), "--model", "random", "--versions", "3", "--max-tracks", "1"],
        input="1\n2\n",
    )
    assert result.exit_code == 0, result.output
    assert "2 answers on 1 tracks" in result.output
