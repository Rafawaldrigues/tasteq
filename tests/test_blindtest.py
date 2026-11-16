"""Blind test planning, logging and statistics."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from tasteq.blindtest import analyze, format_results, matched_random
from tasteq.storage import Storage
from tasteq.synth import write_demo_library


def rows(*trials: tuple[str, str, str]) -> list[dict[str, object]]:
    return [{"condition_a": a, "condition_b": b, "outcome": o} for a, b, o in trials]


def test_matched_random_has_same_strength() -> None:
    learned = np.array([6.0, 4.0, 2.0, 0.0, -1.0, -2.0, 0.0, 1.0, 3.0, 0.0])
    other = matched_random(learned, np.random.default_rng(0), limit_db=10.0)
    rms = np.sqrt(np.mean(np.square(learned)))
    assert np.sqrt(np.mean(np.square(other))) == pytest.approx(rms)
    assert abs(other.mean()) < 1e-9
    assert not np.allclose(other, learned)


def test_matched_random_of_flat_is_flat() -> None:
    assert np.all(matched_random(np.zeros(10), np.random.default_rng(0), 10.0) == 0)


def test_analyze_counts_wins_in_either_slot_and_drops_ties() -> None:
    data = rows(
        *[("learned", "flat", "a")] * 5,
        *[("flat", "learned", "b")] * 4,  # learned won from slot b
        ("flat", "learned", "a"),
        ("learned", "flat", "tie"),
        ("random", "learned", "b"),
    )
    primary, secondary = analyze(data)
    assert (primary.wins, primary.losses, primary.ties) == (9, 1, 1)
    assert primary.p_value == pytest.approx(11 / 1024)  # P(X >= 9), X ~ Bin(10, 0.5)
    assert primary.ci_low < 0.9 < primary.ci_high
    assert (secondary.wins, secondary.losses) == (1, 0)
    assert "preferred over flat more often than chance" in format_results([primary, secondary])


def test_analyze_without_answers() -> None:
    results = analyze([])
    assert all(r.n == 0 and r.p_value == 1.0 for r in results)
    assert "No decisive answers yet" in format_results(results)


# --- with a library ----------------------------------------------------------------


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("librosa")
    pytest.importorskip("torch")
    from tasteq.library import analyze_folder

    folder = tmp_path_factory.mktemp("music")
    write_demo_library(folder, n_tracks=3, duration_s=14.0)
    path = folder / "blind.sqlite"
    with Storage(path) as storage:
        analyze_folder(folder, storage)
    return path


def test_blind_test_logs_trials_without_training(db: Path) -> None:
    from tasteq.blindtest import BlindTest
    from tasteq.session import Tuner

    with Storage(db) as storage:
        tuner = Tuner(storage, "random", session_kind="test")
        test = BlindTest(tuner, n_trials=6, seed=3)
        pairs = [tuple(sorted(t.conditions)) for t in test.trials]
        assert pairs.count(("flat", "learned")) == pairs.count(("learned", "random")) == 3
        first, second = test.render()
        assert first.shape == second.shape
        for answer in ["first", "second", "tie", "first", "first", "second"]:
            test.answer(answer)  # type: ignore[arg-type]
        assert test.done
        assert storage.comparisons() == []
        logged = storage.blind_trials()
        assert len(logged) == 6
        assert {r["presented_first"] for r in logged} <= {"a", "b"}
        assert sum(r.n + r.ties for r in test.results()) == 6


def test_demo_blind_tab_flow(db: Path) -> None:
    from tasteq.demo import DemoSession
    from tasteq.session import Tuner

    with Storage(db) as storage:
        session = DemoSession(Tuner(storage, "random", session_kind="demo"))
        assert "comparisons" in session.blind_status()
        audio = session.start_blind_test(n_trials=2)
        assert audio[0][1].dtype == np.int16
        assert session.blind_status().startswith("Trial **1**")
        assert session.answer_blind("first") is not None
        assert session.answer_blind("second") is None
        assert "| learned vs flat |" in session.blind_status()
        with pytest.raises(RuntimeError):
            session.answer_blind("first")


def test_cli_blindtest_report(db: Path) -> None:
    from tasteq.cli import app

    result = CliRunner().invoke(app, ["blindtest-report", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "blind trials" in result.output
