"""The browser demo's state machine, without a browser."""

from pathlib import Path

import numpy as np
import pytest

from tasteq.storage import Storage
from tasteq.synth import write_demo_library

pytest.importorskip("librosa")
pytest.importorskip("torch")


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tasteq.library import analyze_folder

    folder = tmp_path_factory.mktemp("music")
    write_demo_library(folder, n_tracks=3, duration_s=14.0)
    path = folder / "demo.sqlite"
    with Storage(path) as storage:
        analyze_folder(folder, storage)
    return path


def test_to_int16_keeps_level() -> None:
    from tasteq.demo import to_int16

    out = to_int16(np.array([0.5, -2.0, 0.0]))
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, [16384, -32767, 0])


def test_answers_are_logged_and_tracks_rotate(db: Path) -> None:
    from tasteq.demo import DemoSession

    with Storage(db) as storage:
        from tasteq.session import Tuner

        session = DemoSession(Tuner(storage, "random", session_kind="demo"), questions_per_track=2)
        (rate, first), (_, second) = session.current()
        assert rate == session.tuner.renderer.sample_rate
        assert first.dtype == np.int16
        assert first.shape == second.shape
        start = session.track
        session.answer("first")
        assert session.track is start
        session.answer("tie")
        assert session.track is not start
        rows = storage.comparison_rows()
        assert [r["outcome"] for r in rows][-1] == "tie"
        assert len(rows) == 2


def test_select_and_curve(db: Path) -> None:
    from tasteq.demo import DemoSession
    from tasteq.session import Tuner

    with Storage(db) as storage:
        session = DemoSession(Tuner(storage, "gp", session_kind="demo"))
        label = session.labels[-1]
        session.select(label)
        assert session.status().count(label) == 1
        session.answer("second")
        figure = session.eq_figure()
        assert figure.axes[0].get_title(loc="left") == "Your EQ for this track"
        learned, flat = session.learned_vs_flat()
        assert learned[1].shape == flat[1].shape


def test_build_app(db: Path) -> None:
    pytest.importorskip("gradio")
    from tasteq.demo import DemoSession, build_app
    from tasteq.session import Tuner

    with Storage(db) as storage:
        app = build_app(DemoSession(Tuner(storage, "random", session_kind="demo")))
        assert app is not None
