"""Browser demo: blind A/B questions and the learned EQ curve, on top of :class:`Tuner`.

The demo adds no learning logic of its own. :class:`DemoSession` holds the
state behind the page (which question is open, which track comes next) and is
plain Python, so it is tested without a browser; :func:`build_app` only wires
it to Gradio components.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tasteq.calibrate import calibration_tracks
from tasteq.dsp import DEFAULT_CENTER_FREQS_HZ, FloatArray, frequency_response_db
from tasteq.report import GRID, MODEL_COLORS, MUTED, TEXT
from tasteq.session import Answer, Question, Tuner
from tasteq.storage import StoredTrack

if TYPE_CHECKING:
    from matplotlib.figure import Figure

AudioOut = tuple[int, np.ndarray]
CURVE_FREQS_HZ = np.geomspace(20.0, 20_000.0, 256)
QUESTIONS_PER_TRACK = 4

INTRO = """\
# tasteq

Listen to both versions and pick the one you prefer. They are the same
excerpt with two different EQs, **matched in loudness**, in random order.
After each answer the model updates and proposes a new pair. The curve on the
right is the EQ it would play for this track right now; the shaded band shows
how unsure it still is.
"""


def to_int16(audio: FloatArray) -> np.ndarray:
    """Convert float audio to int16 so the browser player does not renormalize it.

    Gradio rescales float input to full scale, which would undo the loudness
    matching that makes the comparison fair.
    """
    return np.asarray(np.round(np.clip(audio, -1.0, 1.0) * 32767), dtype=np.int16)


def track_label(track: StoredTrack) -> str:
    """Short name shown in the track picker."""
    return Path(track.analysis.path).stem


class DemoSession:
    """State behind the demo page; every method is safe to call from any thread."""

    def __init__(self, tuner: Tuner, questions_per_track: int = QUESTIONS_PER_TRACK) -> None:
        self.tuner = tuner
        self.questions_per_track = questions_per_track
        self.lock = threading.Lock()
        # Visit tracks in the same diverse order as terminal calibration.
        self.order = calibration_tracks(tuner, len(tuner.library.tracks))
        self.by_label = {track_label(t): t for t in self.order}
        self.track = self.order[0]
        self.asked_on_track = 0
        self.question: Question | None = None

    @property
    def labels(self) -> list[str]:
        """Track names in visiting order."""
        return list(self.by_label)

    def select(self, label: str) -> tuple[AudioOut, AudioOut]:
        """Switch to a track and open a question on it."""
        with self.lock:
            self.track = self.by_label[label]
            self.asked_on_track = 0
            return self._open_question()

    def answer(self, answer: Answer) -> tuple[AudioOut, AudioOut]:
        """Record the answer to the open question, then open the next one."""
        with self.lock:
            if self.question is None:
                raise RuntimeError("no open question")
            self.tuner.answer(self.question, answer)
            self.asked_on_track += 1
            if self.asked_on_track >= self.questions_per_track:
                index = self.order.index(self.track)
                self.track = self.order[(index + 1) % len(self.order)]
                self.asked_on_track = 0
            return self._open_question()

    def current(self) -> tuple[AudioOut, AudioOut]:
        """Audio of the open question, opening one if needed."""
        with self.lock:
            if self.question is None or self.question.track is not self.track:
                return self._open_question()
            return self._render(self.question)

    def learned_vs_flat(self) -> tuple[AudioOut, AudioOut]:
        """Render the current track's excerpt with the learned EQ and without EQ."""
        with self.lock:
            rate = self.tuner.renderer.sample_rate
            gains = self.tuner.best_gains(self.track)
            renderer = self.tuner.renderer
            return (
                (rate, to_int16(renderer.excerpt(self.track, gains))),
                (rate, to_int16(renderer.excerpt(self.track, np.zeros_like(gains)))),
            )

    def status(self) -> str:
        """One line about the model's progress."""
        n = self.tuner.n_comparisons
        hint = " (recommendations get reliable after ~30)" if n < 30 else ""
        return (
            f"**{n}** comparisons learned{hint} · now playing **{track_label(self.track)}**, "
            f"question {self.asked_on_track + 1} of {self.questions_per_track}"
        )

    def eq_figure(self) -> Figure:
        """Learned EQ curve for the current track with a 10-90% plausibility band."""
        import matplotlib.pyplot as plt

        with self.lock:
            best = self.tuner.best_gains(self.track)
            plausible = self.tuner.plausible_gains(self.track)
            n = self.tuner.n_comparisons
        curve = frequency_response_db(best, CURVE_FREQS_HZ)
        spread = np.stack([frequency_response_db(g, CURVE_FREQS_HZ) for g in plausible])
        low, high = np.percentile(spread, [10, 90], axis=0)

        color = MODEL_COLORS["gp"]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.fill_between(CURVE_FREQS_HZ, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(CURVE_FREQS_HZ, curve, color=color, linewidth=2.2)
        ax.scatter(DEFAULT_CENTER_FREQS_HZ, best, color=color, s=14, zorder=3)
        ax.axhline(0.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_xlim(20, 20_000)
        ax.set_ylim(-11, 11)
        ax.set_xticks([31, 125, 500, 2000, 8000])
        ax.set_xticklabels(["31", "125", "500", "2k", "8k"])
        ax.set_xlabel("frequency (Hz)", color=MUTED, fontsize=9)
        ax.set_ylabel("gain (dB)", color=MUTED, fontsize=9)
        title = "Your EQ for this track" if n else "No answers yet: flat EQ"
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def _open_question(self) -> tuple[AudioOut, AudioOut]:
        self.question = self.tuner.ask(self.track)
        return self._render(self.question)

    def _render(self, question: Question) -> tuple[AudioOut, AudioOut]:
        first, second = self.tuner.render_question(question)
        rate = self.tuner.renderer.sample_rate
        return (rate, to_int16(first)), (rate, to_int16(second))


def build_app(session: DemoSession) -> Any:
    """Gradio Blocks page for ``session``."""
    import gradio as gr

    def refresh(audio: tuple[AudioOut, AudioOut]) -> tuple[Any, ...]:
        first, second = audio
        return (
            first,
            second,
            session.eq_figure(),
            session.status(),
            gr.update(value=track_label(session.track)),
        )

    with gr.Blocks(title="tasteq") as app:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                track = gr.Dropdown(session.labels, value=session.labels[0], label="Track")
                first = gr.Audio(label="Version 1", type="numpy", format="wav", interactive=False)
                second = gr.Audio(label="Version 2", type="numpy", format="wav", interactive=False)
                with gr.Row():
                    prefer_first = gr.Button("Prefer 1", variant="primary", min_width=0)
                    no_difference = gr.Button("Same", min_width=0)
                    prefer_second = gr.Button("Prefer 2", variant="primary", min_width=0)
            with gr.Column(scale=1):
                curve = gr.Plot(label="Learned EQ", show_label=False)
                status = gr.Markdown()
                with gr.Accordion("Hear the learned EQ against flat", open=False):
                    play = gr.Button("Render this track")
                    learned = gr.Audio(label="Learned EQ", format="wav", interactive=False)
                    flat = gr.Audio(label="Flat (no EQ)", format="wav", interactive=False)

        outputs = [first, second, curve, status, track]
        app.load(lambda: refresh(session.current()), outputs=outputs)
        track.input(lambda label: refresh(session.select(label)), inputs=track, outputs=outputs)
        for button, answer in (
            (prefer_first, "first"),
            (no_difference, "tie"),
            (prefer_second, "second"),
        ):
            button.click(lambda a=answer: refresh(session.answer(a)), outputs=outputs)
        play.click(session.learned_vs_flat, outputs=[learned, flat])
    return app
