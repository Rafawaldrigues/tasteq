"""The tuning loop, independent of any user interface.

:class:`Tuner` owns the model, the library and the log. A front end (the
CLI, the Gradio demo) asks it for the next question, plays the two versions
and reports the answer; everything else, including randomizing which
version is heard first and writing the comparison to the database, happens
here so every front end behaves the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tasteq.dsp import FloatArray
from tasteq.library import Library, Renderer
from tasteq.models import PreferenceModel, make_model
from tasteq.preference import Comparison, Outcome
from tasteq.space import EQSpace
from tasteq.storage import ComparisonRecord, Storage, StoredTrack

Answer = Literal["first", "second", "tie"]
THUMB_WEIGHT = 0.5


@dataclass(frozen=True)
class Question:
    """One A/B question as it will be heard.

    ``a`` is the model's incumbent and ``b`` its challenger; ``a_first``
    records which one the listener hears first (randomized against order
    effects and ear adaptation).
    """

    track: StoredTrack
    context: FloatArray
    a: FloatArray
    b: FloatArray
    a_first: bool

    @property
    def presented(self) -> tuple[FloatArray, FloatArray]:
        """Unit-cube points in listening order: (first, second)."""
        return (self.a, self.b) if self.a_first else (self.b, self.a)

    def outcome(self, answer: Answer) -> Outcome:
        """Translate an answer about listening order into an A/B outcome."""
        if answer == "tie":
            return Outcome.TIE
        first_won = answer == "first"
        return Outcome.A if first_won == self.a_first else Outcome.B


class Tuner:
    """Model + library + log, rebuilt from the stored history on start."""

    def __init__(
        self,
        storage: Storage,
        model_kind: str = "gp",
        space: EQSpace | None = None,
        session_kind: str = "calibrate",
        seed: int = 0,
        renderer: Renderer | None = None,
    ) -> None:
        self.storage = storage
        self.library = Library(storage)
        self.space = space or EQSpace()
        self.renderer = renderer or Renderer()
        self.rng = np.random.default_rng(seed)
        self.model: PreferenceModel = make_model(
            model_kind, self.space, self.library.context_dim, seed=seed
        )
        history = storage.comparisons("human")
        # Comparisons made under another search space or context projection
        # cannot be replayed into this model; they stay in the log.
        usable = [
            c
            for c in history
            if c.a.shape == (self.space.dim,) and c.context.shape == (self.library.context_dim,)
        ]
        if usable:
            self.model.update(usable)
        self.session_id = storage.start_session(
            session_kind, model_kind, {"space": self.space.kind, "seed": seed}
        )

    @property
    def n_comparisons(self) -> int:
        """Comparisons the model has learned from, across all sessions."""
        return len(self.model.comparisons)

    def ask(self, track: StoredTrack) -> Question:
        """Next A/B question on ``track``."""
        context = self.library.context(track)
        a, b = self.model.suggest_pair(context)
        return Question(track, context, a, b, a_first=bool(self.rng.random() < 0.5))

    def render_question(self, question: Question) -> tuple[FloatArray, FloatArray]:
        """Loudness-matched excerpts in listening order."""
        first, second = question.presented
        return (
            self.renderer.excerpt(question.track, self.space.to_gains(first)),
            self.renderer.excerpt(question.track, self.space.to_gains(second)),
        )

    def answer(self, question: Question, answer: Answer) -> Comparison:
        """Record the listener's answer and update the model."""
        comparison = Comparison(question.a, question.b, question.context, question.outcome(answer))
        self._log(
            comparison,
            question.track,
            presented_first="a" if question.a_first else "b",
            source="ab",
            excerpt=True,
        )
        self.model.update([comparison])
        return comparison

    def thumb(self, track: StoredTrack, point: FloatArray, liked: bool) -> Comparison:
        """Turn a 👍/👎 during playback into a weak comparison against flat.

        👍 means "this EQ is better than no EQ", 👎 the opposite. Flat is the
        reference because it is stable and meaningful to the listener; the
        lower weight reflects that a thumb may be about the song, not the EQ.
        """
        comparison = Comparison(
            a=np.asarray(point, dtype=np.float64),
            b=self.space.flat,
            context=self.library.context(track),
            outcome=Outcome.A if liked else Outcome.B,
            weight=THUMB_WEIGHT,
        )
        self._log(comparison, track, presented_first="a", source="thumb", excerpt=False)
        self.model.update([comparison])
        return comparison

    def best_point(self, track: StoredTrack) -> FloatArray:
        """Predicted best unit-cube point for a track."""
        return self.model.best(self.library.context(track))

    def best_gains(self, track: StoredTrack) -> FloatArray:
        """Predicted best EQ gains for a track."""
        return self.space.to_gains(self.best_point(track))

    def close(self) -> None:
        """End the session in the log."""
        self.storage.end_session(self.session_id)

    def _log(
        self,
        comparison: Comparison,
        track: StoredTrack,
        *,
        presented_first: str,
        source: str,
        excerpt: bool,
    ) -> None:
        self.storage.add_comparison(
            self.session_id,
            ComparisonRecord(
                comparison=comparison,
                gains_a=self.space.to_gains(comparison.a),
                gains_b=self.space.to_gains(comparison.b),
                track_id=track.id,
                presented_first=presented_first,
                source=source,
                excerpt_start_s=track.analysis.excerpt_start_s if excerpt else None,
                excerpt_duration_s=track.analysis.excerpt_duration_s if excerpt else None,
            ),
        )
