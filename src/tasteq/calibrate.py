"""Initial calibration: a few A/B questions on a spectrally diverse set of tracks."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from tasteq.analysis import diverse_subset
from tasteq.playback import Player
from tasteq.session import Answer, Question, Tuner
from tasteq.storage import StoredTrack

Prompt = Callable[[str], str]
ANSWER_KEYS: dict[str, Answer] = {"1": "first", "2": "second", "=": "tie"}


def calibration_tracks(tuner: Tuner, n_tracks: int = 10) -> list[StoredTrack]:
    """Choose the ``n_tracks`` tracks that best cover the library's range of sounds."""
    indices = diverse_subset(tuner.library.contexts, n_tracks)
    return [tuner.library.tracks[i] for i in indices]


def calibration_plan(tuner: Tuner, n_tracks: int = 10, per_track: int = 4) -> Iterator[StoredTrack]:
    """Track for each question, track by track (3-5 questions each keeps the ear fresh)."""
    for track in calibration_tracks(tuner, n_tracks):
        for _ in range(per_track):
            yield track


def ask_in_terminal(
    tuner: Tuner, question: Question, player: Player, prompt: Prompt
) -> Answer | None:
    """Play both versions, read the answer and record it; ``None`` means quit."""
    first, second = tuner.render_question(question)
    rate = tuner.renderer.sample_rate
    while True:
        for label, audio in (("1", first), ("2", second)):
            print(f"  ▶ version {label}")
            player.start(audio, rate)
            player.wait()
        reply = prompt("Prefer [1], [2], no difference [=], replay [r], quit [q]").strip()
        if reply in ANSWER_KEYS:
            tuner.answer(question, ANSWER_KEYS[reply])
            return ANSWER_KEYS[reply]
        if reply == "q":
            return None


def run_calibration(
    tuner: Tuner, player: Player, prompt: Prompt, n_tracks: int = 10, per_track: int = 4
) -> int:
    """Interactive calibration; returns the number of answered questions."""
    answered = 0
    current = None
    for track in calibration_plan(tuner, n_tracks, per_track):
        if track is not current:
            print(f"\n♪ {track.analysis.path}")
            current = track
        answer = ask_in_terminal(tuner, tuner.ask(track), player, prompt)
        if answer is None:
            break
        answered += 1
    return answered
