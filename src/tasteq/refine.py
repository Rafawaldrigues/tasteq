"""Per-track refinement: pick among several EQ versions of the busiest moments of a song.

After the general calibration the model has a prior for the listener. Refining
a favourite track then works like this:

1. Pick a few busy, mutually different moments of the track
   (:func:`tasteq.analysis.activity_moments`).
2. Build ``n_versions`` candidate EQs: the model's best guess plus samples of
   what could plausibly be the best, spread apart so the versions are audibly
   different (farthest-point sampling in the unit cube).
3. Run a king-of-the-hill tournament: the current favourite meets each
   challenger in a loudness-matched blind A/B, rotating through the moments.
   "No difference" keeps the favourite.

Every answer is a normal logged comparison with the context of the track, so
the model learns from refinement exactly as it does from calibration.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from tasteq.calibrate import Prompt, ask_in_terminal
from tasteq.dsp import FloatArray
from tasteq.playback import Player
from tasteq.preference import Outcome
from tasteq.session import Tuner
from tasteq.storage import StoredTrack


def candidate_points(tuner: Tuner, track: StoredTrack, n_versions: int) -> FloatArray:
    """``n_versions`` distinct unit-cube points to compare, best guess first.

    Shape ``(n_versions, dim)``. Before the model has learned anything the
    plausible optima are all flat, so the candidates are random points
    around it instead; either way they are chosen to be far from each other.
    """
    if n_versions < 2:
        raise ValueError("need at least two versions to compare")
    best = tuner.best_point(track)
    pool = tuner.model.plausible_best(tuner.library.context(track), max(64, 8 * n_versions))
    if float(np.max(np.ptp(pool, axis=0))) < 0.05:  # cold model: nothing known about the optimum
        jitter = tuner.rng.normal(0.0, 0.2, size=(len(pool), best.shape[0]))
        pool = np.clip(best + jitter, 0.0, 1.0)
    chosen = [best]
    distance = np.linalg.norm(pool - best, axis=1)
    while len(chosen) < n_versions:
        nxt = pool[int(np.argmax(distance))]
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(pool - nxt, axis=1))
    return np.stack(chosen)


@dataclass(frozen=True)
class RefinementResult:
    """Outcome of refining one track."""

    track: StoredTrack
    winner: FloatArray
    answered: int
    quit_early: bool


def run_refinement_for_track(
    tuner: Tuner,
    track: StoredTrack,
    player: Player,
    prompt: Prompt,
    n_versions: int = 6,
    n_moments: int = 3,
) -> RefinementResult:
    """Tournament among ``n_versions`` EQs on ``track``; the winner is the new favourite."""
    versions = candidate_points(tuner, track, n_versions)
    moments: list[float | None] = [*tuner.renderer.moments(track, n_moments)] or [None]
    champion = versions[0]
    answered = 0
    for i, challenger in enumerate(versions[1:]):
        question = tuner.ask_between(track, champion, challenger, moments[i % len(moments)])
        answer = ask_in_terminal(tuner, question, player, prompt)
        if answer is None:
            return RefinementResult(track, champion, answered, quit_early=True)
        answered += 1
        if question.outcome(answer) is Outcome.B:
            champion = challenger
    return RefinementResult(track, champion, answered, quit_early=False)


def run_refinement(
    tuner: Tuner,
    player: Player,
    prompt: Prompt,
    tracks: list[StoredTrack],
    n_versions: int = 6,
    n_moments: int = 3,
) -> Iterator[RefinementResult]:
    """Refine each track in turn, stopping after the first one the listener quits."""
    for track in tracks:
        print(f"\n♪ {track.analysis.path}")
        result = run_refinement_for_track(tuner, track, player, prompt, n_versions, n_moments)
        yield result
        if result.quit_early:
            break
