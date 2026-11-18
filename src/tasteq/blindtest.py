"""Blind test of whether the listener actually prefers the learned EQ.

The hypotheses are fixed before any data is seen:

- **Primary:** the learned EQ is preferred over no EQ (flat) more than half
  the time.
- **Secondary:** the learned EQ is preferred over a *random* EQ of the same
  strength (same RMS gain). This controls for "any change sounds better" or
  for a listener who simply likes more EQ.

Each trial plays two loudness-matched versions of a track's excerpt in random
order without saying which is which. The EQs are fixed when the test starts
and answers never reach the model, so the test measures the model as it was.
"No difference" answers are counted and reported but left out of the
binomial (sign) test.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from tasteq.calibrate import calibration_tracks
from tasteq.dsp import FloatArray
from tasteq.preference import Outcome
from tasteq.session import Answer, Tuner
from tasteq.storage import Storage, StoredTrack

PAIRS: tuple[tuple[str, str], ...] = (("learned", "flat"), ("learned", "random"))
MIN_COMPARISONS = 20
ALPHA = 0.05


@dataclass(frozen=True)
class Trial:
    """One blind A/B trial between two named conditions."""

    track: StoredTrack
    conditions: tuple[str, str]
    gains: tuple[FloatArray, FloatArray]
    a_first: bool

    @property
    def presented(self) -> tuple[FloatArray, FloatArray]:
        """Gains in listening order: (first, second)."""
        return self.gains if self.a_first else (self.gains[1], self.gains[0])

    def outcome(self, answer: Answer) -> Outcome:
        """Translate an answer about listening order into an outcome for ``conditions``."""
        if answer == "tie":
            return Outcome.TIE
        return Outcome.A if (answer == "first") == self.a_first else Outcome.B


def matched_random(gains: FloatArray, rng: np.random.Generator, limit_db: float) -> FloatArray:
    """Random EQ with the same RMS gain as ``gains``, within ``±limit_db``."""
    target = float(np.sqrt(np.mean(np.square(gains))))
    direction = rng.standard_normal(gains.shape)
    direction -= direction.mean()  # no overall tilt in level, like a real EQ choice
    direction /= max(float(np.sqrt(np.mean(np.square(direction)))), 1e-12)
    return np.asarray(np.clip(target * direction, -limit_db, limit_db), dtype=np.float64)


def plan_trials(tuner: Tuner, n_trials: int, seed: int = 0) -> list[Trial]:
    """Trials alternating the two pairs, over a diverse rotation of tracks.

    Learned gains are computed now, so later learning (for example in the
    demo's other tab) cannot change what is being tested.
    """
    rng = np.random.default_rng(seed)
    tracks = itertools.cycle(calibration_tracks(tuner, len(tuner.library.tracks)))
    limit = tuner.space.spec.max_gain_db
    trials = []
    for i, track in zip(range(n_trials), tracks, strict=False):
        learned = tuner.best_gains(track)
        conditions = PAIRS[i % len(PAIRS)]
        other = (
            np.zeros_like(learned)
            if conditions[1] == "flat"
            else matched_random(learned, rng, limit)
        )
        # Which condition is "a" is also randomized, so the log has no fixed slot.
        if rng.random() < 0.5:
            trials.append(Trial(track, conditions, (learned, other), bool(rng.random() < 0.5)))
        else:
            swapped = (conditions[1], conditions[0])
            trials.append(Trial(track, swapped, (other, learned), bool(rng.random() < 0.5)))
    rng.shuffle(trials)
    return trials


@dataclass(frozen=True)
class PairResult:
    """Outcome of the blind test for one pair of conditions."""

    pair: tuple[str, str]
    wins: int
    losses: int
    ties: int
    p_value: float
    ci_low: float
    ci_high: float

    @property
    def n(self) -> int:
        """Decisive trials (ties excluded)."""
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        """Share of decisive trials won by the first condition of ``pair``."""
        return self.wins / self.n if self.n else float("nan")


def analyze(
    rows: list[dict[str, object]], pairs: Sequence[tuple[str, str]] = PAIRS
) -> list[PairResult]:
    """Run a one-sided exact binomial test per pair (is ``pair[0]`` preferred?).

    ``pairs`` defaults to this module's two pairs; other front ends (the web
    study adds learned vs the phase-1 EQ) pass their own.
    """
    results = []
    for pair in pairs:
        wins = losses = ties = 0
        for row in rows:
            conditions = (row["condition_a"], row["condition_b"])
            if set(conditions) != set(pair):
                continue
            outcome = Outcome(row["outcome"])
            if outcome is Outcome.TIE:
                ties += 1
                continue
            winner = conditions[0] if outcome is Outcome.A else conditions[1]
            wins += winner == pair[0]
            losses += winner != pair[0]
        if wins + losses:
            test = stats.binomtest(wins, wins + losses, 0.5, alternative="greater")
            # The test is one-sided, but a two-sided interval is easier to read.
            ci = stats.binomtest(wins, wins + losses).proportion_ci(method="exact")
            results.append(
                PairResult(pair, wins, losses, ties, float(test.pvalue), ci.low, ci.high)
            )
        else:
            results.append(PairResult(pair, 0, 0, ties, 1.0, 0.0, 1.0))
    return results


def format_results(results: list[PairResult]) -> str:
    """Markdown table plus a one-line verdict on the primary hypothesis."""
    lines = [
        "| comparison | preferred | not preferred | no difference | win rate (95% CI) | p |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        rate = f"{r.win_rate:.0%} ({r.ci_low:.0%} to {r.ci_high:.0%})" if r.n else "-"
        lines.append(
            f"| {r.pair[0]} vs {r.pair[1]} | {r.wins} | {r.losses} | {r.ties} | {rate} "
            f"| {r.p_value:.3f} |"
        )
    primary = results[0]
    if primary.n == 0:
        verdict = "No decisive answers yet."
    elif primary.p_value < ALPHA:
        verdict = (
            f"The learned EQ was preferred over flat more often than chance "
            f"(p = {primary.p_value:.3f}, one-sided exact binomial test)."
        )
    else:
        verdict = (
            f"Not enough evidence that the learned EQ beats flat "
            f"(p = {primary.p_value:.3f}); more trials or more training may change that."
        )
    return "\n".join([*lines, "", verdict])


class BlindTest:
    """A planned set of trials, answered one at a time and logged as they come."""

    def __init__(self, tuner: Tuner, n_trials: int = 20, seed: int = 0) -> None:
        self.tuner = tuner
        self.storage: Storage = tuner.storage
        self.n_comparisons = tuner.n_comparisons
        self.trials = plan_trials(tuner, n_trials, seed)
        self.position = 0
        self.session_id = self.storage.start_session(
            "blindtest", tuner.model.name, {"n_trials": n_trials, "seed": seed}
        )

    @property
    def done(self) -> bool:
        """Whether every trial has been answered."""
        return self.position >= len(self.trials)

    @property
    def trial(self) -> Trial:
        """The trial waiting for an answer."""
        return self.trials[self.position]

    def render(self) -> tuple[FloatArray, FloatArray]:
        """Loudness-matched excerpts of the current trial, in listening order."""
        first, second = self.trial.presented
        renderer = self.tuner.renderer
        return renderer.excerpt(self.trial.track, first), renderer.excerpt(self.trial.track, second)

    def answer(self, answer: Answer) -> None:
        """Log the answer to the current trial and move on."""
        trial = self.trial
        self.storage.add_blind_trial(
            self.session_id,
            trial.track.id,
            trial.conditions,
            trial.gains,
            presented_first="a" if trial.a_first else "b",
            outcome=trial.outcome(answer),
            n_comparisons=self.n_comparisons,
        )
        self.position += 1
        if self.done:
            self.storage.end_session(self.session_id)

    def results(self) -> list[PairResult]:
        """Results over this test's trials only."""
        rows = [r for r in self.storage.blind_trials() if r["session_id"] == self.session_id]
        return analyze(rows)
