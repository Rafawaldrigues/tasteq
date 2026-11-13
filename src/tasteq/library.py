"""A user's music library: analyzed tracks, their contexts and loudness-matched rendering."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tasteq.analysis import (
    ANALYSIS_SAMPLE_RATE,
    ContextProjector,
    analyze_track,
    find_audio_files,
    fingerprint,
    load_audio,
)
from tasteq.dsp import FloatArray, LoudnessTarget, render
from tasteq.storage import Storage, StoredTrack

CONTEXT_META_KEY = "context_projector"


@dataclass(frozen=True)
class AnalyzeReport:
    """What :func:`analyze_folder` did."""

    analyzed: int
    cached: int
    failed: list[str]


def analyze_folder(
    folder: Path,
    storage: Storage,
    on_progress: Callable[[Path, str], None] | None = None,
) -> AnalyzeReport:
    """Analyze every audio file under ``folder``, skipping unchanged ones.

    Refits the context projection afterwards, since it depends on the whole
    library. Stored comparisons keep the context they were made with.
    """
    analyzed, cached, failed = 0, 0, []
    for path in find_audio_files(folder):
        key = str(path.resolve())
        if storage.fingerprint_of(key) == fingerprint(path):
            cached += 1
            status = "cached"
        else:
            try:
                analysis = analyze_track(Path(key))
            except Exception as error:  # report and keep going
                failed.append(f"{path}: {error}")
                status = "failed"
            else:
                storage.upsert_track(analysis)
                analyzed += 1
                status = "analyzed"
        if on_progress:
            on_progress(path, status)

    tracks = storage.tracks()
    if tracks:
        features = np.array([t.analysis.feature_vector() for t in tracks])
        storage.set_meta(CONTEXT_META_KEY, ContextProjector.fit(features).to_dict())
    return AnalyzeReport(analyzed, cached, failed)


class Library:
    """Analyzed tracks plus their context vectors, loaded from storage."""

    def __init__(self, storage: Storage) -> None:
        self.tracks = storage.tracks()
        data = storage.get_meta(CONTEXT_META_KEY)
        if not self.tracks or data is None:
            raise LookupError("no analyzed tracks; run `tasteq analyze <folder>` first")
        self.projector = ContextProjector.from_dict(data)
        features = np.array([t.analysis.feature_vector() for t in self.tracks])
        self._contexts = self.projector.transform(features)

    @property
    def context_dim(self) -> int:
        """Dimension of the context vectors."""
        return self.projector.dim

    def context(self, track: StoredTrack) -> FloatArray:
        """Context vector of a track."""
        index = next(i for i, t in enumerate(self.tracks) if t.id == track.id)
        return np.asarray(self._contexts[index], dtype=np.float64)

    @property
    def contexts(self) -> FloatArray:
        """All context vectors, ``(n_tracks, context_dim)``."""
        return np.asarray(self._contexts, dtype=np.float64)


class Renderer:
    """Loads tracks and renders loudness-matched versions, caching decoded audio."""

    def __init__(
        self,
        target: LoudnessTarget | None = None,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
        max_cached: int = 4,
    ) -> None:
        self.target = target or LoudnessTarget()
        self.sample_rate = sample_rate
        self.max_cached = max_cached
        self._audio: OrderedDict[str, FloatArray] = OrderedDict()

    def audio(self, track: StoredTrack) -> FloatArray:
        """Return the decoded audio of the whole track."""
        path = track.analysis.path
        if path not in self._audio:
            self._audio[path] = load_audio(Path(path), self.sample_rate)
            while len(self._audio) > self.max_cached:
                self._audio.popitem(last=False)
        self._audio.move_to_end(path)
        return self._audio[path]

    def excerpt(self, track: StoredTrack, gains_db: FloatArray) -> FloatArray:
        """Render the representative excerpt with ``gains_db``, loudness-matched."""
        start = int(track.analysis.excerpt_start_s * self.sample_rate)
        length = int(track.analysis.excerpt_duration_s * self.sample_rate)
        clip = self.audio(track)[start : start + length]
        return render(clip, self.sample_rate, gains_db, target=self.target).audio

    def full(self, track: StoredTrack, gains_db: FloatArray) -> FloatArray:
        """Render the whole track with ``gains_db``, loudness-matched."""
        return render(self.audio(track), self.sample_rate, gains_db, target=self.target).audio
