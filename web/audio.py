"""Excerpt loading shared by all participants, and encoding for the browser.

:class:`tasteq.library.Renderer` decodes whole songs and caches them per
renderer, which is right for one listener but would hold hundreds of MB per
participant on a server. Here every participant's renderer reads only the
excerpt it needs from one shared, thread-safe cache of short clips.
"""

from __future__ import annotations

import io
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from tasteq.analysis import ANALYSIS_SAMPLE_RATE
from tasteq.dsp import FloatArray, LoudnessTarget, render
from tasteq.library import Renderer
from tasteq.storage import StoredTrack

FORMATS = {
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
}


class ClipCache:
    """Least-recently-used cache of decoded excerpts, keyed by file and time span."""

    def __init__(
        self,
        resolve: Callable[[str], Path] = Path,
        sample_rate: int = ANALYSIS_SAMPLE_RATE,
        max_clips: int = 96,
    ) -> None:
        self.resolve = resolve
        self.sample_rate = sample_rate
        self.max_clips = max_clips
        self._clips: OrderedDict[tuple[str, float, float], FloatArray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str, start_s: float, duration_s: float) -> FloatArray:
        """Return decoded audio ``(n_samples, n_channels)`` of ``path`` from ``start_s``."""
        key = (path, round(start_s, 3), round(duration_s, 3))
        with self._lock:
            if key in self._clips:
                self._clips.move_to_end(key)
                return self._clips[key]
        clip = self._load(self.resolve(path), start_s, duration_s)
        with self._lock:
            self._clips[key] = clip
            while len(self._clips) > self.max_clips:
                self._clips.popitem(last=False)
        return clip

    def _load(self, path: Path, start_s: float, duration_s: float) -> FloatArray:
        rate = sf.info(str(path)).samplerate
        data, rate = sf.read(
            str(path),
            start=int(start_s * rate),
            frames=int(duration_s * rate),
            dtype="float64",
            always_2d=True,
        )
        if rate != self.sample_rate:
            divisor = np.gcd(int(rate), self.sample_rate)
            data = signal.resample_poly(data, self.sample_rate // divisor, rate // divisor, axis=0)
        data = np.asarray(data, dtype=np.float64)
        data.setflags(write=False)  # shared between threads
        return data


class ClipRenderer(Renderer):
    """A :class:`Renderer` that reads excerpts through a shared :class:`ClipCache`."""

    def __init__(self, clips: ClipCache, target: LoudnessTarget | None = None) -> None:
        super().__init__(target=target, sample_rate=clips.sample_rate, max_cached=0)
        self.clips = clips

    def excerpt(
        self, track: StoredTrack, gains_db: FloatArray, start_s: float | None = None
    ) -> FloatArray:
        """Loudness-matched excerpt with ``gains_db`` (the track's own excerpt by default)."""
        if start_s is None:
            start_s = track.analysis.excerpt_start_s
        clip = self.clips.get(track.analysis.path, start_s, track.analysis.excerpt_duration_s)
        return render(clip, self.sample_rate, gains_db, target=self.target).audio

    def audio(self, track: StoredTrack) -> FloatArray:
        """Whole tracks are never decoded on the server."""
        raise NotImplementedError("the web study only renders excerpts")


def encode(audio: FloatArray, sample_rate: int, fmt: str = "flac") -> tuple[bytes, str]:
    """Encode float audio as 16-bit (or MP3) bytes; return ``(data, media_type)``.

    Samples are only clipped, never rescaled: the loudness matching done by
    the renderer must reach the listener unchanged.
    """
    container, subtype, media_type = FORMATS[fmt]
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1.0, 1.0), sample_rate, format=container, subtype=subtype)
    return buffer.getvalue(), media_type
