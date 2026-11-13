"""Track analysis: light spectral features, a representative excerpt and a context vector.

Every track is analyzed once (results are cached in the database). The
features describe *what kind of sound* a track is, so preferences can depend
on it: a bass-heavy electronic track and a sparse acoustic one may call for
different EQs.

The raw feature vector has 14 dimensions (10 relative band energies,
spectral centroid, rolloff, tempo, dynamics). Models get a compact version:
:class:`ContextProjector` standardizes it over the library and keeps the first
few principal components, scaled to ``[0, 1]``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from tasteq.dsp import DEFAULT_CENTER_FREQS_HZ, FloatArray, integrated_loudness

ANALYSIS_SAMPLE_RATE = 44_100
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
FEATURE_NAMES = (
    *(f"band_{int(f)}hz" for f in DEFAULT_CENTER_FREQS_HZ),
    "log_centroid",
    "log_rolloff",
    "tempo_bpm",
    "dynamic_range_db",
)


@dataclass(frozen=True)
class TrackAnalysis:
    """Everything tasteq needs to know about one track."""

    path: str
    fingerprint: str
    duration_s: float
    loudness_lufs: float
    excerpt_start_s: float
    excerpt_duration_s: float
    features: tuple[float, ...]

    def feature_vector(self) -> FloatArray:
        """Raw features as an array, in :data:`FEATURE_NAMES` order."""
        return np.array(self.features, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return asdict(self)


# --- Loading --------------------------------------------------------------------


def load_audio(path: Path, sample_rate: int = ANALYSIS_SAMPLE_RATE) -> FloatArray:
    """Read a file as float64 ``(n_samples, n_channels)`` at ``sample_rate``."""
    data, file_rate = sf.read(str(path), dtype="float64", always_2d=True)
    if file_rate != sample_rate:
        divisor = np.gcd(int(file_rate), sample_rate)
        data = signal.resample_poly(data, sample_rate // divisor, int(file_rate) // divisor, axis=0)
    return np.asarray(data, dtype=np.float64)


def fingerprint(path: Path) -> str:
    """Cheap content fingerprint: size plus a hash of the first and last MiB."""
    size = path.stat().st_size
    digest = hashlib.sha1(str(size).encode())
    with path.open("rb") as f:
        digest.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), 2)
            digest.update(f.read())
    return digest.hexdigest()


def find_audio_files(folder: Path) -> list[Path]:
    """Audio files under ``folder``, sorted for reproducibility."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


# --- Features -----------------------------------------------------------------------


def _band_edges(centers: tuple[float, ...], sample_rate: int) -> list[tuple[float, float]]:
    """Octave-band edges halfway (geometrically) between centers."""
    mids = [np.sqrt(a * b) for a, b in pairwise(centers)]
    lows = [20.0, *mids]
    highs = [*mids, sample_rate / 2.0]
    return list(zip(lows, highs, strict=True))


def band_energy_db(
    mono: FloatArray, sample_rate: int, centers: tuple[float, ...] = DEFAULT_CENTER_FREQS_HZ
) -> FloatArray:
    """Energy per EQ band in dB, relative to the mean over bands (spectral shape)."""
    freqs, psd = signal.welch(mono, fs=sample_rate, nperseg=8192)
    energies = []
    for low, high in _band_edges(centers, sample_rate):
        mask = (freqs >= low) & (freqs < high)
        energies.append(float(np.sum(psd[mask])) + 1e-20)
    levels = 10.0 * np.log10(np.array(energies))
    return np.asarray(levels - levels.mean(), dtype=np.float64)


def spectral_features(mono: FloatArray, sample_rate: int) -> tuple[float, float, float, float]:
    """Log centroid, log 85% rolloff, tempo and dynamic range of a mono signal."""
    import librosa

    y = mono.astype(np.float32)
    centroid = float(np.median(librosa.feature.spectral_centroid(y=y, sr=sample_rate)))
    rolloff = float(np.median(librosa.feature.spectral_rolloff(y=y, sr=sample_rate)))
    tempo = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sample_rate))[0])
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=y)[0] + 1e-10)
    active = rms_db[rms_db > np.max(rms_db) - 60.0]  # ignore digital silence
    dynamic_range = float(np.percentile(active, 95) - np.percentile(active, 10))
    return np.log10(centroid + 1.0), np.log10(rolloff + 1.0), tempo, dynamic_range


def _frame_band_energies(
    mono: FloatArray, sample_rate: int, frame: int, centers: tuple[float, ...]
) -> FloatArray:
    """Energy per EQ band for consecutive non-overlapping frames, ``(n_frames, n_bands)``."""
    n_frames = mono.shape[0] // frame
    frames = mono[: n_frames * frame].reshape(n_frames, frame) * np.hanning(frame)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    bands = [(freqs >= lo) & (freqs < hi) for lo, hi in _band_edges(centers, sample_rate)]
    return np.stack([power[:, mask].sum(axis=1) for mask in bands], axis=1) + 1e-20


def representative_excerpt(
    audio: FloatArray, sample_rate: int, duration_s: float = 12.0, hop_s: float = 1.0
) -> float:
    """Start time of the window that best represents the whole track.

    Picks the window whose spectral shape is closest to the track's overall
    shape, among windows that are not much quieter than the track (so intros,
    breakdowns and fade-outs are avoided). Simple and explainable rather than
    a chorus detector. Band energies are computed once per ``hop_s`` frame and
    summed over sliding windows, so the cost is linear in track length.
    """
    total = audio.shape[0] / sample_rate
    if total <= duration_s:
        return 0.0
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    frame = int(hop_s * sample_rate)
    energies = _frame_band_energies(mono, sample_rate, frame, DEFAULT_CENTER_FREQS_HZ)
    per_window = round(duration_s / hop_s)
    cumulative = np.vstack([np.zeros(energies.shape[1]), np.cumsum(energies, axis=0)])
    windows = cumulative[per_window:] - cumulative[:-per_window]  # (n_windows, n_bands)

    def shape(e: FloatArray) -> FloatArray:
        levels = 10.0 * np.log10(e)
        return np.asarray(levels - levels.mean(axis=-1, keepdims=True), dtype=np.float64)

    distances = np.linalg.norm(shape(windows) - shape(energies.sum(axis=0)), axis=1)
    level_db = 10.0 * np.log10(windows.sum(axis=1))
    distances[level_db < np.max(level_db) - 3.0] = np.inf
    return float(np.argmin(distances) * hop_s)


def analyze_track(path: Path, excerpt_duration_s: float = 12.0) -> TrackAnalysis:
    """Full analysis of one audio file."""
    audio = load_audio(path)
    sample_rate = ANALYSIS_SAMPLE_RATE
    mono = audio.mean(axis=1)
    bands = band_energy_db(mono, sample_rate)
    extra = spectral_features(mono, sample_rate)
    duration = audio.shape[0] / sample_rate
    return TrackAnalysis(
        path=str(path),
        fingerprint=fingerprint(path),
        duration_s=duration,
        loudness_lufs=integrated_loudness(audio, sample_rate),
        excerpt_start_s=representative_excerpt(audio, sample_rate, excerpt_duration_s),
        excerpt_duration_s=min(excerpt_duration_s, duration),
        features=tuple(float(v) for v in (*bands, *extra)),
    )


# --- Context vectors -------------------------------------------------------------


@dataclass(frozen=True)
class ContextProjector:
    """Standardize raw features over a library and keep the top principal components.

    The projection is fit once per library and stored, so the context of a
    track only changes if the user explicitly refits it. Comparisons store the
    context they were made with, so refitting never rewrites history.
    """

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    components: tuple[tuple[float, ...], ...]
    low: tuple[float, ...]
    high: tuple[float, ...]

    @property
    def dim(self) -> int:
        """Dimension of the context vectors produced."""
        return len(self.components)

    @classmethod
    def fit(cls, features: FloatArray, dim: int = 4) -> ContextProjector:
        """Fit on a ``(n_tracks, n_features)`` matrix."""
        x = np.atleast_2d(features)
        dim = max(1, min(dim, x.shape[0] - 1, x.shape[1]))
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-9] = 1.0
        z = (x - mean) / scale
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        components = vt[:dim]
        # Sign convention so the projection is deterministic across fits.
        components *= np.sign(components[np.arange(dim), np.argmax(np.abs(components), axis=1)])[
            :, None
        ]
        scores = z @ components.T
        return cls(
            mean=tuple(mean),
            scale=tuple(scale),
            components=tuple(tuple(row) for row in components),
            low=tuple(scores.min(axis=0)),
            high=tuple(scores.max(axis=0)),
        )

    def transform(self, features: FloatArray) -> FloatArray:
        """Map raw features to context vectors in ``[0, 1]^dim``."""
        z = (np.atleast_2d(features) - np.array(self.mean)) / np.array(self.scale)
        scores = z @ np.array(self.components).T
        span = np.maximum(np.array(self.high) - np.array(self.low), 1e-9)
        out = np.clip((scores - np.array(self.low)) / span, 0.0, 1.0)
        return np.asarray(out[0] if np.ndim(features) == 1 else out, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextProjector:
        """Inverse of :meth:`to_dict`."""
        return cls(
            mean=tuple(data["mean"]),
            scale=tuple(data["scale"]),
            components=tuple(tuple(row) for row in data["components"]),
            low=tuple(data["low"]),
            high=tuple(data["high"]),
        )


def diverse_subset(contexts: FloatArray, k: int) -> list[int]:
    """Pick ``k`` mutually distant points by greedy farthest-point sampling; return indices.

    Used to pick calibration tracks that cover the library's range of sounds.
    Starts from the point closest to the centroid, so the first track is typical.
    """
    x = np.atleast_2d(contexts)
    k = min(k, x.shape[0])
    if k == 0:
        return []
    chosen = [int(np.argmin(np.linalg.norm(x - x.mean(axis=0), axis=1)))]
    distance = np.linalg.norm(x - x[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(x - x[nxt], axis=1))
    return chosen
