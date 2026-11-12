"""Deterministic audio path: 10-band EQ, loudness normalization and true-peak limiting.

Everything a listener hears goes through :func:`render`, which guarantees that
any EQ inside the configured gain range comes out at the same integrated
loudness (ITU-R BS.1770) and below the true-peak ceiling. That guarantee is what
keeps "louder" from winning A/B comparisons.

Audio arrays are float64 with shape ``(n_samples,)`` (mono) or
``(n_samples, n_channels)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pyloudnorm
from numpy.typing import ArrayLike, NDArray
from scipy import signal

FloatArray = NDArray[np.float64]

DEFAULT_CENTER_FREQS_HZ: tuple[float, ...] = (
    31.0,
    62.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
    16000.0,
)


@dataclass(frozen=True)
class EQSpec:
    """Shape of the equalizer: band centers, filter Q and allowed gain range.

    The lowest band is a low shelf, the highest a high shelf, and the rest are
    peaking filters with a shared Q.
    """

    center_freqs_hz: tuple[float, ...] = DEFAULT_CENTER_FREQS_HZ
    peaking_q: float = 1.2
    shelf_slope: float = 1.0
    min_gain_db: float = -10.0
    max_gain_db: float = 10.0

    def __post_init__(self) -> None:
        if len(self.center_freqs_hz) < 2:
            raise ValueError("an EQ needs at least two bands (low and high shelf)")
        if list(self.center_freqs_hz) != sorted(self.center_freqs_hz):
            raise ValueError("center frequencies must be increasing")
        if self.min_gain_db >= self.max_gain_db:
            raise ValueError("min_gain_db must be below max_gain_db")
        if self.peaking_q <= 0 or self.shelf_slope <= 0:
            raise ValueError("Q and shelf slope must be positive")

    @property
    def n_bands(self) -> int:
        """Number of EQ bands (the dimension of a gain vector)."""
        return len(self.center_freqs_hz)

    def validate(self, gains_db: ArrayLike) -> FloatArray:
        """Return ``gains_db`` as an array, raising if its shape or range is wrong."""
        gains = np.asarray(gains_db, dtype=np.float64)
        if gains.shape != (self.n_bands,):
            raise ValueError(f"expected {self.n_bands} gains, got shape {gains.shape}")
        if not np.all(np.isfinite(gains)):
            raise ValueError("gains must be finite")
        if np.any(gains < self.min_gain_db) or np.any(gains > self.max_gain_db):
            raise ValueError(
                f"gains must lie in [{self.min_gain_db}, {self.max_gain_db}] dB, got {gains}"
            )
        return gains

    def clip(self, gains_db: ArrayLike) -> FloatArray:
        """Clip a gain vector into the allowed range."""
        gains = np.asarray(gains_db, dtype=np.float64)
        return np.clip(gains, self.min_gain_db, self.max_gain_db)


@dataclass(frozen=True)
class LoudnessTarget:
    """Loudness every rendering is matched to.

    The default of -23 LUFS (EBU R128) is deliberately low: it leaves headroom
    for +10 dB boosts so the limiter rarely has to act.
    """

    integrated_lufs: float = -23.0
    true_peak_ceiling_dbtp: float = -1.0
    tolerance_lu: float = 0.5


@dataclass(frozen=True)
class RenderResult:
    """Output of :func:`render` plus the measurements that certify it."""

    audio: FloatArray
    loudness_lufs: float
    true_peak_dbtp: float
    gain_db: float
    limiter_engaged: bool


# --- Filter design (RBJ Audio EQ Cookbook) -----------------------------------


def _peaking(freq: float, gain_db: float, q: float, sample_rate: float) -> FloatArray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b = [1.0 + alpha * a, -2.0 * cos_w0, 1.0 - alpha * a]
    den = [1.0 + alpha / a, -2.0 * cos_w0, 1.0 - alpha / a]
    return _normalized_section(b, den)


def _shelf(
    freq: float, gain_db: float, slope: float, sample_rate: float, *, high: bool
) -> FloatArray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sample_rate
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / 2.0 * np.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    two_sqrt_a_alpha = 2.0 * np.sqrt(a) * alpha
    sign = -1.0 if high else 1.0
    b = [
        a * ((a + 1.0) - sign * (a - 1.0) * cos_w0 + two_sqrt_a_alpha),
        sign * 2.0 * a * ((a - 1.0) - sign * (a + 1.0) * cos_w0),
        a * ((a + 1.0) - sign * (a - 1.0) * cos_w0 - two_sqrt_a_alpha),
    ]
    den = [
        (a + 1.0) + sign * (a - 1.0) * cos_w0 + two_sqrt_a_alpha,
        -sign * 2.0 * ((a - 1.0) + sign * (a + 1.0) * cos_w0),
        (a + 1.0) + sign * (a - 1.0) * cos_w0 - two_sqrt_a_alpha,
    ]
    return _normalized_section(b, den)


def _normalized_section(b: list[float], a: list[float]) -> FloatArray:
    return np.array([*b, *a], dtype=np.float64) / a[0]


def design_sos(gains_db: ArrayLike, sample_rate: float, spec: EQSpec | None = None) -> FloatArray:
    """Second-order sections, shape ``(n_bands, 6)``, implementing the EQ curve."""
    spec = spec or EQSpec()
    gains = spec.validate(gains_db)
    nyquist = sample_rate / 2.0
    if spec.center_freqs_hz[-1] >= 0.9 * nyquist:
        raise ValueError(
            f"sample rate {sample_rate} Hz is too low for a band at {spec.center_freqs_hz[-1]} Hz"
        )
    last = spec.n_bands - 1
    sections = []
    for i, (freq, gain) in enumerate(zip(spec.center_freqs_hz, gains, strict=True)):
        if i == 0:
            sections.append(_shelf(freq, gain, spec.shelf_slope, sample_rate, high=False))
        elif i == last:
            sections.append(_shelf(freq, gain, spec.shelf_slope, sample_rate, high=True))
        else:
            sections.append(_peaking(freq, gain, spec.peaking_q, sample_rate))
    return np.vstack(sections)


def frequency_response_db(
    gains_db: ArrayLike,
    freqs_hz: ArrayLike,
    sample_rate: float = 48_000.0,
    spec: EQSpec | None = None,
) -> FloatArray:
    """Magnitude response of the EQ, in dB, at the given frequencies."""
    sos = design_sos(gains_db, sample_rate, spec)
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    _, h = signal.sosfreqz(sos, worN=freqs, fs=sample_rate)
    magnitude = np.maximum(np.abs(h), 1e-12)
    return np.asarray(20.0 * np.log10(magnitude), dtype=np.float64)


def apply_eq(
    audio: ArrayLike, sample_rate: float, gains_db: ArrayLike, spec: EQSpec | None = None
) -> FloatArray:
    """Filter ``audio`` through the EQ. Zero gains return an exact copy."""
    x = _as_audio(audio)
    gains = (spec or EQSpec()).validate(gains_db)
    if not np.any(gains):
        return x.copy()
    sos = design_sos(gains, sample_rate, spec)
    return np.asarray(signal.sosfilt(sos, x, axis=0), dtype=np.float64)


# --- Measurement --------------------------------------------------------------


@lru_cache(maxsize=8)
def _meter(sample_rate: float) -> pyloudnorm.Meter:
    return pyloudnorm.Meter(sample_rate)


def integrated_loudness(audio: ArrayLike, sample_rate: float) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770-4, gated).

    Raises ``ValueError`` for silence or clips shorter than one 400 ms block.
    """
    x = _as_audio(audio)
    if x.shape[0] < int(0.4 * sample_rate):
        raise ValueError("loudness needs at least 400 ms of audio")
    loudness = float(_meter(sample_rate).integrated_loudness(x))
    if not np.isfinite(loudness):
        raise ValueError("cannot measure loudness of silence")
    return loudness


def true_peak_dbtp(audio: ArrayLike, sample_rate: float, oversample: int = 4) -> float:
    """Measure the true peak in dBTP by polyphase oversampling (BS.1770-4 Annex 2)."""
    del sample_rate  # the estimate is rate-independent; kept for a uniform signature
    peak = float(np.max(_oversampled_abs_peak(_as_audio(audio), oversample), initial=0.0))
    return float(20.0 * np.log10(max(peak, 1e-12)))


def _oversampled_abs_peak(x: FloatArray, oversample: int) -> FloatArray:
    """Per-sample peak magnitude across channels and inter-sample positions."""
    up = np.abs(signal.resample_poly(x, oversample, 1, axis=0))
    grouped = up.reshape(x.shape[0], oversample, -1)  # (samples, phases, channels)
    return np.asarray(grouped.max(axis=(1, 2)), dtype=np.float64)


# --- Limiting and rendering ---------------------------------------------------


def limit_true_peak(
    audio: ArrayLike,
    sample_rate: float,
    ceiling_dbtp: float = -1.0,
    lookahead_ms: float = 10.0,
    oversample: int = 4,
) -> FloatArray:
    """Lookahead limiter that keeps the true peak at or below ``ceiling_dbtp``.

    The gain curve is built so it never exceeds the gain each sample needs:
    a forward running minimum over the lookahead window followed by a moving
    average over the same window. Each averaged value only mixes minima whose
    windows contain the current sample, so the smoothed gain is still low
    enough there, while attack and release become ramps of ``lookahead_ms``.
    Linked across channels so the stereo image does not shift.
    """
    x = _as_audio(audio)
    ceiling = 10.0 ** (ceiling_dbtp / 20.0)
    peaks = _oversampled_abs_peak(x, oversample)
    required = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))
    if np.all(required >= 1.0):
        return x.copy()

    window = max(1, round(lookahead_ms * 1e-3 * sample_rate))
    forward_min = _forward_min(required, window)
    # Holding the first value before the start keeps the bound at n < window
    # (forward_min[0] covers those samples) and avoids a gain jump at onset.
    padded = np.concatenate([np.full(window - 1, forward_min[0]), forward_min])
    kernel = np.ones(window) / window
    gain = np.convolve(padded, kernel, mode="valid")
    gain = np.minimum(gain, required)  # guards against float rounding in the average
    return np.asarray(x * (gain[:, None] if x.ndim == 2 else gain), dtype=np.float64)


def _forward_min(values: FloatArray, window: int) -> FloatArray:
    """``out[n] = min(values[n : n + window])``, with the tail padded by the last value."""
    padded = np.concatenate([values, np.full(window - 1, values[-1])])
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.asarray(view.min(axis=1), dtype=np.float64)


def render(
    audio: ArrayLike,
    sample_rate: float,
    gains_db: ArrayLike,
    spec: EQSpec | None = None,
    target: LoudnessTarget | None = None,
    max_iterations: int = 6,
) -> RenderResult:
    """Apply the EQ, then match loudness to ``target`` without exceeding its true peak.

    Steps: EQ → static gain to the target loudness → (only if needed) limiter,
    re-measure, nudge the gain and repeat. The limiter is the last stage, so the
    true-peak ceiling always holds; the loop then brings loudness back within
    tolerance. Raises ``RuntimeError`` if the tolerance cannot be met, which
    would mean the target is too loud for the material.
    """
    target = target or LoudnessTarget()
    equalized = apply_eq(audio, sample_rate, gains_db, spec)
    measured = integrated_loudness(equalized, sample_rate)
    gain_db = target.integrated_lufs - measured

    # Gain changes inside the interpolation kernel can push the reconstructed
    # peak slightly past what the limiter computed, so it aims a bit lower.
    limiter_ceiling = target.true_peak_ceiling_dbtp - _LIMITER_MARGIN_DB

    limiter_engaged = False
    out, loudness = equalized, measured
    for _ in range(max_iterations):
        out = equalized * 10.0 ** (gain_db / 20.0)
        if true_peak_dbtp(out, sample_rate) > target.true_peak_ceiling_dbtp:
            limiter_engaged = True
            out = limit_true_peak(out, sample_rate, limiter_ceiling)
        loudness = integrated_loudness(out, sample_rate)
        error = target.integrated_lufs - loudness
        if abs(error) <= target.tolerance_lu / 2.0:
            break
        gain_db += error

    if abs(target.integrated_lufs - loudness) > target.tolerance_lu:
        raise RuntimeError(
            f"could not reach {target.integrated_lufs} LUFS within "
            f"{target.tolerance_lu} LU (got {loudness:.2f}); lower the target"
        )
    return RenderResult(
        audio=out,
        loudness_lufs=loudness,
        true_peak_dbtp=true_peak_dbtp(out, sample_rate),
        gain_db=gain_db,
        limiter_engaged=limiter_engaged,
    )


_LIMITER_MARGIN_DB = 0.3


def _as_audio(audio: ArrayLike) -> FloatArray:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim not in (1, 2) or x.shape[0] == 0:
        raise ValueError(f"audio must be (n_samples,) or (n_samples, n_channels), got {x.shape}")
    return x
