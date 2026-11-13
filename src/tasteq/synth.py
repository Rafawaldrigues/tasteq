"""Synthetic test signals.

The repository ships no music, so tests, CI and examples use these generators.
They cover the cases that stress loudness matching: broadband noise, sparse
tones (low crest factor) and percussive material (high crest factor).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tasteq.dsp import FloatArray


def pink_noise(duration_s: float, sample_rate: int, rng: np.random.Generator) -> FloatArray:
    """Noise with a 1/f power spectrum, scaled to a peak of 0.5."""
    n = int(duration_s * sample_rate)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectrum[1:] /= np.sqrt(freqs[1:])
    spectrum[0] = 0.0
    return _peak_normalize(np.fft.irfft(spectrum, n))


def tone_mix(
    duration_s: float,
    sample_rate: int,
    rng: np.random.Generator,
    n_tones: int = 4,
) -> FloatArray:
    """Sum of sine tones at random log-spaced frequencies between 40 Hz and 12 kHz."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    freqs = np.exp(rng.uniform(np.log(40.0), np.log(12_000.0), size=n_tones))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_tones)
    amplitudes = rng.uniform(0.2, 1.0, size=n_tones)
    tones = amplitudes[:, None] * np.sin(2.0 * np.pi * freqs[:, None] * t + phases[:, None])
    return _peak_normalize(tones.sum(axis=0))


def drum_loop(
    duration_s: float,
    sample_rate: int,
    rng: np.random.Generator,
    bpm: float = 120.0,
) -> FloatArray:
    """Kick, hi-hat and a bass line: impulsive, bass-heavy and bright at once."""
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    out = np.zeros(n)
    beat = int(60.0 / bpm * sample_rate)

    kick_t = np.arange(int(0.25 * sample_rate)) / sample_rate
    kick = np.sin(2.0 * np.pi * (50.0 * kick_t + 60.0 * (1.0 - np.exp(-30.0 * kick_t)) / 30.0))
    kick *= np.exp(-12.0 * kick_t)
    hat = rng.standard_normal(int(0.05 * sample_rate)) * np.exp(
        -80.0 * np.arange(int(0.05 * sample_rate)) / sample_rate
    )
    hat = np.diff(hat, prepend=0.0)  # crude high-pass so the hat sits in the top bands

    for start in range(0, n, beat):
        _mix_in(out, kick, start)
        _mix_in(out, 0.3 * hat, start + beat // 2)

    bass_freq = rng.choice([41.2, 55.0, 61.7, 73.4])
    out += 0.3 * np.sin(2.0 * np.pi * bass_freq * t)
    return _peak_normalize(out)


def stereo(mono: FloatArray, rng: np.random.Generator, width: float = 0.3) -> FloatArray:
    """Turn a mono signal into a decorrelated stereo pair."""
    side = width * _peak_normalize(rng.standard_normal(size=mono.shape[:1])) * np.abs(mono).mean()
    return np.stack([mono + side, mono - side], axis=1)


def _mix_in(out: FloatArray, sound: FloatArray, start: int) -> None:
    end = min(out.shape[0], start + sound.shape[0])
    if start < end:
        out[start:end] += sound[: end - start]


def _peak_normalize(x: FloatArray, peak: float = 0.5) -> FloatArray:
    return np.asarray(x * (peak / np.max(np.abs(x))), dtype=np.float64)


def write_demo_library(
    folder: Path, n_tracks: int = 8, duration_s: float = 30.0, seed: int = 0
) -> list[Path]:
    """Write a small library of synthetic "tracks" with different tonal balances.

    Lets anyone try calibration and the demo without copyrighted music. The
    tracks mix the generators above with different tilts so the analysis
    finds genuinely different contexts.
    """
    import soundfile as sf

    from tasteq.dsp import apply_eq

    rng = np.random.default_rng(seed)
    sample_rate = 44_100
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_tracks):
        bpm = float(rng.uniform(80, 150))
        mix = (
            rng.uniform(0.3, 1.0) * drum_loop(duration_s, sample_rate, rng, bpm=bpm)
            + rng.uniform(0.0, 0.6) * tone_mix(duration_s, sample_rate, rng, n_tones=6)
            + rng.uniform(0.0, 0.3) * pink_noise(duration_s, sample_rate, rng)
        )
        tilt = np.linspace(-1.0, 1.0, 10) * rng.uniform(-6.0, 6.0)
        mix = apply_eq(_peak_normalize(mix), sample_rate, tilt)
        audio = stereo(_peak_normalize(mix, 0.7), rng, width=0.2)
        path = folder / f"synthetic_{i:02d}_{int(bpm)}bpm.wav"
        sf.write(path, _peak_normalize(audio, 0.9), sample_rate)
        paths.append(path)
    return paths
