"""Clean musical loops for phase 1 of the web study.

Phase 1 asks about general taste (more or less bass, presence, air) before any
real song is heard. The library's test signals (:mod:`tasteq.synth`) are made
to stress loudness matching and are full of broadband noise, which sounds
like hiss to a listener. These loops are built from pitched instruments
instead: a sine bass, an FM electric piano, plucked strings, a soft pad, a
voice-like formant lead and bells, over a short drum kit. Each loop leans on
a different part of the spectrum so the answers cover the whole EQ.

Everything is synthesized here, so the sounds carry no license.
Oscillators are band-limited to keep aliasing (another source of hiss) out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from tasteq.dsp import FloatArray

SAMPLE_RATE = 44_100
_NYQUIST_LIMIT = 16_000.0  # highest partial any oscillator may produce


def _midi(note: float) -> float:
    return float(440.0 * 2.0 ** ((note - 69.0) / 12.0))


def _env(n: int, attack_s: float, decay_s: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Linear attack, then exponential decay with time constant ``decay_s``."""
    t = np.arange(n) / sr
    attack = np.clip(t / max(attack_s, 1e-4), 0.0, 1.0)
    return np.asarray(attack * np.exp(-np.maximum(t - attack_s, 0.0) / decay_s))


def _additive(
    freq: float, n: int, amps: Callable[[int], float], sr: int = SAMPLE_RATE
) -> FloatArray:
    """Sum of harmonics ``k * freq`` below the band limit, weighted by ``amps(k)``."""
    t = np.arange(n) / sr
    out = np.zeros(n)
    k = 1
    while k * freq < _NYQUIST_LIMIT and k <= 60:
        a = amps(k)
        if a:
            out += a * np.sin(2.0 * np.pi * k * freq * t)
        k += 1
    return out


# ---------------------------------------------------------------- instruments


def kick(sr: int = SAMPLE_RATE) -> FloatArray:
    """Return a pitch-swept sine kick with a soft click."""
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    phase = 2.0 * np.pi * (48.0 * t + 90.0 * (1.0 - np.exp(-28.0 * t)) / 28.0)
    click = np.sin(2.0 * np.pi * 1800.0 * t) * np.exp(-t / 0.004) * 0.15
    return np.asarray(np.sin(phase) * np.exp(-t / 0.16) + click)


def snare(rng: np.random.Generator, sr: int = SAMPLE_RATE) -> FloatArray:
    """Return a tuned snare: a tonal body and a very short wire burst."""
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    body = (np.sin(2.0 * np.pi * 185.0 * t) + 0.5 * np.sin(2.0 * np.pi * 330.0 * t)) * np.exp(
        -t / 0.05
    )
    # A short, band-limited burst for the wires; it dies out in ~60 ms.
    b, a = signal.butter(2, [1500.0, 7000.0], btype="band", fs=sr)
    wires = signal.lfilter(b, a, rng.standard_normal(n)) * np.exp(-t / 0.035)
    return np.asarray(0.8 * body + 0.6 * wires / np.max(np.abs(wires)))


def hat(rng: np.random.Generator, sr: int = SAMPLE_RATE) -> FloatArray:
    """Return a closed hat built from metallic partials, not noise."""
    n = int(0.06 * sr)
    t = np.arange(n) / sr
    # Metallic partials rather than noise: a cluster of inharmonic square-ish tones.
    freqs = np.array([3150.0, 4410.0, 5230.0, 6370.0, 7520.0, 8840.0, 10130.0, 11870.0, 13420.0])
    weights = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.8])
    phases = rng.uniform(0, 2 * np.pi, (len(freqs), 1))
    tones = (weights[:, None] * np.sin(2.0 * np.pi * freqs[:, None] * t + phases)).sum(0)
    return np.asarray(tones / weights.sum() * np.exp(-t / 0.018))


def bass(freq: float, n: int, bright: float = 0.25) -> FloatArray:
    """Return a sine bass note with a touch of upper harmonics."""
    tone = _additive(freq, n, lambda k: {1: 1.0, 2: bright, 3: bright * 0.4}.get(k, 0.0))
    return tone * _env(n, 0.005, 0.5)


def epiano(freq: float, n: int, sr: int = SAMPLE_RATE) -> FloatArray:
    """Two-operator FM: a bell-ish attack that mellows into a sine."""
    t = np.arange(n) / sr
    index = 2.2 * np.exp(-t / 0.25) + 0.3
    mod = index * np.sin(2.0 * np.pi * freq * t)
    tone = np.sin(2.0 * np.pi * freq * t + mod) + 0.25 * np.sin(2.0 * np.pi * 2 * freq * t)
    tone += 0.06 * np.exp(-t / 0.08) * np.sin(2.0 * np.pi * 7 * freq * t)  # tine
    return np.asarray(tone * _env(n, 0.003, 0.9))


def bell(freq: float, n: int, sr: int = SAMPLE_RATE) -> FloatArray:
    """Return an FM bell note."""
    t = np.arange(n) / sr
    mod = 3.0 * np.exp(-t / 0.4) * np.sin(2.0 * np.pi * 3.5 * freq * t)
    return np.asarray(np.sin(2.0 * np.pi * freq * t + mod) * _env(n, 0.002, 0.7))


def pluck(
    freq: float, n: int, rng: np.random.Generator, damping: float = 0.996, sr: int = SAMPLE_RATE
) -> FloatArray:
    """Karplus-Strong string, vectorized as one feedback filter."""
    period = max(2, round(sr / freq))
    excitation = np.zeros(n)
    burst = rng.uniform(-1.0, 1.0, period)
    b, a = signal.butter(1, min(0.9, 4000.0 / (sr / 2)))
    excitation[:period] = signal.lfilter(b, a, burst)
    feedback = np.zeros(period + 2)
    feedback[0] = 1.0
    feedback[period] = -0.5 * damping
    feedback[period + 1] = -0.5 * damping
    return np.asarray(signal.lfilter([1.0], feedback, excitation))


def pad(freqs: list[float], n: int, sr: int = SAMPLE_RATE) -> FloatArray:
    """Soft detuned saws, low-passed, with a slow swell."""
    out = np.zeros(n)
    for f in freqs:
        for detune in (0.996, 1.0, 1.004):
            out += _additive(f * detune, n, lambda k: 1.0 / k**1.6)
    b, a = signal.butter(2, 2200.0, fs=sr)
    return np.asarray(signal.lfilter(b, a, out) * _env(n, 0.6, 3.0))


def voice(freq: float, n: int, vowel: str = "ah", sr: int = SAMPLE_RATE) -> FloatArray:
    """Return a sung-vowel note: a band-limited pulse through three formant filters."""
    t = np.arange(n) / sr
    vibrato = 1.0 + 0.006 * np.sin(2.0 * np.pi * 5.2 * t) * np.clip(t / 0.4, 0.0, 1.0)
    phase = 2.0 * np.pi * freq * np.cumsum(vibrato) / sr
    source = np.zeros(n)
    k = 1
    while k * freq < _NYQUIST_LIMIT and k <= 60:
        source += np.sin(k * phase) / k
        k += 1
    formants = {
        "ah": [(730, 1.0), (1090, 0.5), (2440, 0.25)],
        "oo": [(300, 1.0), (870, 0.35), (2240, 0.12)],
    }
    out = np.zeros(n)
    for f, gain in formants[vowel]:
        b, a = signal.iirpeak(f, 6.0, fs=sr)
        out += gain * signal.lfilter(b, a, source)
    return np.asarray(out * _env(n, 0.08, 1.4))


# ---------------------------------------------------------------- arrangement


@dataclass(frozen=True)
class Loop:
    """One phase-1 loop: tempo, key and which instruments carry it."""

    name: str
    bpm: float
    root: int  # MIDI note of the key's tonic
    drums: str  # "", "kick", "kit" or "soft"
    parts: tuple[str, ...]
    tilt_db: float  # gentle spectral tilt, so the loops differ in balance


LOOPS = (
    Loop("deep-groove", 92, 38, "kick", ("bass", "epiano"), -2.0),
    Loop("bright-pluck", 124, 45, "soft", ("pluck-arp", "bass"), 2.5),
    Loop("warm-keys", 84, 41, "", ("epiano", "bass"), -1.0),
    Loop("vocal-pad", 76, 43, "", ("voice", "pad"), 0.0),
    Loop("full-band", 110, 40, "kit", ("bass", "epiano", "pluck-arp"), 0.5),
    Loop("acoustic", 96, 43, "", ("strum", "bass-soft"), 1.0),
    Loop("synth-lead", 118, 42, "kit", ("lead", "pad", "bass"), 1.5),
    Loop("sub-sparkle", 100, 36, "kick", ("sub", "bells"), 0.0),
)

# I-V-vi-IV and i-VI-III-VII, as semitone offsets of chord roots and qualities.
_PROGRESSIONS = (
    [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")],
    [(0, "min"), (8, "maj"), (3, "maj"), (10, "maj")],
)
_TRIADS = {"maj": (0, 4, 7), "min": (0, 3, 7)}


def _place(out: FloatArray, sound: FloatArray, start: int, gain: float = 1.0) -> None:
    end = min(out.shape[0], start + sound.shape[0])
    if start < end:
        out[start:end] += gain * sound[: end - start]


def _render_loop(loop: Loop, duration_s: float, rng: np.random.Generator) -> FloatArray:
    sr = SAMPLE_RATE
    beat = 60.0 / loop.bpm
    bar = 4 * beat
    n_bars = max(1, round(duration_s / bar))
    n = int(n_bars * bar * sr)  # whole bars, so the excerpt loops cleanly
    out = np.zeros(n)
    prog = _PROGRESSIONS[len(loop.name) % 2]

    for i in range(n_bars):
        bar_start = int(i * bar * sr)
        offset, quality = prog[i % len(prog)]
        root = loop.root + offset
        chord = [root + 12 + iv for iv in _TRIADS[quality]]
        bar_len = int(bar * sr)
        beat_len = int(beat * sr)

        # Drums
        for b in range(4):
            at = bar_start + b * beat_len
            if loop.drums in ("kick", "kit", "soft") and (b % 2 == 0 or loop.drums == "kick"):
                _place(out, kick(), at, 0.9 if loop.drums != "soft" else 0.55)
            if loop.drums == "kit" and b % 2 == 1:
                _place(out, snare(rng), at, 0.45)
            if loop.drums in ("kit", "soft"):
                _place(out, hat(rng), at + beat_len // 2, 0.12)
            else:
                # Every loop keeps a little air on the off-beats, so the top
                # bands have something to change.
                _place(out, hat(rng), at + beat_len // 2, 0.1)

        for part in loop.parts:
            if part in ("bass", "bass-soft", "sub"):
                f = _midi(root - 12)
                bright = {"bass": 0.3, "bass-soft": 0.15, "sub": 0.0}[part]
                for b in (0, 2) if part != "sub" else (0,):
                    length = 2 * beat_len if part != "sub" else bar_len
                    _place(out, bass(f, length, bright), bar_start + b * beat_len, 0.55)
            elif part == "epiano":
                for b in (0, 2):
                    for note in chord:
                        _place(
                            out, epiano(_midi(note), 2 * beat_len), bar_start + b * beat_len, 0.14
                        )
            elif part == "pluck-arp":
                steps = 8
                for s in range(steps):
                    note = chord[s % 3] + 12 * (s // 3 % 2)
                    _place(
                        out, pluck(_midi(note), beat_len, rng), bar_start + s * beat_len // 2, 0.22
                    )
            elif part == "strum":
                for b in range(4):
                    for j, note in enumerate([root, *chord]):
                        _place(
                            out,
                            pluck(_midi(note), 2 * beat_len, rng, 0.997),
                            bar_start + b * beat_len + j * int(0.018 * sr),
                            0.16,
                        )
            elif part == "pad":
                _place(
                    out, pad([_midi(x) for x in chord], bar_len + int(0.5 * sr)), bar_start, 0.07
                )
            elif part == "voice":
                melody = [chord[2], chord[1], chord[0] + 12, chord[1]]
                for b, note in enumerate(melody):
                    _place(
                        out,
                        voice(_midi(note), beat_len, "ah" if b % 2 == 0 else "oo"),
                        bar_start + b * beat_len,
                        0.35,
                    )
            elif part == "lead":
                melody = [chord[0] + 12, chord[2], chord[1] + 12, chord[2] + 12]
                for b, note in enumerate(melody):
                    tone = _additive(_midi(note), beat_len, lambda k: 1.0 / k**1.3) * _env(
                        beat_len, 0.01, 0.6
                    )
                    _place(out, tone, bar_start + b * beat_len, 0.12)
            elif part == "bells":
                for s in range(4):
                    note = chord[(s * 2) % 3] + 24
                    _place(out, bell(_midi(note), 2 * beat_len), bar_start + s * beat_len, 0.12)

    # A short room, so dry synthesis does not sound clinical. The impulse is
    # low-passed and quiet; the tail is smooth, not a noise floor.
    ir_n = int(0.9 * sr)
    t = np.arange(ir_n) / sr
    b, a = signal.butter(2, 3500.0, fs=sr)
    ir = signal.lfilter(b, a, rng.standard_normal(ir_n)) * np.exp(-t / 0.22)
    ir /= np.sqrt(np.sum(ir**2))
    wet = signal.fftconvolve(np.concatenate([out, out]), ir)[n : 2 * n]  # circular, loop-safe
    mono = out + 0.12 * wet

    if loop.tilt_db:
        from tasteq.dsp import apply_eq

        mono = apply_eq(mono, sr, np.linspace(-1.0, 1.0, 10) * loop.tilt_db)

    # Gentle stereo: delay the reverb a touch differently per side.
    left = mono
    right = np.roll(out, int(0.0004 * sr)) + 0.12 * np.roll(wet, int(0.011 * sr))
    if loop.tilt_db:
        from tasteq.dsp import apply_eq

        right = apply_eq(right, sr, np.linspace(-1.0, 1.0, 10) * loop.tilt_db)
    stereo = np.stack([left, right], axis=1)
    return np.asarray(0.89 * stereo / np.max(np.abs(stereo)))


def write_screening_loops(
    folder: Path, n_loops: int = 8, duration_s: float = 30.0, seed: int = 0
) -> list[Path]:
    """Write ``n_loops`` phase-1 loops as ``loop_XX_<name>.wav`` and return their paths."""
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_loops):
        loop = LOOPS[i % len(LOOPS)]
        rng = np.random.default_rng([seed, i])
        audio = _render_loop(loop, duration_s, rng)
        path = folder / f"loop_{i:02d}_{loop.name}.wav"
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_24")
        paths.append(path)
    return paths
