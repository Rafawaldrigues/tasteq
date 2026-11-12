from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tasteq import synth
from tasteq.dsp import (
    DEFAULT_CENTER_FREQS_HZ,
    EQSpec,
    FloatArray,
    LoudnessTarget,
    apply_eq,
    design_sos,
    frequency_response_db,
    integrated_loudness,
    limit_true_peak,
    render,
    true_peak_dbtp,
)

SR = 44_100
SPEC = EQSpec()
TARGET = LoudnessTarget()

SignalFactory = Callable[[np.random.Generator], FloatArray]
SIGNALS: dict[str, SignalFactory] = {
    "pink": lambda rng: synth.pink_noise(2.0, SR, rng),
    "tones": lambda rng: synth.tone_mix(2.0, SR, rng),
    "drums": lambda rng: synth.drum_loop(2.0, SR, rng),
    "stereo_drums": lambda rng: synth.stereo(synth.drum_loop(2.0, SR, rng), rng),
}

gain_vectors = st.lists(
    st.floats(SPEC.min_gain_db, SPEC.max_gain_db, allow_nan=False),
    min_size=SPEC.n_bands,
    max_size=SPEC.n_bands,
).map(np.array)


def one_band(index: int, gain_db: float) -> FloatArray:
    gains = np.zeros(SPEC.n_bands)
    gains[index] = gain_db
    return gains


# --- EQSpec -------------------------------------------------------------------


def test_spec_rejects_wrong_shape_and_range() -> None:
    with pytest.raises(ValueError, match="expected 10 gains"):
        SPEC.validate(np.zeros(9))
    with pytest.raises(ValueError, match="must lie in"):
        SPEC.validate(one_band(0, 10.5))
    with pytest.raises(ValueError, match="finite"):
        SPEC.validate(one_band(3, np.nan))


def test_spec_rejects_inconsistent_configuration() -> None:
    with pytest.raises(ValueError, match="increasing"):
        EQSpec(center_freqs_hz=(1000.0, 100.0))
    with pytest.raises(ValueError, match="below"):
        EQSpec(min_gain_db=5.0, max_gain_db=5.0)


def test_clip_keeps_gains_in_range() -> None:
    clipped = SPEC.clip(np.linspace(-20, 20, SPEC.n_bands))
    assert clipped.min() == SPEC.min_gain_db
    assert clipped.max() == SPEC.max_gain_db


def test_sample_rate_too_low_for_top_band() -> None:
    with pytest.raises(ValueError, match="too low"):
        design_sos(np.zeros(SPEC.n_bands), 22_050)


# --- Filters ------------------------------------------------------------------


@pytest.mark.parametrize("index", range(1, len(DEFAULT_CENTER_FREQS_HZ) - 1))
@pytest.mark.parametrize("gain_db", [-10.0, -3.0, 4.5, 10.0])
def test_peaking_band_hits_its_gain_at_center(index: int, gain_db: float) -> None:
    freq = DEFAULT_CENTER_FREQS_HZ[index]
    response = frequency_response_db(one_band(index, gain_db), [freq], SR)
    assert response[0] == pytest.approx(gain_db, abs=1e-6)


@pytest.mark.parametrize(("index", "far_freq"), [(0, 5.0), (9, 21_000.0)])
@pytest.mark.parametrize("gain_db", [-8.0, 6.0])
def test_shelves_reach_half_gain_at_corner_and_full_gain_beyond(
    index: int, far_freq: float, gain_db: float
) -> None:
    corner = DEFAULT_CENTER_FREQS_HZ[index]
    at_corner, beyond = frequency_response_db(one_band(index, gain_db), [corner, far_freq], 48_000)
    assert at_corner == pytest.approx(gain_db / 2, abs=1e-6)
    assert beyond == pytest.approx(gain_db, abs=1.0)


def test_flat_eq_is_identity() -> None:
    x = synth.pink_noise(0.5, SR, np.random.default_rng(0))
    np.testing.assert_array_equal(apply_eq(x, SR, np.zeros(SPEC.n_bands)), x)


@given(gains=gain_vectors)
def test_opposite_curves_cancel(gains: FloatArray) -> None:
    freqs = np.geomspace(20, 20_000, 64)
    total = frequency_response_db(gains, freqs, SR) + frequency_response_db(-gains, freqs, SR)
    np.testing.assert_allclose(total, 0.0, atol=1e-6)


def test_filtering_matches_analytic_response() -> None:
    gains = np.array([6, -4, 3, 0, -2, 5, -6, 2, 8, -10], dtype=float)
    impulse = np.zeros(2**16)
    impulse[0] = 1.0
    measured = np.abs(np.fft.rfft(apply_eq(impulse, SR, gains)))
    freqs = np.fft.rfftfreq(impulse.size, 1 / SR)
    band = (freqs > 20) & (freqs < 20_000)
    expected = frequency_response_db(gains, freqs[band], SR)
    np.testing.assert_allclose(20 * np.log10(measured[band]), expected, atol=1e-3)


def test_stereo_channels_are_filtered_independently() -> None:
    rng = np.random.default_rng(1)
    left, right = synth.pink_noise(0.5, SR, rng), synth.tone_mix(0.5, SR, rng)
    gains = one_band(4, 7.0)
    out = apply_eq(np.stack([left, right], axis=1), SR, gains)
    np.testing.assert_allclose(out[:, 0], apply_eq(left, SR, gains))
    np.testing.assert_allclose(out[:, 1], apply_eq(right, SR, gains))


# --- Measurement --------------------------------------------------------------


def test_loudness_of_reference_sine() -> None:
    # BS.1770: a full-scale 997 Hz sine on one channel reads -3.01 LUFS.
    t = np.arange(3 * SR) / SR
    assert integrated_loudness(np.sin(2 * np.pi * 997 * t), SR) == pytest.approx(-3.01, abs=0.05)


def test_loudness_rejects_silence_and_short_clips() -> None:
    with pytest.raises(ValueError, match="silence"):
        integrated_loudness(np.zeros(SR), SR)
    with pytest.raises(ValueError, match="400 ms"):
        integrated_loudness(np.ones(SR // 10), SR)


def test_true_peak_sees_inter_sample_peaks() -> None:
    # A sine at fs/4 with a 45° phase never lands a sample on its crest:
    # every sample is at ±0.707 (-3 dBFS) while the waveform peaks at 0 dBTP.
    n = np.arange(SR)
    x = np.sin(np.pi / 2 * n + np.pi / 4)
    assert 20 * np.log10(np.max(np.abs(x))) == pytest.approx(-3.01, abs=0.01)
    assert true_peak_dbtp(x, SR) == pytest.approx(0.0, abs=0.2)


# --- Limiter ------------------------------------------------------------------


def test_limiter_leaves_quiet_audio_untouched() -> None:
    x = 0.1 * synth.pink_noise(1.0, SR, np.random.default_rng(2))
    np.testing.assert_array_equal(limit_true_peak(x, SR, -1.0), x)


@pytest.mark.parametrize("name", SIGNALS)
def test_limiter_enforces_ceiling(name: str) -> None:
    x = 4.0 * SIGNALS[name](np.random.default_rng(3))  # about +6 dBFS peaks
    limited = limit_true_peak(x, SR, -1.0)
    assert true_peak_dbtp(limited, SR) <= -1.0 + 0.05
    # Gain reduction only: the limiter never makes anything louder.
    assert np.all(np.abs(limited) <= np.abs(x) + 1e-12)


# --- Render: the guarantees the learner relies on ------------------------------


@given(
    gains=gain_vectors,
    signal_name=st.sampled_from(sorted(SIGNALS)),
    seed=st.integers(0, 2**16),
)
def test_render_matches_loudness_and_respects_true_peak(
    gains: FloatArray, signal_name: str, seed: int
) -> None:
    x = SIGNALS[signal_name](np.random.default_rng(seed))
    result = render(x, SR, gains, target=TARGET)
    assert abs(result.loudness_lufs - TARGET.integrated_lufs) <= TARGET.tolerance_lu
    assert result.true_peak_dbtp <= TARGET.true_peak_ceiling_dbtp
    # Independent re-measurement, so the test does not trust render's own report.
    assert (
        abs(integrated_loudness(result.audio, SR) - TARGET.integrated_lufs) <= TARGET.tolerance_lu
    )
    assert true_peak_dbtp(result.audio, SR) <= TARGET.true_peak_ceiling_dbtp


@given(a=gain_vectors, b=gain_vectors, seed=st.integers(0, 2**16))
def test_any_two_renderings_are_loudness_matched(a: FloatArray, b: FloatArray, seed: int) -> None:
    x = synth.drum_loop(2.0, SR, np.random.default_rng(seed))
    la = render(x, SR, a).loudness_lufs
    lb = render(x, SR, b).loudness_lufs
    assert abs(la - lb) <= TARGET.tolerance_lu


def test_render_is_deterministic() -> None:
    x = synth.drum_loop(2.0, SR, np.random.default_rng(4))
    gains = np.linspace(-10, 10, SPEC.n_bands)
    np.testing.assert_array_equal(render(x, SR, gains).audio, render(x, SR, gains).audio)


def test_limiter_is_rare_at_default_target() -> None:
    # The target leaves headroom: on noise, even a +10 dB tilt needs no limiting.
    x = synth.pink_noise(2.0, SR, np.random.default_rng(5))
    assert not render(x, SR, np.full(SPEC.n_bands, 10.0)).limiter_engaged
    assert not render(x, SR, np.linspace(-10, 10, SPEC.n_bands)).limiter_engaged


def test_render_raises_when_target_is_unreachable() -> None:
    x = synth.drum_loop(2.0, SR, np.random.default_rng(6))
    loud = LoudnessTarget(integrated_lufs=0.0)
    with pytest.raises(RuntimeError, match="lower the target"):
        render(x, SR, np.zeros(SPEC.n_bands), target=loud)
