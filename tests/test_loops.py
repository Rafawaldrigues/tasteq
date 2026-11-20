from pathlib import Path

import numpy as np
import soundfile as sf

from web.loops import LOOPS, write_screening_loops


def test_loops_are_clean_stereo_and_cover_the_top_bands(tmp_path: Path) -> None:
    paths = write_screening_loops(tmp_path, n_loops=2, duration_s=6.0)
    assert [p.name for p in paths] == [f"loop_{i:02d}_{LOOPS[i].name}.wav" for i in range(2)]
    for path in paths:
        audio, rate = sf.read(path)
        assert audio.shape == (len(audio), 2)
        assert 0.5 < np.abs(audio).max() <= 0.9
        mono = audio.mean(axis=1)
        power = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
        freqs = np.fft.rfftfreq(len(mono), 1.0 / rate)
        top = power[(freqs >= 10_000) & (freqs < 16_000)].mean()
        mids = power[(freqs >= 100) & (freqs < 1_000)].mean()
        # Some air for the treble bands, but far from a noise bed.
        assert -75.0 < 10 * np.log10(top / mids) < -20.0
