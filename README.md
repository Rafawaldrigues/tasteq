# tasteq

**An open-source equalizer that learns *your* sound.** Compare two versions of a
song, say which one you prefer, and tasteq uses Bayesian preference optimization
to converge on your ideal 10-band EQ, conditioned on the kind of music you are
playing.

[![CI](https://github.com/Rafawaldrigues/tasteq/actions/workflows/ci.yml/badge.svg)](https://github.com/Rafawaldrigues/tasteq/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

> **Status:** active development. The DSP core, three preference models, the
> simulated-listener benchmark and the calibrate/listen CLI work; benchmark
> figures and the interactive demo are next. See the [roadmap](#roadmap).

## Why

Tuning an EQ by hand is tedious and does not follow the music. Existing tools
either run a one-off static test (Sonarworks SoundID), are paid and
platform-locked (EQK), or target producers (iQ / Ear Machine). tasteq is
**open source, keeps learning from everyday feedback, and is evaluated
rigorously**.

## How it works

1. **Analyze** a local music folder: features and a representative excerpt per
   track, cached.
2. **Calibrate** with ~10 diverse tracks and 3–5 A/B comparisons each.
3. **Listen and learn**: tracks play with the predicted best EQ; 👍/👎 and
   occasional A/B tests keep refining the model.

Two rules keep the learning honest:

- **AI controls the EQ, never the audio.** The audio path is deterministic DSP
  (biquad filters); the model only chooses 10 gains in [-10, +10] dB.
- **Louder never wins.** Every version is normalized to −23 LUFS (±0.5 LU) with
  true peak ≤ −1 dBTP, verified by property-based tests for any EQ in range.

Design decisions and their trade-offs are in [docs/design.md](docs/design.md).

## Quickstart

```bash
pip install -e ".[ml,audio]"
tasteq analyze ~/Music              # one-time analysis, cached in ~/.tasteq/tasteq.sqlite
tasteq calibrate                    # ~10 diverse tracks x 4 A/B questions
tasteq listen                       # whole tracks with your EQ; + / - while playing
```

No music handy? `tasteq demo-library ./demo-music` writes synthetic,
copyright-free tracks to try it with. `tasteq show` prints the learned EQ per
track, and `tasteq bench --config configs/main.yaml` reproduces the
simulated-listener benchmark.

Local playback uses PortAudio when installed and otherwise a system player
(`pw-play`, `paplay`, `aplay` or `ffplay`).

## Roadmap

Planned for this repository:

- [x] Loudness-matched EQ rendering with property-based tests
- [x] Simulated listeners + preference models (random, preference GP, neural
      Bradley–Terry ensemble) + reproducible benchmark
- [ ] Benchmark results and figures in this README
- [x] Track analysis and context features
- [x] Calibration and listen-and-learn CLI
- [ ] Interactive Gradio demo with a live EQ curve and uncertainty band
- [ ] Blind test, technical report

Longer term (not implemented here):

1. Dynamic EQ per section, using look-ahead from the pre-analysis
2. Long-term listening profile (how often you play something ≠ wanting it louder)
3. Per-instrument control via source separation ("more vocals, less hi-hat")
4. Real-time system-wide EQ and an Android app
5. A prior learned from many users for the cold start

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml,audio]"
pytest
```

## License

MIT. The repository contains no copyrighted music; tests use synthetic audio.
