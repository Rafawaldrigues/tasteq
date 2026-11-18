# Two-phase study, simulated

`tasteq study --config configs/study.yaml`, run with the code of the commit that adds this file (see `docs/report.md` §4.8).
8 seeds × 3 modes × 5 simulated listeners × 5 models = 600 simulated
participants. Each one answers screening questions on the 8 synthetic sounds,
picks 2–3 styles at random, then answers 4 questions per song on a diverse
subset of those styles. Scores are on the songs heard in phase 2 and on the
songs of the same styles held back.

Modes: rápida 8 + 12, média 12 + 24, longa 16 + 40 questions.

"One EQ for all" is the best single curve for the songs a person heard,
computed from the hidden taste: a model below it has learned something
song-specific. Means are dominated by `study_balance`, for which even the
best single curve is worse than flat.

Normalized regret: 0 is the listener's best EQ for that song, 1 is no better than flat.

| mode | model | after screening | heard songs | unheard songs | one EQ for all (ceiling) | beats one-EQ ceiling |
|---|---|---|---|---|---|---|
| rapida | neural | 5.33 | 5.70 | 6.39 | 0.52 | 0/40 |
| rapida | adaptive_mlp | 4.05 | 4.40 | 4.78 | 0.52 | 0/40 |
| rapida | gp | 2.47 | 2.08 | 2.48 | 0.52 | 0/40 |
| rapida | adaptive_curve_loose | 1.21 | 1.16 | 1.27 | 0.52 | 4/40 |
| rapida | adaptive_curve | 1.27 | 1.22 | 1.25 | 0.52 | 6/40 |
| media | neural | 3.84 | 5.01 | 6.98 | 0.48 | 0/40 |
| media | adaptive_mlp | 4.05 | 3.59 | 4.59 | 0.48 | 0/40 |
| media | gp | 2.45 | 1.60 | 2.06 | 0.48 | 0/40 |
| media | adaptive_curve_loose | 1.08 | 1.05 | 1.10 | 0.48 | 2/40 |
| media | adaptive_curve | 1.09 | 1.06 | 1.13 | 0.48 | 3/40 |
| longa | neural | 5.55 | 5.52 | 5.83 | 0.49 | 0/40 |
| longa | adaptive_mlp | 4.47 | 4.51 | 5.32 | 0.49 | 0/40 |
| longa | gp | 2.76 | 1.55 | 1.88 | 0.49 | 0/40 |
| longa | adaptive_curve_loose | 1.17 | 1.14 | 1.17 | 0.49 | 4/40 |
| longa | adaptive_curve | 1.24 | 1.20 | 1.23 | 0.49 | 2/40 |

Heard-song regret at the end of the longest mode, per listener:

| listener | neural | adaptive_mlp | gp | adaptive_curve_loose | adaptive_curve | one EQ for all |
|---|---|---|---|---|---|---|
| study_steady_bass | 2.39 | 2.18 | 0.61 | 0.89 | 0.91 | 0.00 |
| study_steady_vocal | 5.71 | 4.29 | 0.72 | 1.08 | 1.11 | 0.00 |
| study_by_style | 3.79 | 3.24 | 1.24 | 1.00 | 1.09 | 0.66 |
| study_balance | 11.61 | 9.95 | 4.01 | 1.74 | 1.88 | 1.10 |
| study_per_song | 4.08 | 2.89 | 1.15 | 1.00 | 0.99 | 0.67 |

## With a minimum audible step (2 dB RMS)

`tasteq study --config configs/study_audible.yaml`, 4 seeds.

Normalized regret: 0 is the listener's best EQ for that song, 1 is no better than flat.

| mode | model | after screening | heard songs | unheard songs | one EQ for all (ceiling) | beats one-EQ ceiling | ties |
|---|---|---|---|---|---|---|---|
| rapida | adaptive_curve | 1.17 | 1.09 | 1.17 | 0.41 | 1/20 | 4% |
| rapida | gp | 2.15 | 1.83 | 2.37 | 0.41 | 0/20 | 3% |
| media | adaptive_curve | 1.15 | 1.02 | 1.13 | 0.51 | 0/20 | 4% |
| media | gp | 2.28 | 1.60 | 2.46 | 0.51 | 0/20 | 8% |
| longa | adaptive_curve | 1.24 | 0.96 | 1.01 | 0.49 | 0/20 | 5% |
| longa | gp | 2.59 | 1.55 | 1.87 | 0.49 | 0/20 | 6% |

Heard-song regret at the end of the longest mode, per listener:

| listener | adaptive_curve | gp | one EQ for all |
|---|---|---|---|
| study_per_song | 0.90 | 1.30 | 0.69 |
| study_by_style | 0.87 | 1.11 | 0.60 |
| study_steady_bass | 0.51 | 0.34 | 0.00 |
| study_steady_vocal | 0.70 | 1.29 | 0.00 |
| study_balance | 1.80 | 3.70 | 1.14 |
