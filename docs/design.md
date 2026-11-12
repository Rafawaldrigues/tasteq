# Design decisions

This document records the decisions behind tasteq and the reasoning for each,
including the places where the original idea was changed after review. It is
the source of truth for scope; the README summarizes it.

## Problem statement

Find, for one listener, the 10-band EQ curve `x ∈ [-10, 10]^10 dB` that maximizes
an unknown utility `u(x, c)`, where `c` describes the track being played. The
only observations are noisy pairwise preferences between two loudness-matched
renderings of the same excerpt. This is **preferential Bayesian optimization**
with a small budget (≈30–50 comparisons for calibration, a few more per day of
use).

## Scope (current)

A Python library + CLI + local demo, offline, on a PC. It must show with
evidence that the system learns an EQ preference. Out of scope for now: real-time
system-wide EQ, mobile, streaming services, source separation, dynamic EQ inside
a track, modelling order effects, multi-user priors.

## Decisions

### 1. The model controls the EQ; it never processes audio

The audio path is classic, deterministic DSP (RBJ biquads, second-order
sections). The model only outputs 10 gains. This keeps the audio path testable
and explainable, and keeps any learning bug from becoming an audio artifact.

### 2. Parametrization

- Bands: 31, 62, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz. Low shelf at 31 Hz,
  high shelf at 16 kHz, peaking filters with Q = 1.2 in between.
- Gain range: −10 to +10 dB per band (configurable).
- A wide range is not the same as wide steps: candidate pairs are drawn inside
  an **adaptive trust region** centered on the current best, which starts large
  and shrinks as posterior uncertainty falls.

**Change after review — smooth low-dimensional option.** Ten independent
dimensions are a lot for ~50 binary comparisons, and adjacent bands of a real
EQ curve are strongly correlated. Every model therefore accepts an optional
*basis* that maps a low-dimensional latent `z` (e.g. 4–5 DCT coefficients over
log-frequency) to the 10 gains. Raw 10-D vs. smooth basis is an ablation in the
benchmark rather than an assumption.

### 3. Loudness normalization (critical)

Louder almost always wins an A/B test, which would poison learning. Every
rendering is normalized to the same integrated loudness (ITU-R BS.1770 / EBU
R128) with tolerance ≤ 0.5 LU and true peak ≤ −1 dBTP, without clipping.

**Change after review — low target, limiter as a safety net.** Normalizing to a
loud target and then limiting would cut loudness again and break the tolerance.
The default target is **−23 LUFS** (EBU R128), which leaves room for +10 dB
boosts. A transparent lookahead true-peak limiter exists only as a safety net;
the render reports whether it engaged so tests and analysis can see it. A
property-based test (`hypothesis`) checks the guarantees for any gain vector in
range on several synthetic signal families, including impulsive ones.

**Known limitation.** LUFS is a K-weighted energy measure, not a perceptual
loudness model. Two curves matched in LUFS can still differ slightly in perceived
loudness (especially bass-heavy vs. bright). This residual bias is documented
rather than claimed solved.

### 4. Pairwise feedback

- Answers: "prefer A", "prefer B", "no difference".
- A/B order is randomized and logged (not modelled yet).
- Both versions play the same 10–15 s representative excerpt chosen during
  analysis, and can be replayed before answering.

**Change after review — ties.** `PairwiseGP` only models strict preferences.
Ties are stored, ignored by the GP likelihood, and used as a soft label (0.5)
by the neural model. Whether ties help is measured in the benchmark.

### 5. Models

A common `PreferenceModel` interface (`update`, `suggest_pair`, `best`,
`predict -> (mean, std)`) with:

1. **Random trust-region baseline** — same proposal geometry, no model.
2. **Preference GP** — BoTorch `PairwiseGP` with an expected-utility-of-best-
   option (EUBO) acquisition. Expected to win in the low-data regime.
3. **Neural Bradley–Terry** — MLP utility, deep ensemble for uncertainty,
   Thompson sampling for exploration.
4. *(Optional)* linear contextual dueling bandit as a middle point.

**Change after review — additive context.** Ten calibration tracks cannot teach
a full interaction between EQ and context. The GP uses an additive kernel
`k((x,c),(x',c')) = k_x(x,x') + k_x(x,x')·k_c(c,c')`: a shared taste that is
learned fast, plus a context-dependent correction that grows only with data.
During acquisition the context is held fixed and only `x` is optimized.

### 6. Track context

Light, cached features (relative band energy, spectral centroid and rolloff,
tempo, dynamic range) reduced to ≤ 6 normalized dimensions. One fixed EQ per
track.

### 7. Calibration

About 10 tracks chosen to maximize spectral diversity (farthest-point sampling
in feature space), 3–5 comparisons each.

### 8. Listen + learn mode

The full track plays with the predicted best EQ. 👍/👎 is converted into a
comparison **against flat on the same track** (👍 → EQ ≻ flat, 👎 → flat ≻ EQ).
Flat is a stable, interpretable anchor; the drawbacks (a thumbs-down may be about
the song, not the EQ; a mild pull towards flat) are handled by giving these
comparisons a lower weight, which is also a benchmark parameter. Every N tracks
an exploratory A/B is inserted.

### 9. Evaluation

- **Simulated users.** Personas with hidden utilities, answering through a noisy
  Bradley–Terry/Thurstone choice model.

  **Change after review — avoid a circular benchmark.** If every persona were
  smooth and unimodal, the GP would win by construction. The suite also includes
  *misspecified* personas: band interactions, multimodal tastes, non-smooth
  utilities, genre-dependent tastes, inconsistent and very noisy users. Cases
  where the GP loses are reported.

- **Metrics.** Normalized simple regret
  `(u* − u(x̂)) / (u* − u(flat))`, comparisons needed to reach 90% of the
  optimum, held-out pairwise accuracy. Mean ± 95% CI over seeds.
- **Human blind test.** Learned vs. flat vs. random, loudness-matched,
  randomized order, logged. With a single listener this is *anecdotal evidence*
  and is reported with a binomial test and honest confidence intervals. Simulated
  results show the algorithm recovers assumed preferences; they do not prove
  real listeners prefer the result.
- **Reproducibility.** `tasteq bench --config configs/<name>.yaml`, versioned
  YAML configs, fixed seeds, a short `ci` config and a `full` config.

### 10. Persistence

SQLite: tracks, features, comparisons (timestamp, order, excerpt, model,
persona/human, weight) and sessions. Any model can be rebuilt from history.

## Tooling choices

| Need | Choice | Why |
|---|---|---|
| EQ filters | own RBJ biquads on `scipy.signal.sosfilt` | exact analytic frequency response for plots and tests; avoids `pedalboard`, which is GPL-3.0 and would conflict with an MIT license |
| Loudness | `pyloudnorm` | reference BS.1770 implementation in Python |
| True peak | 4× polyphase oversampling | BS.1770-4 Annex 2 method |
| Preference GP | BoTorch / GPyTorch | maintained `PairwiseGP` + preference acquisitions |
| Neural model | PyTorch | — |
| Features | `librosa` | standard, well-tested |
| Demo | Gradio | native audio component, quick A/B layout |
| Packaging | `pyproject.toml` + hatchling | standard, no lock-in |

Compute: everything is designed to run on CPU; models accept a `device` setting
so a CUDA GPU is used when available, but nothing depends on it.
