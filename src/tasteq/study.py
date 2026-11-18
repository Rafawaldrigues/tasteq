"""Simulation of the two-phase human study, on the real study tracks.

The web study asks each person questions in two phases:

1. *Screening*: A/B questions on synthetic sounds, to learn their overall taste.
2. *Music*: they pick favourite styles and answer on songs of those styles.

Before asking people for their time, this module runs the same protocol on
simulated listeners whose hidden taste is known, in each study mode (quick,
medium, long). It answers the study's questions where the answer is known:

- Does the EQ recommended for a song get closer to that listener's best EQ
  for that song as the person answers more questions?
- Does the model adapt *per song*, or does it only find one EQ for everything?
  The yardstick is the best single EQ for all the person's songs, computed
  from the hidden taste: a model that beats it on the songs it heard has
  learned something song-specific.
- Does what it learned carry over to songs of the same styles it never heard?

Listeners are built from the tracks' styles and band energies (see
``configs/study_tracks.yaml``), so the per-song tastes are tied to real songs.
"""

from __future__ import annotations

import json
import math
import os
import time
import zlib
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel
from scipy import optimize

from tasteq.acquisition import TrustRegion
from tasteq.analysis import diverse_subset
from tasteq.bench import ModelEntry, normalized_regret
from tasteq.dsp import FloatArray
from tasteq.models import make_model
from tasteq.preference import Comparison
from tasteq.simulate import (
    BASS_BOOST,
    GENTLE_WARMTH,
    SPEC,
    V_SHAPE,
    VOCAL_FORWARD,
    Persona,
    UtilityFn,
)
from tasteq.space import EQSpace

# --- Tracks -----------------------------------------------------------------------


@dataclass(frozen=True)
class StudyTrack:
    """A study track as the models and the simulated listeners see it."""

    name: str
    style: str
    context: FloatArray
    band_db: FloatArray

    @property
    def synthetic(self) -> bool:
        """True for screening sounds, False for songs."""
        return self.style == "synthetic"


def load_tracks(path: Path) -> list[StudyTrack]:
    """Read ``configs/study_tracks.yaml``."""
    data = yaml.safe_load(path.read_text())
    return [
        StudyTrack(
            t["name"],
            t["style"],
            np.asarray(t["context"], dtype=np.float64),
            np.asarray(t["band_db"], dtype=np.float64),
        )
        for t in data["tracks"]
    ]


# --- Simulated listeners -------------------------------------------------------------


def _per_track(
    targets: dict[bytes, FloatArray], default: FloatArray, sensitivity_db: float
) -> UtilityFn:
    """Quadratic taste whose target curve depends on the track (looked up by context)."""

    def utility(gains: FloatArray, context: FloatArray) -> FloatArray:
        target = targets.get(np.round(context, 9).tobytes(), default)
        return np.asarray(-0.5 * np.sum(((gains - target) / sensitivity_db) ** 2, axis=-1))

    return utility


def _smooth_curve(rng: np.random.Generator, amplitude_db: float) -> FloatArray:
    """Draw a random but plausible EQ curve: a few cosines over the bands."""
    bands = np.arange(SPEC.n_bands) / (SPEC.n_bands - 1)
    curve = sum(rng.normal() * np.cos(np.pi * k * bands) for k in range(1, 4))
    return np.asarray(amplitude_db * curve / max(float(np.max(np.abs(curve))), 1e-9))


STYLE_TARGETS = {
    "hiphop": BASS_BOOST,
    "afrobeat": BASS_BOOST,
    "electronic": BASS_BOOST,
    "pop": VOCAL_FORWARD,
    "acoustic": VOCAL_FORWARD,
    "rock": V_SHAPE,
}


def study_personas(tracks: list[StudyTrack]) -> dict[str, Persona]:
    """Build the simulated listeners, from one EQ for all to one EQ per song.

    The two ``steady_*`` listeners want the same EQ on every track: a model
    that invents per-song differences for them is overfitting. The others
    want something different per style, per song balance, or per song.
    """
    key = {t.name: np.round(t.context, 9).tobytes() for t in tracks}
    songs = [t for t in tracks if not t.synthetic]
    reference = np.mean([t.band_db for t in songs], axis=0)

    by_style = {key[t.name]: STYLE_TARGETS.get(t.style, GENTLE_WARMTH) for t in tracks}
    # Pushes each track towards the average balance of the songs: thin
    # tracks get bass, boomy ones lose it.
    balance = {key[t.name]: np.clip(-0.5 * (t.band_db - reference), -6.0, 6.0) for t in tracks}
    per_song = {
        key[t.name]: GENTLE_WARMTH
        + (
            0.0
            if t.synthetic
            else _smooth_curve(np.random.default_rng(zlib.crc32(t.name.encode())), 4.0)
        )
        for t in tracks
    }
    flat_taste = np.zeros(SPEC.n_bands)

    return {
        p.name: p
        for p in [
            Persona(
                "study_steady_bass",
                "Same bass boost on every track.",
                _per_track({}, BASS_BOOST, 3.5),
                well_specified=True,
            ),
            Persona(
                "study_steady_vocal",
                "Same vocal-forward curve on every track.",
                _per_track({}, VOCAL_FORWARD, 3.5),
                well_specified=True,
            ),
            Persona(
                "study_by_style",
                "Bass on hip-hop/afrobeat/electronic, voice on pop/acoustic, V on rock.",
                _per_track(by_style, GENTLE_WARMTH, 3.0),
                well_specified=True,
            ),
            Persona(
                "study_balance",
                "Corrects each track's own tonal balance towards the average song.",
                _per_track(balance, flat_taste, 3.0),
                well_specified=True,
            ),
            Persona(
                "study_per_song",
                "Mild warmth overall plus an idiosyncratic curve for each song.",
                _per_track(per_song, GENTLE_WARMTH, 3.0),
                well_specified=False,
            ),
        ]
    }


@lru_cache(maxsize=1024)
def _best_single_eq(persona: Persona, contexts_key: tuple[tuple[float, ...], ...]) -> FloatArray:
    """Find the one EQ with the best mean utility over these tracks (the one-EQ ceiling)."""
    contexts = [np.array(c) for c in contexts_key]
    rng = np.random.default_rng(0)

    def mean_utility(g: FloatArray) -> FloatArray:
        return np.asarray(np.mean([persona.utility_of(g, c) for c in contexts], axis=0))

    def loss(g: FloatArray) -> float:
        return -float(mean_utility(g)[0])

    starts = np.vstack(
        [
            rng.uniform(SPEC.min_gain_db, SPEC.max_gain_db, (4000, SPEC.n_bands)),
            np.zeros(SPEC.n_bands),
        ]
    )
    values = mean_utility(starts)
    best = starts[int(np.argmax(values))]
    for start in starts[np.argsort(values)[-4:]]:
        result = optimize.minimize(
            loss,
            start,
            method="L-BFGS-B",
            bounds=[(SPEC.min_gain_db, SPEC.max_gain_db)] * SPEC.n_bands,
        )
        if -result.fun > float(mean_utility(best)[0]):
            best = np.asarray(result.x)
    return np.asarray(best, dtype=np.float64)


# --- Configuration -------------------------------------------------------------------


class ModeConfig(BaseModel):
    """Questions per phase in one study mode."""

    screening: int
    music: int


class StudyConfig(BaseModel):
    """A study simulation, loaded from a versioned YAML file."""

    name: str
    tracks_file: Path = Path("configs/study_tracks.yaml")
    seeds: int = 8
    modes: dict[str, ModeConfig]
    screening_per_track: int = 2
    music_per_track: int = 4
    min_unheard: int = 2
    styles_per_person: tuple[int, int] = (2, 3)
    eval_every: int = 4
    personas: list[str] | None = None
    models: list[ModelEntry]

    @classmethod
    def from_yaml(cls, path: Path) -> StudyConfig:
        """Load and validate a config file."""
        return cls.model_validate(yaml.safe_load(path.read_text()))


# --- One run ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyCheckpoint:
    """Scores after ``n_music`` song questions (and the mode's screening)."""

    model: str
    persona: str
    mode: str
    seed: int
    n_screening: int
    n_music: int
    heard_regret: float
    unheard_regret: float
    single_eq_regret: float
    seconds: float


@dataclass(frozen=True)
class Participant:
    """What one simulated person hears; depends only on (persona, mode, seed)."""

    styles: tuple[str, ...]
    screening: tuple[int, ...]
    heard: tuple[int, ...]
    unheard: tuple[int, ...]


def _stable_seed(*parts: object) -> int:
    return zlib.crc32(repr(parts).encode())


def make_participant(
    config: StudyConfig, tracks: list[StudyTrack], persona: str, mode: str, seed: int
) -> Participant:
    """Pick styles, then the songs heard (diverse) and held out, like the web study."""
    rng = np.random.default_rng(_stable_seed("participant", persona, mode, seed))
    styles = sorted({t.style for t in tracks if not t.synthetic})
    low, high = config.styles_per_person
    chosen = tuple(sorted(rng.choice(styles, size=int(rng.integers(low, high + 1)), replace=False)))
    pool = [i for i, t in enumerate(tracks) if t.style in chosen]
    wanted = math.ceil(config.modes[mode].music / config.music_per_track)
    n_heard = max(1, min(wanted, len(pool) - config.min_unheard))
    order = diverse_subset(np.array([tracks[i].context for i in pool]), len(pool))
    heard = tuple(pool[j] for j in order[:n_heard])
    screening = tuple(i for i, t in enumerate(tracks) if t.synthetic)
    return Participant(chosen, screening, heard, tuple(i for i in pool if i not in heard))


def run_participant(
    config: StudyConfig, entry: ModelEntry, persona_name: str, mode: str, seed: int
) -> list[StudyCheckpoint]:
    """Screening then music questions for one simulated person; scored along the way."""
    tracks = load_tracks(config.tracks_file)
    persona = study_personas(tracks)[persona_name]
    who = make_participant(config, tracks, persona_name, mode, seed)
    counts = config.modes[mode]
    space = EQSpace(kind=entry.space.kind, n_components=entry.space.n_components)
    context_dim = len(tracks[0].context)
    model = make_model(
        entry.kind,
        space,
        context_dim,
        seed=_stable_seed("model", entry.name, persona_name, mode, seed),
        trust_region=TrustRegion(),
        **entry.params,
    )
    answers = np.random.default_rng(_stable_seed("answers", persona_name, mode, seed))
    heard = [tracks[i].context for i in who.heard]
    unheard = [tracks[i].context for i in who.unheard]
    single = _best_single_eq(persona, tuple(tuple(c) for c in heard))
    single_regret = float(np.mean([normalized_regret(persona, single, c) for c in heard]))

    def regret(contexts: list[FloatArray]) -> float:
        if not contexts:
            return float("nan")
        return float(
            np.mean([normalized_regret(persona, model.best_gains(c), c) for c in contexts])
        )

    def ask(context: FloatArray) -> None:
        a, b = model.suggest_pair(context)
        outcome = persona.choose(space.to_gains(a), space.to_gains(b), context, answers)
        model.update([Comparison(a, b, context, outcome)])

    def checkpoint(n_music: int) -> StudyCheckpoint:
        return StudyCheckpoint(
            entry.name,
            persona_name,
            mode,
            seed,
            counts.screening,
            n_music,
            regret(heard),
            regret(unheard),
            single_regret,
            time.perf_counter() - start,
        )

    start = time.perf_counter()
    for step in range(counts.screening):
        ask(
            tracks[who.screening[(step // config.screening_per_track) % len(who.screening)]].context
        )
    checkpoints = [checkpoint(0)]
    for step in range(1, counts.music + 1):
        ask(heard[((step - 1) // config.music_per_track) % len(heard)])
        if step % config.eval_every == 0 or step == counts.music:
            checkpoints.append(checkpoint(step))
    return checkpoints


# --- Running many ------------------------------------------------------------------------


def _personas(config: StudyConfig) -> list[str]:
    return config.personas or list(study_personas(load_tracks(config.tracks_file)))


def _jobs(
    config: StudyConfig, done: set[tuple[str, str, str, int]]
) -> Iterator[tuple[ModelEntry, str, str, int]]:
    for seed in range(config.seeds):
        for mode in config.modes:
            for persona in _personas(config):
                for entry in config.models:
                    if (entry.name, persona, mode, seed) not in done:
                        yield entry, persona, mode, seed


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _run_job(
    config: StudyConfig, entry: ModelEntry, persona: str, mode: str, seed: int
) -> list[dict[str, Any]]:
    import warnings

    warnings.filterwarnings("ignore")
    return [asdict(c) for c in run_participant(config, entry, persona, mode, seed)]


def run_study(config: StudyConfig, out_dir: Path, workers: int | None = None) -> Path:
    """Run every pending simulated participant, appending to ``out_dir/runs.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    runs_path = out_dir / "runs.jsonl"
    done = {
        (r["model"], r["persona"], r["mode"], r["seed"])
        for r in load_rows(runs_path)
        if r["n_music"] == 0
    }
    jobs = list(_jobs(config, done))
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(jobs)} participants pending ({len(done)} already done), {workers} workers")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_run_job, config, *job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            entry, persona, mode, seed = futures[future]
            rows = future.result()
            with runs_path.open("a") as f:
                f.writelines(json.dumps(row) + "\n" for row in rows)
            last = rows[-1]
            print(
                f"[{i}/{len(jobs)}] {entry.name:>15} {persona:>18} {mode:>6} seed={seed} "
                f"heard={last['heard_regret']:.2f} unheard={last['unheard_regret']:.2f} "
                f"single={last['single_eq_regret']:.2f} ({last['seconds']:.0f}s)",
                flush=True,
            )
    return runs_path


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read checkpoints written by :func:`run_study`."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- Summary -------------------------------------------------------------------------------


def summarize(rows: list[dict[str, Any]]) -> str:
    """Markdown tables: final scores per mode and model, overall and per listener."""

    def final(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for r in group:
            k = (r["model"], r["persona"], r["mode"], r["seed"])
            if k not in last or r["n_music"] > last[k]["n_music"]:
                last[k] = r
        return list(last.values())

    finals = final(rows)
    screening = [r for r in rows if r["n_music"] == 0]
    models = list(dict.fromkeys(r["model"] for r in rows))
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    personas = list(dict.fromkeys(r["persona"] for r in rows))

    def mean(values: list[float]) -> float:
        clean = [v for v in values if not math.isnan(v)]
        return float(np.mean(clean)) if clean else float("nan")

    lines = [
        "Normalized regret: 0 is the listener's best EQ for that song, 1 is no better than flat.",
        "",
        "| mode | model | after screening | heard songs | unheard songs "
        "| one EQ for all (ceiling) | beats one-EQ ceiling |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode in modes:
        for model in models:
            fs = [r for r in finals if r["mode"] == mode and r["model"] == model]
            ss = [r for r in screening if r["mode"] == mode and r["model"] == model]
            beats = sum(r["heard_regret"] < r["single_eq_regret"] for r in fs)
            lines.append(
                f"| {mode} | {model} | {mean([r['heard_regret'] for r in ss]):.2f} "
                f"| {mean([r['heard_regret'] for r in fs]):.2f} "
                f"| {mean([r['unheard_regret'] for r in fs]):.2f} "
                f"| {mean([r['single_eq_regret'] for r in fs]):.2f} | {beats}/{len(fs)} |"
            )
    lines += ["", "Heard-song regret at the end of the longest mode, per listener:", ""]
    longest = modes[-1]
    lines.append("| listener | " + " | ".join(models) + " | one EQ for all |")
    lines.append("|---|" + "---|" * (len(models) + 1))
    for persona in personas:
        cells = [
            mean(
                [
                    r["heard_regret"]
                    for r in finals
                    if r["persona"] == persona and r["model"] == m and r["mode"] == longest
                ]
            )
            for m in models
        ]
        ceiling = mean(
            [
                r["single_eq_regret"]
                for r in finals
                if r["persona"] == persona and r["mode"] == longest
            ]
        )
        lines.append(
            f"| {persona} | " + " | ".join(f"{c:.2f}" for c in cells) + f" | {ceiling:.2f} |"
        )
    return "\n".join(lines) + "\n"
