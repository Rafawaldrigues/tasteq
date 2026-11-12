"""Benchmark: every model against every simulated persona, over many seeds.

One *run* is one (model, persona, seed) triple: the model asks the persona
``n_comparisons`` questions, cycling through a fixed set of calibration
tracks (contexts), and is scored every ``eval_every`` comparisons on
held-out tracks it never practiced on.

Common random numbers: tracks, evaluation contexts and held-out pairs depend
only on (persona, seed), so all models face exactly the same tasks.

Results are appended to ``runs.jsonl`` as they finish, so an interrupted
benchmark resumes where it stopped.
"""

from __future__ import annotations

import json
import os
import time
import zlib
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

from tasteq.acquisition import TrustRegion
from tasteq.dsp import FloatArray
from tasteq.models import make_model
from tasteq.preference import Comparison
from tasteq.simulate import PERSONAS, Persona, optimum
from tasteq.space import EQSpace, SpaceKind

# --- Configuration ------------------------------------------------------------


class SpaceConfig(BaseModel):
    """Search-space settings for one model entry."""

    kind: SpaceKind = "raw"
    n_components: int = 5


class ModelEntry(BaseModel):
    """A model to benchmark: a display name, a kind and its parameters."""

    name: str
    kind: str
    space: SpaceConfig = Field(default_factory=SpaceConfig)
    params: dict[str, Any] = Field(default_factory=dict)


class BenchConfig(BaseModel):
    """A benchmark experiment, loaded from a versioned YAML file."""

    name: str
    seeds: int = 10
    n_comparisons: int = 60
    eval_every: int = 5
    context_dim: int = 2
    n_tracks: int = 10
    n_eval_contexts: int = 6
    n_heldout_pairs: int = 300
    personas: list[str] = Field(default_factory=lambda: list(PERSONAS))
    models: list[ModelEntry]
    reach_thresholds: list[float] = Field(default_factory=lambda: [0.5, 0.8])

    @classmethod
    def from_yaml(cls, path: Path) -> BenchConfig:
        """Load and validate a config file."""
        config = cls.model_validate(yaml.safe_load(path.read_text()))
        unknown = set(config.personas) - set(PERSONAS)
        if unknown:
            raise ValueError(f"unknown personas: {sorted(unknown)}")
        return config


# --- One run --------------------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    """Scores of one model after ``n`` comparisons."""

    model: str
    persona: str
    seed: int
    n: int
    regret: float
    accuracy: float
    seconds: float


@dataclass(frozen=True)
class Task:
    """Everything a run needs; derived only from (persona, seed)."""

    tracks: FloatArray
    eval_contexts: FloatArray
    heldout_a: FloatArray
    heldout_b: FloatArray
    heldout_context: FloatArray


def _stable_seed(*parts: object) -> int:
    return zlib.crc32(repr(parts).encode())


def make_task(config: BenchConfig, persona: str, seed: int) -> Task:
    """Tracks, evaluation contexts and held-out pairs for one (persona, seed)."""
    rng = np.random.default_rng(_stable_seed("task", persona, seed))
    k = config.context_dim
    # Held-out pairs live within ±5 dB of flat, where an EQ is actually usable,
    # so accuracy is not inflated by trivially different extremes.
    m = config.n_heldout_pairs
    return Task(
        tracks=rng.uniform(size=(config.n_tracks, k)),
        eval_contexts=rng.uniform(size=(config.n_eval_contexts, k)),
        heldout_a=rng.uniform(0.25, 0.75, size=(m, 10)),
        heldout_b=rng.uniform(0.25, 0.75, size=(m, 10)),
        heldout_context=rng.uniform(size=(m, k)),
    )


def normalized_regret(persona: Persona, gains: FloatArray, context: FloatArray) -> float:
    """``(u* − u(x)) / (u* − u(flat))``: 0 is optimal, 1 is no better than flat."""
    _, best = optimum(persona, context)
    flat = float(persona.utility_of(np.zeros_like(gains), context)[0])
    value = float(persona.utility_of(gains, context)[0])
    return (best - value) / max(best - flat, 1e-9)


def run_one(
    config: BenchConfig, entry: ModelEntry, persona_name: str, seed: int
) -> list[Checkpoint]:
    """Run one model on one persona for one seed and return its checkpoints."""
    persona = PERSONAS[persona_name]
    task = make_task(config, persona_name, seed)
    space = EQSpace(kind=entry.space.kind, n_components=entry.space.n_components)
    model = make_model(
        entry.kind,
        space,
        config.context_dim,
        seed=_stable_seed("model", entry.name, persona_name, seed),
        trust_region=TrustRegion(),
        **entry.params,
    )
    answers = np.random.default_rng(_stable_seed("answers", persona_name, seed))
    raw = EQSpace(kind="raw")
    heldout_gains_a = raw.to_gains(task.heldout_a)
    heldout_gains_b = raw.to_gains(task.heldout_b)

    def evaluate(n: int, elapsed: float) -> Checkpoint:
        regret = np.mean(
            [normalized_regret(persona, model.best_gains(c), c) for c in task.eval_contexts]
        )
        return Checkpoint(
            entry.name,
            persona_name,
            seed,
            n,
            float(regret),
            _heldout_accuracy(model, space, persona, task, heldout_gains_a, heldout_gains_b),
            elapsed,
        )

    start = time.perf_counter()
    checkpoints = [evaluate(0, 0.0)]
    for step in range(1, config.n_comparisons + 1):
        context = task.tracks[(step - 1) % config.n_tracks]
        a, b = model.suggest_pair(context)
        outcome = persona.choose(space.to_gains(a), space.to_gains(b), context, answers)
        model.update([Comparison(a, b, context, outcome)])
        if step % config.eval_every == 0:
            checkpoints.append(evaluate(step, time.perf_counter() - start))
    return checkpoints


def _heldout_accuracy(
    model: Any,
    space: EQSpace,
    persona: Persona,
    task: Task,
    gains_a: FloatArray,
    gains_b: FloatArray,
) -> float:
    """Share of held-out pairs whose noise-free winner the model ranks first.

    Pairs are stated in gains and projected into the model's space, so models
    with different spaces are scored on the same questions. Pairs the persona
    cannot tell apart are skipped.
    """
    correct, total = 0, 0
    for context in np.unique(task.heldout_context, axis=0):
        rows = np.all(task.heldout_context == context, axis=1)
        ua = persona.utility_of(gains_a[rows], context)
        ub = persona.utility_of(gains_b[rows], context)
        audible = np.abs(ua - ub) >= persona.tie_margin
        if not audible.any():
            continue
        xa = project_gains(space, gains_a[rows][audible])
        xb = project_gains(space, gains_b[rows][audible])
        ma, _ = model.predict(xa, context)
        mb, _ = model.predict(xb, context)
        correct += int(np.sum((ma > mb) == (ua[audible] > ub[audible])))
        total += int(audible.sum())
    return correct / max(total, 1)


def project_gains(space: EQSpace, gains: FloatArray) -> FloatArray:
    """Closest unit-cube point(s) of ``space`` for the given gains (least squares)."""
    spec = space.spec
    centered = (gains - (spec.max_gain_db + spec.min_gain_db) / 2) / (
        (spec.max_gain_db - spec.min_gain_db) / 2
    )
    if space.kind == "raw":
        z = centered
    else:
        z, *_ = np.linalg.lstsq(space.basis, centered.T, rcond=None)
        z = z.T
    return np.clip((z + 1.0) / 2.0, 0.0, 1.0)


# --- Running many ------------------------------------------------------------------


def _jobs(
    config: BenchConfig, done: set[tuple[str, str, int]]
) -> Iterator[tuple[ModelEntry, str, int]]:
    for seed in range(config.seeds):
        for persona in config.personas:
            for entry in config.models:
                if (entry.name, persona, seed) not in done:
                    yield entry, persona, seed


def _init_worker() -> None:
    # One process per run; intra-op threading would only oversubscribe the CPU.
    import torch

    torch.set_num_threads(1)


def _run_job(
    config: BenchConfig, entry: ModelEntry, persona: str, seed: int
) -> list[dict[str, Any]]:
    import warnings

    warnings.filterwarnings("ignore")
    return [asdict(c) for c in run_one(config, entry, persona, seed)]


def run_benchmark(config: BenchConfig, out_dir: Path, workers: int | None = None) -> Path:
    """Run every pending job, appending checkpoints to ``out_dir/runs.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
    runs_path = out_dir / "runs.jsonl"
    done = {(r["model"], r["persona"], r["seed"]) for r in load_runs(runs_path) if r["n"] == 0}
    jobs = list(_jobs(config, done))
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(jobs)} runs pending ({len(done)} already done), {workers} workers")

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_run_job, config, *job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            entry, persona, seed = futures[future]
            rows = future.result()
            with runs_path.open("a") as f:
                f.writelines(json.dumps(row) + "\n" for row in rows)
            final = rows[-1]
            print(
                f"[{i}/{len(jobs)}] {entry.name:>10} {persona:>16} seed={seed} "
                f"regret={final['regret']:.3f} acc={final['accuracy']:.3f} "
                f"({final['seconds']:.0f}s)",
                flush=True,
            )
    return runs_path


def load_runs(path: Path) -> list[dict[str, Any]]:
    """Read checkpoints written by :func:`run_benchmark`."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
