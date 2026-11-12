from pathlib import Path

import numpy as np
import pytest

from tasteq.bench import BenchConfig, ModelEntry, load_runs, make_task, normalized_regret, run_one
from tasteq.report import comparisons_to_reach, mean_ci, summary_table, write_report
from tasteq.simulate import PERSONAS, optimum

pytest.importorskip("torch")

CONFIG = BenchConfig(
    name="test",
    seeds=1,
    n_comparisons=4,
    eval_every=2,
    n_tracks=2,
    n_eval_contexts=2,
    n_heldout_pairs=40,
    personas=["bass_lover"],
    models=[ModelEntry(name="random", kind="random")],
)


def test_regret_is_zero_at_optimum_and_one_at_flat() -> None:
    persona, context = PERSONAS["v_shape"], np.array([0.5, 0.5])
    best, _ = optimum(persona, context)
    assert normalized_regret(persona, best, context) == pytest.approx(0.0, abs=1e-6)
    assert normalized_regret(persona, np.zeros(10), context) == pytest.approx(1.0)


def test_tasks_are_shared_across_models_and_differ_across_seeds() -> None:
    a, b = make_task(CONFIG, "bass_lover", 0), make_task(CONFIG, "bass_lover", 0)
    np.testing.assert_array_equal(a.tracks, b.tracks)
    c = make_task(CONFIG, "bass_lover", 1)
    assert not np.array_equal(a.tracks, c.tracks)


def test_run_produces_checkpoints_and_report(tmp_path: Path) -> None:
    rows = [vars(c) for c in run_one(CONFIG, CONFIG.models[0], "bass_lover", 0)]
    assert [r["n"] for r in rows] == [0, 2, 4]
    assert rows[0]["regret"] == pytest.approx(1.0)
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)

    paths = write_report(rows, tmp_path, [0.5])
    assert all(p.exists() for p in paths)
    assert "bass_lover" in (tmp_path / "summary.md").read_text()


def test_report_helpers() -> None:
    mean, half = mean_ci(np.array([1.0, 2.0, 3.0]))
    assert mean == 2.0
    assert half > 0
    rows = [{"n": 0, "regret": 1.0}, {"n": 5, "regret": 0.4}, {"n": 10, "regret": 0.1}]
    assert comparisons_to_reach(rows, 0.5) == 5
    assert comparisons_to_reach(rows, 0.95) is None
    table = summary_table(
        [dict(r, model="gp", persona="bass_lover", seed=0, accuracy=0.8) for r in rows], [0.5]
    )
    assert "| bass_lover | gp | 0.10 ± 0.00 |" in table


def test_config_rejects_unknown_persona(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\npersonas: [nobody]\nmodels: [{name: r, kind: random}]\n")
    with pytest.raises(ValueError, match="unknown personas"):
        BenchConfig.from_yaml(path)


def test_shipped_configs_are_valid() -> None:
    for path in Path("configs").glob("*.yaml"):
        BenchConfig.from_yaml(path)


def test_load_runs_missing_file(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "none.jsonl") == []
