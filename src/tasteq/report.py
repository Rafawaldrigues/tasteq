"""Aggregate benchmark checkpoints into figures and a Markdown summary."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tasteq.simulate import PERSONAS

# Fixed colors per model so a model keeps its color in every figure.
# Slots 1-4 of a CVD-validated categorical palette; the model-free baseline is
# a neutral gray because it is a reference line, not a competitor.
MODEL_COLORS = {
    "gp": "#2a78d6",
    "neural": "#eb6834",
    "gp_dct": "#1baf7a",
    "neural_dct": "#eda100",
    "random": "#8a8a85",
}
FALLBACK_COLORS = ["#e87ba4", "#008300"]
TEXT = "#2b2b29"
MUTED = "#6b6b66"
GRID = "#e4e3de"

Rows = list[dict[str, Any]]


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of a normal-approximation 95% confidence interval."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(values.size))


def _by(rows: Rows, *keys: str) -> dict[tuple[Any, ...], Rows]:
    groups: dict[tuple[Any, ...], Rows] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def _models(rows: Rows) -> list[str]:
    seen = list(dict.fromkeys(r["model"] for r in rows))
    return sorted(seen, key=lambda m: list(MODEL_COLORS).index(m) if m in MODEL_COLORS else 99)


def _color(model: str, models: list[str]) -> str:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    extras = [m for m in models if m not in MODEL_COLORS]
    return FALLBACK_COLORS[extras.index(model) % len(FALLBACK_COLORS)]


def _style_axis(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _curve(rows: Rows, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ns = sorted({r["n"] for r in rows})
    stats = [mean_ci(np.array([r[metric] for r in rows if r["n"] == n])) for n in ns]
    mean, half = (np.array(s) for s in zip(*stats, strict=True))
    return np.array(ns), mean, half


def plot_overview(rows: Rows, path: Path) -> None:
    """Regret vs. comparisons, averaged over personas, one panel per persona group."""
    import matplotlib.pyplot as plt

    models = _models(rows)
    groups = ["smooth taste", "irregular taste", "noisy answers"]
    fig, axes = plt.subplots(1, len(groups), figsize=(12, 3.8), sharey=True)
    for ax, group_name in zip(axes, groups, strict=True):
        subset = [r for r in rows if PERSONAS[r["persona"]].group == group_name]
        if not subset:
            ax.set_visible(False)
            continue
        # Average over personas within each seed first, so the CI is over seeds.
        per_seed: Rows = []
        for (model, seed, n), group in _by(subset, "model", "seed", "n").items():
            per_seed.append(
                {
                    "model": model,
                    "seed": seed,
                    "n": n,
                    "regret": np.mean([g["regret"] for g in group]),
                }
            )
        for model in models:
            ns, mean, half = _curve([r for r in per_seed if r["model"] == model], "regret")
            color = _color(model, models)
            ax.fill_between(ns, mean - half, mean + half, color=color, alpha=0.15, linewidth=0)
            ax.plot(ns, mean, color=color, linewidth=2, label=model)
            ax.annotate(
                model,
                (ns[-1], mean[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                color=TEXT,
                fontsize=8,
                va="center",
            )
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        names = ", ".join(sorted({r["persona"] for r in subset}))
        ax.set_title(f"{group_name.capitalize()}\n", color=TEXT, fontsize=10, loc="left")
        ax.set_title(names, color=MUTED, fontsize=7, loc="right")
        ax.set_xlabel("pairwise comparisons", color=MUTED, fontsize=9)
        _style_axis(ax)
    axes[0].set_ylabel("normalized regret (1 = flat EQ, lower is better)", color=MUTED, fontsize=9)
    axes[0].set_ylim(bottom=0)
    [ax for ax in axes if ax.get_visible()][-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_per_persona(rows: Rows, path: Path, metric: str = "regret") -> None:
    """Small multiples: one panel per persona, one line per model, 95% CI bands."""
    import matplotlib.pyplot as plt

    models = _models(rows)
    personas = list(dict.fromkeys(r["persona"] for r in rows))
    cols = 3
    nrows = int(np.ceil(len(personas) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(11, 3 * nrows), sharex=True, squeeze=False)
    for ax, persona in zip(axes.flat, personas, strict=False):
        for model in models:
            subset = [r for r in rows if r["persona"] == persona and r["model"] == model]
            if not subset:
                continue
            ns, mean, half = _curve(subset, metric)
            color = _color(model, models)
            ax.fill_between(ns, mean - half, mean + half, color=color, alpha=0.15, linewidth=0)
            ax.plot(ns, mean, color=color, linewidth=2, label=model)
        if metric == "regret":
            ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(f"{persona} ({PERSONAS[persona].group})", color=TEXT, fontsize=9, loc="left")
        _style_axis(ax)
    for ax in axes.flat[len(personas) :]:
        ax.set_visible(False)
    label = "normalized regret" if metric == "regret" else "held-out pair accuracy"
    fig.supylabel(label, color=MUTED, fontsize=9)
    fig.supxlabel("pairwise comparisons", color=MUTED, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(models), loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def comparisons_to_reach(rows: Rows, fraction: float) -> int | None:
    """First checkpoint where the run captured ``fraction`` of the possible gain."""
    for row in sorted(rows, key=lambda r: r["n"]):
        if 1.0 - row["regret"] >= fraction:
            return int(row["n"])
    return None


def summary_table(rows: Rows, thresholds: list[float]) -> str:
    """Markdown table: final regret, final accuracy and comparisons to reach targets."""
    final_n = max(r["n"] for r in rows)
    header = ["persona", "model", f"regret @ {final_n}", f"accuracy @ {final_n}"]
    header += [f"n to {int(t * 100)}% (reached)" for t in thresholds]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    models = _models(rows)
    for persona in dict.fromkeys(r["persona"] for r in rows):
        for model in models:
            runs = _by([r for r in rows if r["persona"] == persona and r["model"] == model], "seed")
            if not runs:
                continue
            finals = [max(rs, key=lambda r: r["n"]) for rs in runs.values()]
            regret = mean_ci(np.array([f["regret"] for f in finals]))
            accuracy = mean_ci(np.array([f["accuracy"] for f in finals]))
            cells = [
                persona,
                model,
                f"{regret[0]:.2f} ± {regret[1]:.2f}",
                f"{accuracy[0]:.2f} ± {accuracy[1]:.2f}",
            ]
            for t in thresholds:
                reached = [
                    n for rs in runs.values() if (n := comparisons_to_reach(rs, t)) is not None
                ]
                share = f"{len(reached)}/{len(runs)}"
                cells.append(
                    f"{int(np.median(reached))} ({share})" if reached else f"- (0/{len(runs)})"
                )
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(rows: Rows, out_dir: Path, thresholds: list[float]) -> list[Path]:
    """Write all figures and ``summary.md`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "overview.png",
        out_dir / "regret_per_persona.png",
        out_dir / "accuracy_per_persona.png",
    ]
    plot_overview(rows, paths[0])
    plot_per_persona(rows, paths[1], "regret")
    plot_per_persona(rows, paths[2], "accuracy")
    summary = out_dir / "summary.md"
    summary.write_text(
        "Mean ± 95% CI over seeds. Regret is normalized: 0 = true optimum, "
        "1 = no better than flat. *n to X%*: median comparisons until the "
        "recommendation captures X% of the achievable gain over flat, among "
        "seeds that got there.\n\n" + summary_table(rows, thresholds) + "\n"
    )
    return [*paths, summary]
