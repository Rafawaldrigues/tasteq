"""Command-line interface. Subcommands are added as each stage lands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tasteq import __version__

app = typer.Typer(help="An equalizer that learns your personal sound preference.")


@app.callback()
def main() -> None:
    """Keep subcommand-style invocation even while there is a single command."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def bench(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Benchmark YAML.")],
    out: Annotated[Path, typer.Option(help="Directory for results.")] = Path("results"),
    workers: Annotated[
        int | None, typer.Option(help="Parallel processes (default: CPUs - 2).")
    ] = None,
    report_only: Annotated[bool, typer.Option(help="Skip running; rebuild figures only.")] = False,
) -> None:
    """Run models against simulated listeners and write figures + summary."""
    from tasteq.bench import BenchConfig, load_runs, run_benchmark
    from tasteq.report import write_report

    cfg = BenchConfig.from_yaml(config)
    out_dir = out / cfg.name
    runs_path = out_dir / "runs.jsonl"
    if not report_only:
        run_benchmark(cfg, out_dir, workers)
    rows = load_runs(runs_path)
    if not rows:
        raise typer.BadParameter(f"no results in {runs_path}")
    for path in write_report(rows, out_dir, cfg.reach_thresholds):
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
