"""Command-line interface. Subcommands are added as each stage lands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from tasteq import __version__

app = typer.Typer(help="An equalizer that learns your personal sound preference.")

DEFAULT_DB = Path(os.environ.get("TASTEQ_DB", Path.home() / ".tasteq" / "tasteq.sqlite"))
DbOption = Annotated[Path, typer.Option(help="SQLite database (env: TASTEQ_DB).")]
ModelOption = Annotated[str, typer.Option(help="Preference model: gp, neural or random.")]


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


@app.command()
def analyze(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Music folder.")],
    db: DbOption = DEFAULT_DB,
) -> None:
    """Analyze every track in a folder (cached; only new or changed files are read)."""
    from tasteq.library import analyze_folder
    from tasteq.storage import Storage

    with Storage(db) as storage:
        report = analyze_folder(
            folder, storage, on_progress=lambda path, status: typer.echo(f"{status:>9}  {path}")
        )
    typer.echo(f"{report.analyzed} analyzed, {report.cached} cached, {len(report.failed)} failed")
    for failure in report.failed:
        typer.echo(f"  {failure}", err=True)


@app.command()
def calibrate(
    db: DbOption = DEFAULT_DB,
    model: ModelOption = "gp",
    tracks: Annotated[int, typer.Option(help="Calibration tracks.")] = 10,
    per_track: Annotated[int, typer.Option(help="A/B questions per track.")] = 4,
) -> None:
    """Run the initial calibration: A/B questions on a spectrally diverse set of tracks."""
    from tasteq.calibrate import run_calibration
    from tasteq.playback import default_player
    from tasteq.session import Tuner
    from tasteq.storage import Storage

    player = default_player()
    with Storage(db) as storage:
        tuner = Tuner(storage, model, session_kind="calibrate")
        answered = run_calibration(tuner, player, typer.prompt, tracks, per_track)
        tuner.close()
    typer.echo(f"{answered} answers recorded ({tuner.n_comparisons} in total).")


@app.command()
def listen(
    db: DbOption = DEFAULT_DB,
    model: ModelOption = "gp",
    explore_every: Annotated[int, typer.Option(help="A/B question every N tracks (0: never).")] = 3,
) -> None:
    """Play the library with your learned EQ; 👍/👎 and occasional A/Bs keep it learning."""
    from tasteq.listen import run_listening
    from tasteq.playback import default_player
    from tasteq.session import Tuner
    from tasteq.storage import Storage

    player = default_player()
    with Storage(db) as storage:
        tuner = Tuner(storage, model, session_kind="listen")
        played = run_listening(
            tuner,
            player,
            lambda text: typer.prompt(text, default="", show_default=False),
            explore_every,
        )
        tuner.close()
    typer.echo(f"{played} tracks played ({tuner.n_comparisons} comparisons in total).")


@app.command()
def show(db: DbOption = DEFAULT_DB, model: ModelOption = "gp") -> None:
    """Print the learned EQ for each track."""
    from tasteq.dsp import DEFAULT_CENTER_FREQS_HZ
    from tasteq.session import Tuner
    from tasteq.storage import Storage

    with Storage(db) as storage:
        tuner = Tuner(storage, model, session_kind="show")
        header = "".join(
            f"{int(f) if f < 1000 else str(int(f // 1000)) + 'k':>6}"
            for f in DEFAULT_CENTER_FREQS_HZ
        )
        typer.echo(f"{tuner.n_comparisons} comparisons learned from\n")
        typer.echo(f"{'track':<32}{header}")
        for track in tuner.library.tracks:
            gains = tuner.best_gains(track)
            name = Path(track.analysis.path).name[:30]
            typer.echo(f"{name:<32}" + "".join(f"{g:+6.1f}" for g in gains))
        tuner.close()


@app.command()
def demo(
    db: Annotated[
        Path | None, typer.Option(help="SQLite database (default: TASTEQ_DB, or a demo one).")
    ] = None,
    model: ModelOption = "gp",
    synthetic: Annotated[
        bool, typer.Option(help="Use generated, copyright-free tracks in a separate database.")
    ] = False,
    port: Annotated[int, typer.Option(help="Local port.")] = 7860,
) -> None:
    """Open the browser demo: blind A/B questions and a live EQ curve."""
    from tasteq.demo import DemoSession, build_app
    from tasteq.library import analyze_folder
    from tasteq.session import Tuner
    from tasteq.storage import Storage
    from tasteq.synth import write_demo_library

    if synthetic:
        home = DEFAULT_DB.parent
        db = db or home / "demo.sqlite"
        folder = home / "demo-music"
        if not folder.exists():
            typer.echo(f"writing synthetic tracks to {folder}")
            write_demo_library(folder)
    db = db or DEFAULT_DB
    with Storage(db) as storage:
        if synthetic:
            analyze_folder(folder, storage)
        try:
            tuner = Tuner(storage, model, session_kind="demo")
        except LookupError:
            raise typer.BadParameter(
                "no analyzed tracks: run `tasteq analyze <folder>` first, or use --synthetic"
            ) from None
        try:
            build_app(DemoSession(tuner)).launch(server_port=port)
        finally:
            tuner.close()


@app.command("demo-library")
def demo_library(
    folder: Annotated[Path, typer.Argument(help="Where to write the synthetic tracks.")],
    tracks: Annotated[int, typer.Option(help="Number of tracks.")] = 8,
) -> None:
    """Write synthetic, copyright-free tracks to try tasteq without your own music."""
    from tasteq.synth import write_demo_library

    for path in write_demo_library(folder, tracks):
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
