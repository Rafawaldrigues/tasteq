"""Command-line interface. Subcommands are added as each stage lands."""

from __future__ import annotations

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


if __name__ == "__main__":
    app()
