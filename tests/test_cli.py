from typer.testing import CliRunner

from tasteq import __version__
from tasteq.cli import app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__
