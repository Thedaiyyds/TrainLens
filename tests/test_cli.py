import re

from typer.testing import CliRunner

from trainlens.cli import app

runner = CliRunner()

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def test_help() -> None:
    result = runner.invoke(app, ["--help"], prog_name="trainlens")
    output = ANSI_ESCAPE.sub("", result.output)

    assert result.exit_code == 0
    assert "trainlens" in output
    assert "Diff your PyTorch training runs." in output
    assert "--help" in output


def test_unknown_command_is_rejected() -> None:
    result = runner.invoke(app, ["not-a-command"])

    assert result.exit_code != 0
    assert "No such command" in result.output
