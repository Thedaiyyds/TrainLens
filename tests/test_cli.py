import re

import pytest
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


@pytest.mark.parametrize("command", ["run", "list", "diff"])
def test_unimplemented_commands_are_rejected(command: str) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code != 0
    assert "No such command" in result.output