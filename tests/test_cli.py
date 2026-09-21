import pytest
from typer.testing import CliRunner

from trainlens.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"], prog_name="trainlens")

    assert result.exit_code == 0
    assert "trainlens" in result.output
    assert "Diff your PyTorch training runs." in result.output
    assert "--help" in result.output


@pytest.mark.parametrize("command", ["run", "list", "diff"])
def test_unimplemented_commands_are_rejected(command: str) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code != 0
    assert "No such command" in result.output
