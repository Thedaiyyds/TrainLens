"""Command-line entry point for TrainLens."""

from pathlib import Path

import typer
from typer.core import TyperCommand

from trainlens.storage import RunStore

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Diff your PyTorch training runs."""


class _RunCommand(TyperCommand):
    """Keep the literal separator and prevent parsing any training arguments."""

    def parse_args(self, ctx, args):
        if "--" in args:
            separator = args.index("--")
            ctx.meta["training_command"] = args[separator + 1 :]
            args = args[:separator]
        elif "--help" not in args:
            raise typer.BadParameter(
                "Use -- before PYTHON SCRIPT.py [SCRIPT_ARGS...]", ctx=ctx
            )
        return super().parse_args(ctx, args)


@app.command("run", cls=_RunCommand)
def run(ctx: typer.Context, name: str = typer.Option(..., "--name")) -> None:
    """Record a run: trainlens run --name NAME -- PYTHON SCRIPT.py [ARGS...]."""
    from trainlens.supervisor import RunStorageError, supervise_run

    try:
        record = supervise_run(
            name, ctx.meta.get("training_command", []), invocation_cwd=Path.cwd()
        )
    except (RunStorageError, OSError, ValueError) as error:
        typer.echo(f"TrainLens: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"TrainLens: saved {record.name!r} ({record.run_id}): {record.status}", err=True
    )
    if record.diagnostics.failure_message:
        typer.echo(f"TrainLens: {record.diagnostics.failure_message}", err=True)
    for warning in record.diagnostics.collection_warnings:
        typer.echo(f"TrainLens: {warning}", err=True)
    raise typer.Exit(0 if record.status == "succeeded" else 1)


def _cell(value: object) -> str:
    if value is None:
        return "N/A"
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


@app.command("list")
def list_runs() -> None:
    """Show saved runs in the current directory, newest first."""
    try:
        records = RunStore().list_records()
    except (OSError, ValueError) as error:
        typer.echo(f"TrainLens: could not list runs: {error}", err=True)
        raise typer.Exit(1) from error
    rows = [["ID", "NAME", "STARTED", "STATUS", "RUNTIME", "EXIT CODE"]]
    for record in records:
        runtime = (
            None if record.runtime_seconds is None else f"{record.runtime_seconds} s"
        )
        rows.append(
            [
                _cell(value)
                for value in (
                    record.run_id,
                    record.name,
                    record.started_at,
                    record.status,
                    runtime,
                    record.exit_code,
                )
            ]
        )
    widths = [max(map(len, column)) for column in zip(*rows)]
    for row in rows:
        typer.echo(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        )
