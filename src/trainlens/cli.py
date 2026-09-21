"""Command-line entry point for TrainLens."""

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Diff your PyTorch training runs."""
