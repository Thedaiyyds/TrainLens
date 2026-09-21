# TrainLens

**Diff your PyTorch training runs.**

TrainLens is a local CLI for comparing PyTorch training runs. The current project
contains only the installable package and CLI skeleton. `trainlens --help` works;
run execution, listing, comparison, and CUDA instrumentation are not implemented.

## Development

The package currently requires Python 3.10 or newer. This is the skeleton's Python
requirement, not a validated PyTorch/CUDA compatibility matrix for v0.1.
Development of this skeleton requires no PyTorch or NVIDIA software.

From the repository root on macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
trainlens --help
pytest
ruff check .
```

The virtual environment keeps dependencies separate from your system Python.
Editable installation (`-e`) makes source edits available without reinstalling.
The `dev` extra installs pytest for automated tests and Ruff for code checks;
Typer is the only direct runtime dependency. Setuptools builds the package.
Run the same checks after changes. Optionally check formatting with
`ruff format --check .`.

Source code lives in `src/trainlens`; tests exercise the installed package rather
than relying on the repository root being importable. Package metadata, the
`trainlens` command entry point, and tool settings live in `pyproject.toml`.

Read [AGENTS.md](AGENTS.md), the [v0.1 specification](docs/spec-v0.1.md), and
[ADR 0001](docs/adr/0001-bootstrap-instrumentation.md) before implementation.
