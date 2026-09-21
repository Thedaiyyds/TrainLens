# TrainLens

**Diff your PyTorch training runs.**

TrainLens is a local CLI for comparing PyTorch training runs. The current project
includes the package, CLI skeleton, RunRecord model, and local RunStore.
`trainlens --help` works;
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

## Local run storage

Given an existing `RunRecord` named `record`, use the Python API:

```python
from trainlens.storage import RunStore

store = RunStore()  # Captures the current working directory.
path = store.save(record)
loaded = store.load(record.run_id)
```

`RunStore(project_dir)` can instead receive an explicit project directory. Save
creates a separate UTF-8 JSON record under `.trainlens` there, preserving all model
fields, including `None` as `null`. Load restores nested dataclasses. Use the store
API rather than depending on its internal file layout.

In the current implementation, save creates new records; duplicate IDs or names
raise `FileExistsError`.
Missing runs (including an absent store) raise `FileNotFoundError` without creating
directories. Malformed JSON raises `json.JSONDecodeError`; invalid record data,
unsafe paths, and unknown schema versions raise `ValueError`. Only schema 1 is
readable; files are never automatically upgraded or rewritten. Other filesystem
errors propagate as `OSError`. Corrupt existing records also block name checks
during save rather than being silently ignored.

The current Task implementation supports one writer at a time and has no update
operation. These are implementation limitations, not stable public contracts.
The internal file layout is also an implementation detail; concurrency and
crash-recovery mechanisms remain deferred by the [specification](docs/spec-v0.1.md).
Ordinary write failures clean up the newly created file; abrupt process termination
can leave an incomplete file. This API does not run training or add CLI commands.

## Metadata collection

Collectors reuse the existing models and do not execute training:

```python
from trainlens.collectors.environment import (
    collect_environment,
    collect_python_executable,
)
from trainlens.collectors.git import collect_git
from trainlens.models import RunDiagnostics

diagnostics = RunDiagnostics()
environment = collect_environment(diagnostics=diagnostics)
python_executable = collect_python_executable()  # RunRecord.python_executable
git = collect_git(invocation_cwd)  # The caller supplies the original project cwd.
```

Environment helpers observe their calling interpreter. The future bootstrap must
call them in the selected training process. PyTorch is imported only when collecting;
it is not an installation dependency. Missing imports and query failures preserve
unknown fields as `None` and append collection warnings, without changing training
failure information. A PyTorch build without CUDA records CUDA availability as
false and an empty CUDA GPU inventory.

For CUDA builds, availability and GPU queries are deferred until PyTorch CUDA is
already initialized. Before that, these fields remain unknown with a warning; a
later call after script execution can collect them. No helper initializes CUDA,
allocates tensors, or reads/resets allocator peaks. Device-name query failures keep
the known device identifier with an unknown name.

Git collection uses the supplied cwd, records the commit and whole-worktree dirty
state (including staged and untracked changes), and excludes that cwd's `.trainlens`
subtree. It does not edit Git configuration or ignore files. Missing Git or command
failures preserve known fields and explain unavailable values through `GitInfo`.
