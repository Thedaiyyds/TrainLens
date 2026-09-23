# TrainLens

**Diff your PyTorch training runs.**

TrainLens is a local CLI for comparing PyTorch training runs. The initial non-CUDA
`trainlens run` workflow records script execution and metadata locally. CUDA peak
memory instrumentation, listing, comparison, and reports remain future work;
this is not the complete v0.1 workflow.

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

## Record a script run

```sh
trainlens run --name baseline -- python train.py --epochs 5
trainlens run --name experiment -- .venv/bin/python scripts/train.py "hello world" ""
```

The name and `--` separator are required. Everything after `--` is the original
Python command, with argument boundaries preserved. This first workflow accepts
`python`, `python3`, versioned Python names, or paths to those executables, followed
by a `.py` script. Interpreter flags, `-m`, `-c`, stdin programs, arbitrary commands,
shell wrappers, and distributed launchers are unsupported.

The selected Python must have TrainLens importable. It runs a fresh TrainLens
child which executes the script in that same process; TrainLens does not substitute
the parent's interpreter. The script starts in the invocation cwd, can import
sibling modules, and inherits stdin/stdout/stderr. TrainLens diagnostics go to
stderr, leaving stdout for the script.

The parent collects Git state before execution and measures child wall time using
a monotonic clock. The child collects startup environment metadata with the existing
collector. Private temporary JSON files carry that metadata and are removed after
the child exits; they are not part of the persisted record format. PyTorch may be
imported before the script, so this is not transparent direct-Python equivalence.

After the child exits, TrainLens saves one final RunRecord under the invocation
cwd's `.trainlens`, including original command, UTC timestamps, runtime, observed
exit code, outcome, available metadata, and diagnostics. CUDA peaks remain explicitly
`not_collected`, never fake zeros. Nonzero script exits and tracebacks remain visible
and produce failed records when storage works. Missing child metadata stays unknown
with a warning; it does not change the observed child exit outcome.

Obvious duplicate names and storage failures are checked before execution. Save
rechecks duplicates and never replaces an existing record. A storage failure after
training is reported as a recording error and makes the CLI return non-success.
These checks do not reserve names against concurrent writers. Initial record
persistence, record updates, signal-forwarding policy, and crash recovery are not
implemented; abrupt supervisor termination can leave no saved record.

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

Environment helpers observe their calling interpreter. The internal bootstrap child
calls them in the selected training process. PyTorch is imported only when collecting;
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

## Internal bootstrap foundation

`trainlens.bootstrap.execute_script(script_path, args, invocation_cwd=...)`
executes a `.py` script as `__main__` in the calling interpreter. Relative script
paths are resolved against the supplied invocation cwd. Arguments remain separate
strings, the script receives `__file__`, and its directory is available for sibling
imports without becoming the working directory. The script may change cwd itself.

The function returns `None` on normal completion and propagates `SystemExit` and
user exceptions. It leaves stdin/stdout/stderr attached as supplied by the caller.
Its temporary argv, import path, and cwd changes are restored when execution
unwinds. This is an internal primitive for a one-shot bootstrap process, not an
isolation sandbox or a facility for concurrent runs; imported modules and other
script side effects remain in the process.

This execution primitive does not launch another interpreter, collect metadata or
CUDA peaks, or persist records. The run supervisor launches the selected interpreter's
private child entry point, which calls this primitive directly.
