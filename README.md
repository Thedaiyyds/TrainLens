# TrainLens

**Diff your PyTorch training runs.**

TrainLens is a local CLI for comparing PyTorch training runs. The
`trainlens run` workflow records script execution and metadata locally, and
`trainlens list` displays saved runs. `trainlens diff` compares saved records in
Markdown. The training child now includes CUDA allocator peak instrumentation.
Real Linux + NVIDIA validation has **not** been completed: CUDA correctness is not
yet accepted, and v0.1 is not complete.

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
a monotonic clock. The child collects startup environment metadata, then refreshes
it and queries allocator peaks after the script returns or unwinds. Useful startup
fields survive a failed or partial refresh. Private temporary JSON files distinguish
startup metadata from final observations and are removed after the child exits;
they are not part of the persisted record format. PyTorch may be
imported before the script, so this is not transparent direct-Python equivalence.

After the child exits, TrainLens saves one final RunRecord under the invocation
cwd's `.trainlens`, including original command, UTC timestamps, runtime, observed
exit code, outcome, available metadata, and diagnostics. CUDA peaks are per-device
byte values when available, otherwise explicitly missing with a reason.
Nonzero script exits and tracebacks remain visible
and produce failed records when storage works. Missing child metadata stays unknown
with a warning; it does not change the observed child exit outcome.

Obvious duplicate names and storage failures are checked before execution. Save
rechecks duplicates and never replaces an existing record. A storage failure after
training is reported as a recording error and makes the CLI return non-success.
These checks do not reserve names against concurrent writers. Initial record
persistence, record updates, signal-forwarding policy, and crash recovery are not
implemented; abrupt supervisor termination can leave no saved record.

## List saved runs

```sh
trainlens list
```

Reads the current directory's saved runs and prints ID, name, start time, status,
runtime in seconds, and exit code to stdout. Runs with known start times appear
newest first, followed by unknown start times. Missing values display as `N/A`;
measured zero remains zero. An empty store produces only the table header and
does not create storage directories.

Listing reads historical records only: it does not execute training, collect fresh
metadata, or require PyTorch/CUDA. Invalid records produce a diagnostic on stderr
and a nonzero exit, leaving the files untouched. Table spacing is for human reading,
not a stable machine-readable format.

## Compare saved runs

```sh
trainlens diff baseline experiment
trainlens diff baseline experiment > comparison.md
```

The first selector is the baseline; the second is the experiment. Each must exactly
match a saved name or run ID. Missing or ambiguous selectors fail clearly, including
when a name matches another run's ID. Errors go to stderr; stdout contains only the
Markdown report, so shell redirection produces a clean report file.

The report shows identities, original argument arrays, working directories,
lifecycle, environment, Git state, runtime, saved CUDA peaks, and diagnostics.
Delta means `experiment - baseline`; percentage means `delta / baseline * 100`.
A zero baseline permits a delta but has `N/A` percentage. Missing measurements
produce `N/A` comparisons, and measured zeros remain numeric. Runtime is process
wall time including bootstrap overhead, not isolated training-loop time.

CUDA rows use exact saved device identifiers, without guessing hardware pairing or
summing device peaks. Numeric comparison requires both values and equal, known
measurement scopes. Different or unknown scopes leave values visible with `N/A`
comparisons and an explanation. Historical records with `not_collected` metrics
remain readable alongside new records.

Reports use saved data only: no training is rerun, no current environment or Git
state is queried, and the reader needs no PyTorch/CUDA. Historical warnings and
training failures belong in the report, not stderr. Bad records fail without being
rewritten. The report describes observations and does not declare a winner; its
Markdown layout is not a stable machine-readable API.

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

For CUDA builds, startup collection defers availability and GPU queries. The
post-script refresh queries them only if PyTorch CUDA is already initialized;
otherwise these fields remain unknown with a warning. The environment helper does not initialize
CUDA, allocate tensors, or read/reset allocator peaks. Device-name query failures keep
the known device identifier with an unknown name.

Git collection uses the supplied cwd, records the commit and whole-worktree dirty
state (including staged and untracked changes), and excludes that cwd's `.trainlens`
subtree. It does not edit Git configuration or ignore files. Missing Git or command
failures preserve known fields and explain unavailable values through `GitInfo`.

## CUDA allocator observations

After script execution, the same child Python process calls
`torch.cuda.max_memory_allocated(device)` and
`torch.cuda.max_memory_reserved(device)`. These are PyTorch allocator high-water
marks in integer bytes, recorded separately for each visible `cuda:N` device.
Devices are neither summed nor paired by guessed hardware identity. An actual API
result of zero stays numeric zero. A failed query leaves that value `null`, retains
any successful companion value, and records a reason and collection warning.

The measurement scope is `bootstrap_to_script_exit_v1`: the fresh child's allocator
history through the final query after the script returns or unwinds through
`SystemExit` or a Python exception. TrainLens does not reset peak counters, clear
the cache, synchronize CUDA, or allocate a tensor to probe it. A script that resets
its own counters changes the history exposed by the APIs; TrainLens does not undo
or detect those resets. Queries precede interpreter shutdown and `atexit` handlers.

The collector first checks for a CUDA build and `torch.cuda.is_initialized()`.
Without PyTorch, a CUDA build, or initialized CUDA, metrics remain unavailable with
distinct reasons. It does not initialize CUDA just to measure it, and does not
substitute MPS, ROCm, NVML, or `nvidia-smi` measurements. These peaks are not total
GPU usage. Alternate/custom allocators and `cudaMallocAsync` equivalence have not
been validated; this implementation observes the requested APIs without establishing
a backend compatibility policy.

Final collection is best effort and does not change the script's exit outcome.
`os._exit`, SIGKILL, or an interpreter crash may bypass finalization. If only startup
metadata reaches the parent, it is retained with `finalization_missing` metrics and
a warning. Missing or malformed payloads leave metadata unknown with a warning;
the parent never queries CUDA to fill the gap. Abrupt supervisor failure can still
prevent persistence entirely.

Current automated coverage uses CPU tests, fake PyTorch APIs, and real Python
subprocesses. **Real Linux + NVIDIA validation: NOT RUN.** The follow-up hardware
validation remains a release gate; mocks do not establish real allocator correctness.

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
