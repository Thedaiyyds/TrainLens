# NVIDIA CUDA correctness validation

**Implemented:** the fixtures, runner, and CPU/mock infrastructure tests.
**Pending:** real Linux + NVIDIA execution. No real-GPU acceptance evidence is
included in this change, and TrainLens v0.1 is not yet accepted.

## Run on a real machine

Use a native Linux machine with an NVIDIA GPU, a working driver, and an existing
CUDA-enabled PyTorch environment. Choose a GPU with room for the default 64 MiB
allocation plus interpreter/context overhead. Install TrainLens into that same
Python environment from the checkout to be validated:

```sh
python -m pip install -e '.[dev]'
python validation/cuda/run_validation.py --output /tmp/trainlens-cuda-evidence.json
```

Run from the repository root. The output file must not already exist; choose a new
path outside `.trainlens` for each validation attempt. The runner checks that the
installed TrainLens source matches this checkout before probing CUDA.
PyTorch and the NVIDIA driver are not installed
or changed by the runner. Use PyTorch's [installation instructions](https://pytorch.org/get-started/locally/)
to prepare an appropriate environment separately.

The runner reports OS, architecture, Python, PyTorch, CUDA build, CUDA availability,
visible device count, the `cuda:0` GPU name, driver version, allocator backend,
TrainLens version/commit, worktree status, and relevant allocator/visibility
environment variables. `nvidia-smi` is used only for driver-version diagnostics;
its memory readings are never used. This suite uses the native caching allocator
and enabled caching, as the controlled evidence scope described in proposed
[ADR 0001](../../docs/adr/0001-bootstrap-instrumentation.md). An unknown/alternate
backend, disabled caching, non-Linux platform, missing CUDA, or incomplete
driver/commit evidence produces SKIP, not an official PASS. This restriction does
not add a production allocator compatibility policy.

To adjust allocation sizes or startup timeout:

```sh
python validation/cuda/run_validation.py --large-mib 128 --small-mib 16 --timeout 180
```

Sizes must satisfy `large > small > 0`. They use MiB (1,048,576 bytes).
`CUDA_VISIBLE_DEVICES`, if needed, must be set before launching the runner.
The selected device is always the process-local `cuda:0` after that visibility
mapping. The runner does not modify CUDA configuration.

## What is compared

Each workload runs via the public TrainLens CLI, using the runner's Python for
both CLI and selected training interpreter. Each CLI invocation starts a fresh
bootstrap child and executes its fixture there. The reader loads the persisted
RunRecord through `RunStore`; it never invokes the internal CUDA collector.

`allocate_peak.py` creates a `torch.uint8` tensor on `cuda:0`, records direct API
readings while it is alive, deletes it, and records readings again. The direct
JSON is independent of TrainLens transport/storage. There is no subsequent CUDA
allocation between those final readings and bootstrap finalization; ordinary
JSON writing and exception formatting use CPU memory. TrainLens queries after
script unwinding, before interpreter teardown, under the existing
`bootstrap_to_script_exit_v1` scope.

| Scenario | Required result |
| --- | --- |
| Normal allocation | Saved allocated/reserved peaks exactly equal direct after-free readings, in integer bytes, with the expected scope and successful outcome. |
| Peak after free | Direct peaks stay unchanged after deletion; current allocated bytes decrease by at least the requested size. Saved peaks exactly match the retained peaks. |
| Fresh-process isolation | Large and small allocations use distinct run IDs and child PIDs. The smaller fresh run has a lower allocated peak and both records match their own references. Reserved peaks need not differ because allocation granularity can differ. |
| Python exception | `failure_after_allocate.py` saves reference readings, then raises the intentional RuntimeError. CLI fails, saved training exit is 1, traceback/failure remain visible, and final peaks exactly match. |
| Abrupt exit | `abrupt_exit.py` allocates then calls `os._exit(7)`. CLI fails; saved training exit is 7, metrics have no device values and reason `finalization_missing`. No reference peak is claimed. |

Equality has **no tolerance**. Missing values are never converted to zero.
Neither fixtures nor runner reset peaks, clear the cache, synchronize CUDA, sum
device peaks, or use NVML/MPS/ROCm substitutions. This is correctness validation,
not a benchmark. Preflight may initialize CUDA in the runner's own process;
that state cannot supply the fresh training child's allocator peaks.

## Results and storage safety

Every scenario reports PASS, FAIL, or SKIP. Overall exit codes are 0 for all checks
passing, 1 for a failed check/preflight, and 2 for skipped validation. A skip prints
`REAL NVIDIA VALIDATION NOT RUN / SKIPPED`. A preflight PASS alone is not a suite
PASS. The runner continues independent scenarios after a failure.

Runs and reference files use newly created temporary working directories. The
developer's current `.trainlens` store is never read, written, or deleted.
Temporary directories are removed after the suite. On a CLI timeout, only its
newly created process group is terminated, including its bootstrap child.
The optional evidence JSON retains environment diagnostics, scenario outcomes,
complete saved-record snapshots, references, and captured CLI stdout/stderr even
after temporary storage is removed. Preserve this file for human review.

For release evidence, execute a committed checkout, inspect the recorded commit
and worktree status, and retain OS/GPU/driver/PyTorch/CUDA/backend information with
the results. A suite PASS describes only these controlled checks on that recorded
environment. Compatibility-matrix selection and v0.1 acceptance remain separate
human review steps.

## Ordinary CPU tests

```sh
pytest tests/test_cuda_validation.py
ruff check .
```

Tests explicitly labeled CPU/mock check parsing, comparisons, missing data,
scope/outcome handling, preflight gates, aggregation, safe imports, and a complete
CLI/subprocess/storage path with fake PyTorch in an isolated test environment.
They do not run on NVIDIA hardware and cannot replace the real-machine command.
Importing validation modules does not import PyTorch or initialize CUDA.
