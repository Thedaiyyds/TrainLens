# ADR 0001: Collect CUDA allocator peaks inside a bootstrap training process

- Status: proposed
- Date: 2026-09-20
- Scope: TrainLens v0.1, single-process Python/PyTorch training
- Related specification: [spec-v0.1.md](../spec-v0.1.md)

## Context

TrainLens must run an existing Python training script and record two specific metrics: `torch.cuda.max_memory_allocated` and `torch.cuda.max_memory_reserved`. It must also remain usable and testable without CUDA, including on macOS Apple Silicon.

PyTorch's allocated peak measures the largest amount of device memory occupied by tensors through its allocator on a device. The reserved peak measures the allocator's maximum managed memory, including cached capacity. Values are bytes; their default history begins with the process, and resetting peak statistics changes that history. These are per-device allocator counters, not generic measurements of all GPU memory used by a program. See PyTorch's [allocator implementation and API definitions](https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py) and [reserved-peak API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_reserved.html).

PyTorch distinguishes memory occupied by tensors from cached memory still visible in `nvidia-smi`. Allocations made directly through CUDA or other libraries can also fall outside PyTorch's allocator accounting. Consequently, neither peak should be labeled total process GPU usage, and their difference is not an OOM or fragmentation diagnosis. See [CUDA memory management](https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management) and [allocator visibility limitations](https://docs.pytorch.org/docs/main/torch_cuda_memory).

### Why an external parent cannot supply these metrics alone

Each Python process has its own PyTorch allocator state. Calling a peak API in the CLI parent queries that parent's state; it does not query the training child. These APIs accept a device, not a target PID. A parent that allocates nothing could observe zero while the child trains, and cannot retrieve the child's lost counter history after it exits. This is an architectural consequence of the [PyTorch API implementation](https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py).

The parent can reliably observe command execution, elapsed wall time, and process exit. To obtain allocator counters, it needs the training process to read and transmit them before termination. Being the child's parent, sharing a GPU, or selecting the same CUDA device does not give access to that allocator's counters.

### Why NVML or nvidia-smi is not an equivalent substitute

NVML exposes a process's used GPU memory, rather than separate PyTorch allocated/reserved peak counters. Its documented `usedGpuMemory` field represents used GPU bytes for a process. See [NVIDIA's process-information API](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlProcessInfo__t.html).

An external polling design would measure a different quantity. The maximum of samples can also miss an allocation and release between polls. Increasing polling frequency cannot turn device/process usage into PyTorch's allocator high-water counters. NVML could support separately named observability metrics in a future release, but it cannot fulfill these two v0.1 requirements. It is not a mandatory dependency or a fallback for missing allocator peaks.

## Decision

Use a TrainLens bootstrap runner that executes the user's script **in the same Python process as TrainLens instrumentation**. The CLI supervisor may be a separate parent process. "Same process" means the bootstrap, imported PyTorch, and training script share the child's interpreter and allocator state; it does not mean executing training inside the long-lived CLI parent.

```text
TrainLens CLI supervisor
  |-- reserves run identity and writes initial record
  |-- starts selected Python interpreter
  |     `-- TrainLens bootstrap
  |           |-- establishes collection and execution context
  |           |-- executes user's script as __main__
  |           `-- reads allocator peaks and sends structured result
  `-- observes child exit and finalizes canonical RunRecord
```

The parent owns storage and lifecycle outcome. The child owns Python/PyTorch/CUDA environment observations and allocator measurements. Use an internal payload channel or atomic child result file separate from stdout/stderr; the transport is an implementation detail. Associate every payload with the reserved run ID and validate its schema before merging.

### Bootstrap lifecycle and execution compatibility

1. The supervisor resolves the requested interpreter and script, records the original argv/cwd and pre-run Git state, reserves the name, and writes the initial record.
2. Start a fresh process using the requested interpreter with TrainLens available in that environment. Record the actual launch argv separately from the user's command. Never launch the training script as a second subprocess from inside the bootstrap: that would lose the shared allocator state this decision requires.
3. Establish exception-safe finalization before script entry and collect startup metadata. PyTorch import/probe failures are isolated collection errors; do not replace the script's own import behavior with a fabricated training outcome. Send startup metadata before executing the script when available, so it can survive a later fatal exit.
4. Set the script execution context: argv, `__main__`, `__file__`, cwd, and script-directory import search behavior. A `runpy.run_path(..., run_name="__main__")`-based implementation is a candidate, with explicit argv/path setup and compatibility tests. Python documents that `runpy` executes within the current process; it is not a sandbox. See [Python runpy documentation](https://docs.python.org/3/library/runpy.html).
5. On normal return, `SystemExit`, or an exception unwinding through the bootstrap, query the supported CUDA allocator counters before the process exits, and send the final metric payload. Preserve original exception output and exit semantics; collection failure must not replace the training error.
6. The supervisor waits for child termination, records monotonic elapsed process time and exit outcome, and atomically finalizes the record. A received metric payload is not proof that process shutdown later succeeded.

Transparent direct-Python equivalence is not promised. Pre-importing PyTorch changes import timing and can affect scripts that configure environment variables before importing it. Require allocator/visibility configuration to be set before launching TrainLens. Bootstrap frames can appear in tracebacks, and interpreter startup/import costs are included in runtime. Validate the supported script behavior explicitly; defer interpreter flags, module execution, multiprocessing, and custom launchers.

### Measurement boundary and allocator policy

Use a fresh child for each run. TrainLens instrumentation must not allocate CUDA tensors. Rely on the fresh process's peak counters rather than forcing CUDA initialization and resetting every visible device before script entry. The scope identifier is `bootstrap_to_script_exit_v1`: allocator activity from process startup through the final query after script return or exception unwinding. It includes script imports, model/data setup, and cleanup executed before that query.

Do not call `empty_cache`, inject device synchronization, or reset peaks during training. Such actions change the workload or the counter history. User code that resets peaks can make the final API value describe only a suffix of the run; v0.1 does not intercept every reset or reconstruct earlier history. Document those scripts as outside the guaranteed metric scope.

Capture each device separately with its process-local ordinal and available stable identity. Do not sum device peaks and call that a simultaneous peak. Do not infer that a zero-valued device counter means an unavailable GPU. If CUDA is available but the script never initializes it, report `not_used` without forcing initialization just to manufacture measurements. Defer GPU property queries that initialize PyTorch CUDA until after execution, checking the script's initialization state first. Guard all queries and preserve unknown device metadata explicitly.

The official v0.1 metric contract is validated with the native caching allocator. Other backends, disabled caching, and custom allocators receive `unsupported` metrics. This conservative scope avoids treating different allocator behavior as equivalent; for example, PyTorch documents distinct reserved-peak semantics under `cudaMallocAsync`. See the [reserved-peak API notes](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_reserved.html). Record the backend, relevant configuration, and PyTorch version to support interpretation and future expansion.

### Finalization limits and non-CUDA behavior

The final query precedes later `atexit` callbacks and interpreter teardown. Training must finish before the script returns; training continuing in background threads is outside the measured scope. A normal exception can still permit metric collection, but SIGKILL, `os._exit`, interpreter crashes, and fatal device failures may bypass it. A `finally` block is a best-effort collection boundary, not a guarantee against process termination. An `atexit`-only design would have both ordering problems and the same fatal-exit limitations.

The surviving supervisor can still record an exit and runtime when a final payload is missing. Mark memory as unavailable with `finalization_missing`, and keep exit outcome distinct from metric completeness. If the supervisor also dies, preserve the last valid record without claiming completion. Do not invent peaks from zero, a previous run, or an NVML sample.

Without CUDA, preserve the full local run/list/diff/report workflow. CUDA absence is a recorded capability state, not a TrainLens failure. Missing PyTorch, no usable CUDA device, failed queries, unused CUDA, and missing finalization have separate reasons. Do not substitute Apple MPS memory for CUDA allocator metrics. A user script's CUDA requirement can still make that script fail normally.

## Alternatives considered

| Alternative | Assessment for v0.1 |
| --- | --- |
| External parent calls PyTorch peak APIs | Measures the parent's allocator; cannot satisfy child-process peak requirements. |
| Poll NVML/nvidia-smi | Useful for different device/process usage metrics, but not the required allocator peaks; polling can miss transients. |
| Ask users to add TrainLens calls to training code | Can read the right counters but conflicts with the script-wrapping workflow and relies on users placing lifecycle calls correctly. |
| Inject through `sitecustomize` or global import hooks | Can reach the target process, but introduces startup-path coupling, interference with user hooks, and unintended child-process propagation. Defer this complexity. |
| Run training inside the CLI process | Shares allocator state but mixes CLI dependencies, interpreter choice, state across runs, and training failure handling. Prefer a fresh instrumented child. |
| Bootstrap that launches an uninstrumented training subprocess | Retains the same measurement gap as parent-only collection and is rejected. |

## Consequences

- The design obtains allocator-native peaks without training-loop edits or high-frequency GPU polling.
- Parent lifecycle recording remains useful when optional instrumentation fails. Non-CUDA development does not need NVIDIA dependencies.
- The selected interpreter needs TrainLens installed. Packaging and supported Python/PyTorch versions must be validated before release.
- Bootstrap overhead and execution-context changes must be documented and tested. This is process-level comparison, not an isolated benchmark harness.
- Whole-run peak correctness depends on the documented scope: no user peak resets, allocator replacement, training subprocesses, or post-return training work.
- No mechanism here diagnoses OOMs, tunes memory, aggregates distributed workers, or guarantees collection after fatal termination.

## Validation required before acceptance

Use the specification's acceptance criteria as release gates. On macOS arm64 and Linux CPU, verify script execution semantics, lifecycle/storage failures, missing-metric handling, and deterministic Markdown comparison. Test known unsupported launchers without claiming detection of every distributed script.

On Linux + NVIDIA, compare collected values to direct API readings made inside a controlled training fixture at the same boundary. Allocate and release tensors to verify peaks persist, vary allocation size across fresh runs, test exception finalization, and verify explicit missing data on abrupt exit. Use a single process, one GPU, and the native allocator for official v0.1 evidence. CUDA mocks alone do not accept this ADR's measurement claim.

The exact Python/PyTorch/OS version matrix and internal payload transport remain implementation decisions. Any expansion of allocator, invocation, or process support requires additional validation rather than silently broadening this ADR.
