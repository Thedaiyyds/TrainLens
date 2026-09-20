# TrainLens v0.1 specification

Status: proposed for review; implementation has not started.

**Diff your PyTorch training runs.**

## Product positioning

TrainLens is an open-source, local CLI for recording and comparing PyTorch training runs. It shows what ran, in which environment, whether it finished, how long it took, and how CUDA peak memory changed.

The workflow requires no service, account, network connection, or training-script edits. Comparisons describe observed differences; they do not establish causality, statistical significance, or equivalent training quality.

v0.1 officially supports single-process Python/PyTorch training. Non-CUDA functionality must be developable and testable on macOS/Apple Silicon. Official CUDA validation happens on Linux with an NVIDIA GPU.

## User stories

- Record a named baseline and experiment by wrapping an existing Python training script.
- Compare runtime, CUDA allocator peaks, commands, environments, and Git state in a shareable Markdown report.
- Develop and test the core workflow without CUDA, with missing metrics clearly explained.
- Inspect a failed run's outcome and available metadata without mistaking missing data for zero.

## CLI behavior

```sh
trainlens run --name baseline -- python train.py --epochs 5
trainlens run --name experiment -- python train.py --epochs 5 --batch-size 64
trainlens list
trainlens diff baseline experiment
trainlens diff baseline experiment > comparison.md
```

### Run

- Require a name and the `--` separator; preserve the supplied Python command, script arguments, and invocation working directory.
- Execute through the selected Python environment, with training output visible and no training-script edits required.
- Assign a stable run ID. Names are unique within the local store; duplicate names must not overwrite existing runs.
- Save command, environment, timestamps, elapsed runtime, training outcome, and available CUDA peaks.
- Distinguish training failure from collection failure. A failed training command must not appear successful; optional metric failures must not turn successful training into a failed run.
- Report launch or storage failures clearly. Retain available metadata for failed or interrupted runs where possible, without promising collection after abrupt termination.

### List

Show locally saved runs, newest first, with ID, name, start time, status, runtime, and exit code. An empty store produces an empty result. Missing values are displayed as `N/A`.

### Diff and Markdown report

- Resolve the baseline and experiment by saved name or run ID. Missing or ambiguous selections produce a clear error.
- Emit Markdown to stdout so users can save it with shell redirection; keep diagnostics separate from the report.
- Include both runs' identities, commands, working directories, timestamps, outcomes, environment and Git differences, runtime, and CUDA peak allocated/reserved memory.
- Show numeric changes as `experiment - baseline`, with units. Percentage change is relative to the baseline; it is `N/A` for a zero baseline or missing value.
- Make missing metrics, incomplete runs, and environment differences visible. Compare compatible measurements; do not present unavailable or incomparable values as numeric deltas or declare an automatic winner.
- Read saved records only: listing and diffing do not rerun training or require PyTorch or CUDA on the reader's machine.

## Functional requirements

- **Local storage:** Save versioned JSON records under `.trainlens` in the invocation working directory. Each run has its own record; subsequent runs must not overwrite earlier results. No remote storage is required.
- **Runtime and outcome:** Record UTC timestamps, elapsed process wall time in seconds, status, and training exit code when known. Runtime includes bootstrap overhead; it is not isolated training-loop or CUDA kernel time. Unknown completion must not be represented as success.
- **Environment:** Record the training environment's Python version/executable, OS and architecture, PyTorch version, CUDA build version and availability, and available GPU information. CUDA build version and runtime availability are separate facts. Collect relevant metadata without dumping the entire environment.
- **Git:** Record the pre-run commit and dirty state when available. TrainLens's generated records must not themselves make the project appear dirty. Missing Git information does not prevent a run.
- **Instrumentation:** Use a bootstrap runner that executes the training script in the same Python process as TrainLens instrumentation. The CLI supervisor may remain separate. See [ADR 0001](adr/0001-bootstrap-instrumentation.md) for the architecture and technical rationale.
- **CUDA metrics:** Collect `torch.cuda.max_memory_allocated` and `torch.cuda.max_memory_reserved` from the training process and store integer bytes with device context. These are PyTorch allocator peaks, not total GPU usage or an OOM diagnosis. Preserve the measurement scope and any collection limitations.
- **Missing metrics:** Store unavailable values as null with a reason, and render them as `N/A`. Zero means a measured zero. Keep metric availability separate from training outcome; abrupt termination may leave metrics unavailable.
- **Non-CUDA behavior:** Recording, listing, and reporting work on macOS/Apple Silicon and Linux CPU environments without NVIDIA software. Do not substitute MPS or NVML usage for CUDA allocator peaks. TrainLens does not rewrite a CUDA-dependent training script to make it run on CPU.

## RunRecord data contract

The following information is required in the logical record; exact nesting and optional diagnostic fields can be settled during implementation. Unknown values remain explicit rather than fabricated.

| Group | Data |
| --- | --- |
| Identity | `schema_version` (initially 1), TrainLens version, stable run ID, user-supplied name |
| Execution | Original command as an argument array, Python executable, invocation working directory |
| Lifecycle | Start/end timestamps when known, elapsed runtime in seconds, status, training exit code when known |
| Environment | Python version, OS/architecture, PyTorch version, CUDA build version and availability, available GPU information |
| Git | Commit and dirty state, or an explanation of unavailable Git metadata |
| Metrics | Peak allocated/reserved bytes with device context, availability and reasons, measurement scope |
| Diagnostics | Collection warnings or errors sufficient to explain incomplete metadata and metrics |

Saved records must remain interpretable through `schema_version`. Reports use the recorded environment and measurements, not the current machine's state.

## Non-goals

- `torchrun`, DDP, FSDP, DeepSpeed, ZeRO, multi-process or multi-node training, worker aggregation, and official multi-GPU validation.
- Automatic tuning, recommendations, OOM diagnosis, memory-leak diagnosis, AI explanations, or modifying training code.
- A Web UI, dashboard, cloud service, telemetry, accounts, remote storage, or automatic report sharing.
- MPS/ROCm memory metrics, CPU memory profiling, GPU utilization sampling, NVML peak estimates, tracing, or per-step profiling.
- Loss/accuracy/throughput extraction, hyperparameter discovery, dependency lockfile capture, code/data snapshots, and reproducibility guarantees.
- Training-log persistence, run deletion/renaming, arbitrary interpreter invocation modes, and automatic recovery after supervisor failure.

## Acceptance criteria

1. On macOS/Apple Silicon, wrap two CPU PyTorch runs, save their JSON records, list them, and generate a Markdown comparison without NVIDIA software. CUDA metrics are explicitly unavailable.
2. The wrapped script receives its arguments and working directory correctly, runs in the selected Python environment, and keeps its output visible.
3. Successful and failing scripts record the correct training outcome. Optional collection failures retain usable records and explain missing values.
4. Saved records contain the required environment, Git, runtime, and identity data. Separate runs remain separate; duplicate names do not overwrite data.
5. A report correctly shows runtime and memory differences, handles missing values and zero baselines, and can be generated from saved records without PyTorch or CUDA installed.
6. On Linux + NVIDIA, a controlled single-process run produces peaks matching direct in-process PyTorch API readings at the same measurement boundary. Allocate and release tensors to verify that peaks are retained, and use a fresh run to verify that peaks do not carry over.

## v0.1 definition of done

- The run/list/diff workflow and Markdown report meet the acceptance criteria, with automated non-CUDA tests on macOS arm64 and Linux CPU.
- Real Linux + NVIDIA validation is completed and its environment recorded; mocked CUDA tests alone are insufficient.
- Installation, local storage, supported usage, metric meanings, unavailable values, and bootstrap limitations are documented.
- The tested Python/PyTorch/OS compatibility matrix is published, and excluded features are not advertised as supported.

## Deferred details and review decisions

Name syntax, numeric CLI error codes, signal handling details, GPU pairing, alternate allocator policies, uncommon Git states, additional invocation forms, and concurrency/crash-recovery mechanisms are not frozen by this spec.

Review the proposed cwd-local store, unique names, and Markdown-on-stdout workflow, and choose the tested Python/PyTorch/OS versions before release. ADR 0001 remains unchanged and proposed; its detailed policies require a later review rather than becoming additional product acceptance criteria through this reference.
