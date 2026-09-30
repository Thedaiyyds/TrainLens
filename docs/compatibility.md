# Compatibility and validation evidence

Python `>=3.10` is a package installation requirement, not evidence that every
Python/PyTorch/OS combination has passed. The version remains `0.1.0.dev0`.
v0.1 acceptance is pending. SPEC and ADR 0001 remain proposed.

Status and execution type are separate:

- **Verified:** the stated check actually passed in the stated environment.
- **Automated:** repeatable scripted checks; CI results need a successful job link.
- **Manually tested / local:** a developer invoked checks locally; no recurring
  macOS CI or general platform guarantee is implied.
- **Pending:** no qualifying evidence yet; configured checks alone are not a PASS.
- **Not supported / out of scope:** excluded from v0.1, regardless of mocks.

Evidence inspected on 2026-09-30. Task #12's implementation is `fd2c2fd`;
the Linux job below records that commit, and local checks exercised the same code.

| Environment / Python | PyTorch | CUDA / GPU | Validation type | Status and evidence |
| --- | --- | --- | --- | --- |
| macOS 26.6.2, arm64 / CPython 3.13.15 | Absent | Unavailable / no NVIDIA validation | Manually executed local automated checks | Verified: full pytest (326 passed), Ruff, 16 focused packaging unit cases. CUDA unit cases use mocks, not GPU evidence. |
| macOS 26.6.2, arm64 / CPython 3.13.15 | Absent in fresh venv | Explicit unavailable metrics / no NVIDIA | Manually executed local installed-wheel smoke | Verified: `python validation/packaging/run_smoke.py` returned PASS; wheel inspection and fresh-venv help/run/list/diff completed. |
| GitHub `ubuntu-latest` Linux runner / Python 3.10 job | Not required; CUDA tests mocked | No real CUDA evidence | Automated CI, editable install | Verified at main `9ad30ba`: [successful CI job](https://github.com/Thedaiyyds/TrainLens/actions/runs/36712488558/job/109877356480). That job predates packaging smoke. Exact patch version/runner image must be taken from job logs, not inferred from the selector. |
| GitHub Linux x86_64, kernel 6.17.0-1022-azure, glibc 2.39 / CPython 3.10.21 | Absent in smoke venv by assertion | Explicit unavailable metrics / no GPU exercise | Automated CI, editable tests and installed-wheel smoke | Verified at `fd2c2fd`: [successful job and evidence log](https://github.com/Thedaiyyds/TrainLens/actions/runs/36716866195/job/109891816784), 326 pytest cases, Ruff, wheel inspection, fresh-venv install and console help/run/list/diff PASS. |
| macOS arm64 and Linux CPU / selected Python + actual CPU PyTorch versions | Present | No CUDA required | SPEC acceptance: two real CPU PyTorch training runs and report | Pending. A no-PyTorch script or fake torch module does not establish this compatibility. |
| Linux / Python and PyTorch versions to be selected | CUDA-enabled PyTorch | One visible NVIDIA GPU, native allocator | Real [CUDA validation runner](../validation/cuda/README.md) | **NOT RUN / PENDING.** Record OS, GPU, driver, Python, PyTorch, CUDA build, backend, commit, and scenario results. |
| Other Python versions, OS images, or allocator configurations | Unspecified | Unspecified | Not exercised here | Pending; no support guarantee from package metadata or a pure-Python wheel tag. |
| Distributed/multi-node training, official multi-GPU validation, MPS/ROCm memory metrics | Any | Any | Excluded features | Not supported / out of scope under the [SPEC](spec-v0.1.md#non-goals). |

The packaging smoke checks `trainlens-0.1.0.dev0-py3-none-any.whl`, metadata,
license, console entry point, installed import location, two persisted successful
no-PyTorch runs, listing, Markdown comparison, and honest missing CUDA metrics.
It installs normal runtime dependencies; it is not an offline dependency-resolution
or PyTorch/CUDA test. See [execution instructions](../validation/packaging/README.md).

Before release, choose and publish the actual tested Python/PyTorch/OS matrix.
For each new verified row, retain the command, commit, environment, date, and
result or CI log link. Do not promote a pending row based on a configured workflow,
an import-only check, or mocked CUDA counters. See the [release checklist](release-checklist-v0.1.md).
