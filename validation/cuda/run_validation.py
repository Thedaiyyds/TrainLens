"""Opt-in real NVIDIA validation. Importing this module does not import PyTorch."""

import argparse
import importlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import trainlens
from trainlens.models import RunRecord
from trainlens.storage import RunStore

FIXTURES = Path(__file__).resolve().parent
REPOSITORY = FIXTURES.parent.parent
EXPECTED_SCOPE = "bootstrap_to_script_exit_v1"
SCENARIOS = (
    "Normal allocation",
    "Peak after free",
    "Fresh process isolation",
    "Python exception",
    "Abrupt exit",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _diagnostic_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return (result.stdout.strip() or None) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def preflight() -> dict:
    """Observe the runner's environment, only when explicitly invoked.

    Native allocator is this validation fixture's evidence scope, not an added
    product compatibility contract. No configuration is changed here.
    """
    environment = {
        "os": platform.platform(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "trainlens_version": version("trainlens"),
        "trainlens_source": str(Path(trainlens.__file__).resolve()),
        "trainlens_commit": _diagnostic_command(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"]
        ),
        "worktree_status": _diagnostic_command(
            ["git", "-C", str(REPOSITORY), "status", "--porcelain"]
        ),
        "pytorch": None,
        "cuda_build": None,
        "cuda_available": None,
        "device_count": None,
        "selected_device": "cuda:0",
        "gpu": None,
        "driver": None,
        "allocator_backend": None,
        "configuration": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_NO_CUDA_MEMORY_CACHING",
            )
        },
    }

    def result(status: str, reason: str) -> dict:
        return {"status": status, "reason": reason, "environment": environment}

    if (
        Path(trainlens.__file__).resolve()
        != (REPOSITORY / "src/trainlens/__init__.py").resolve()
    ):
        return result(
            "FAIL",
            "Installed TrainLens does not match this checkout; install it with pip -e",
        )
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        return result("SKIP", f"PyTorch unavailable: {error}")
    except Exception as error:
        return result("FAIL", f"PyTorch import failed: {error}")
    try:
        environment["pytorch"] = str(torch.__version__)
        environment["cuda_build"] = torch.version.cuda
        environment["cuda_available"] = bool(torch.cuda.is_available())
        if not environment["cuda_build"] or not environment["cuda_available"]:
            return result("SKIP", "NVIDIA CUDA unavailable")
        environment["device_count"] = torch.cuda.device_count()
        if environment["device_count"] < 1:
            return result("SKIP", "No visible CUDA devices")
        environment["gpu"] = torch.cuda.get_device_name(0)
        environment["driver"] = _diagnostic_command(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        )
        getter = getattr(torch.cuda.memory, "get_allocator_backend", None)
        if getter is not None:
            environment["allocator_backend"] = getter()
    except Exception as error:
        return result("FAIL", f"CUDA environment probe failed: {error}")
    if environment["platform"] != "Linux":
        return result("SKIP", "Real NVIDIA evidence requires native Linux")
    if environment["allocator_backend"] != "native":
        return result(
            "SKIP",
            "Allocator backend unknown or not validated by this suite (requires native)",
        )
    if not environment["driver"] or not environment["trainlens_commit"]:
        return result(
            "SKIP",
            "Driver version or TrainLens commit unavailable; environment evidence incomplete",
        )
    if environment["configuration"]["PYTORCH_NO_CUDA_MEMORY_CACHING"] not in (
        None,
        "0",
    ):
        return result("SKIP", "Disabled CUDA caching is outside this validation scope")
    return result("PASS", "Linux + CUDA + native allocator preflight completed")


def read_reference(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(data, dict)
        and data.keys()
        == {
            "pid",
            "device",
            "requested_bytes",
            "allocator_backend",
            "while_live",
            "after_free",
        },
        "Invalid direct reference fields",
    )
    _require(data["device"] == "cuda:0", "Unexpected reference device")
    _require(
        data["allocator_backend"] == "native", "Child allocator backend is not native"
    )
    for field in ("pid", "requested_bytes"):
        _require(
            type(data[field]) is int and data[field] > 0, f"Invalid reference {field}"
        )
    for boundary in ("while_live", "after_free"):
        values = data[boundary]
        _require(
            isinstance(values, dict)
            and values.keys()
            == {
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "current_allocated_bytes",
            },
            f"Invalid {boundary} reference fields",
        )
        for name, value in values.items():
            _require(type(value) is int and value >= 0, f"Invalid reference {name}")
    return data


def run_fixture(root: Path, name: str, script: str, size: int, timeout: int) -> dict:
    cwd = root / name
    cwd.mkdir()
    reference = cwd / "direct.json"
    command = [
        sys.executable,
        "-c",
        "from trainlens.cli import app; app()",
        "run",
        "--name",
        name,
        "--",
        sys.executable,
        str(FIXTURES / script),
        "--bytes",
        str(size),
    ]
    if script != "abrupt_exit.py":
        command.extend(["--output", str(reference)])
    # The CLI launches the bootstrap child. A timeout must terminate both, rather
    # than leave a GPU workload running after its temporary store is removed.
    with subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        completed = subprocess.CompletedProcess(
            command, process.returncode, stdout, stderr
        )
    try:
        record = RunStore(cwd).resolve(name)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Saved record unavailable: {error}; CLI exit={completed.returncode}; "
            f"stderr={completed.stderr}"
        ) from error
    evidence = {
        "record": record,
        "direct": None,
        "cli_exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    try:
        if reference.exists():
            evidence["direct"] = read_reference(reference)
    except (OSError, ValueError) as error:
        evidence["error"] = f"Direct reference invalid: {error}"
    return evidence


def check_peaks(evidence: dict, *, failed: bool = False) -> None:
    record: RunRecord = evidence["record"]
    _require(
        record.status == ("failed" if failed else "succeeded"),
        "Unexpected training status",
    )
    _require(record.exit_code == (1 if failed else 0), "Unexpected training exit code")
    _require(
        evidence["cli_exit_code"] == (1 if failed else 0), "Unexpected CLI exit code"
    )
    _require(
        record.metrics.measurement_scope == EXPECTED_SCOPE, "Wrong measurement scope"
    )
    _require(record.metrics.unavailable_reason is None, "CUDA metrics unavailable")
    devices = [item for item in record.metrics.cuda_devices if item.device == "cuda:0"]
    _require(len(devices) == 1, "Missing or duplicate cuda:0 metrics")
    device = devices[0]
    _require(device.unavailable_reason is None, "Partial CUDA metrics unavailable")
    _require(evidence["direct"] is not None, "Direct reference missing")
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        value = getattr(device, field)
        _require(type(value) is int, f"Saved {field} missing")
        reference = evidence["direct"]["after_free"][field]
        _require(value == reference, f"{field}: saved {value} != direct {reference}")
    if failed:
        _require(
            "intentional validation failure"
            in (record.diagnostics.failure_message or ""),
            "Training failure diagnostic missing",
        )
        _require(
            "Traceback (most recent call last)" in evidence["stderr"],
            "Training traceback missing",
        )


def check_retention(evidence: dict) -> None:
    check_peaks(evidence)
    direct = evidence["direct"]
    live, freed = direct["while_live"], direct["after_free"]
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        _require(freed[field] == live[field], f"{field} changed after free")
    _require(
        freed["peak_allocated_bytes"] >= direct["requested_bytes"],
        "Allocation peak below requested size",
    )
    _require(
        live["current_allocated_bytes"] - freed["current_allocated_bytes"]
        >= direct["requested_bytes"],
        "Tensor release was not observed",
    )


def check_isolation(large: dict, small: dict) -> None:
    check_retention(large)
    check_retention(small)
    _require(large["record"].run_id != small["record"].run_id, "Run identities reused")
    _require(
        large["direct"]["pid"] != small["direct"]["pid"], "Training process reused"
    )
    _require(
        small["direct"]["requested_bytes"] < large["direct"]["requested_bytes"],
        "Sizes must differ",
    )
    _require(
        small["direct"]["after_free"]["peak_allocated_bytes"]
        < large["direct"]["after_free"]["peak_allocated_bytes"],
        "Small fresh run retained a large allocation peak",
    )


def check_abrupt(evidence: dict) -> None:
    record: RunRecord = evidence["record"]
    _require(
        record.status == "failed" and record.exit_code == 7,
        "Abrupt exit outcome differs",
    )
    _require(evidence["cli_exit_code"] == 1, "Abrupt run CLI should fail")
    _require(not record.metrics.cuda_devices, "Abrupt run contains fabricated peaks")
    _require(
        record.metrics.unavailable_reason == "finalization_missing",
        "Finalization missing reason absent",
    )
    _require(
        record.metrics.measurement_scope is None, "Abrupt run claims a measured scope"
    )
    _require(
        evidence["direct"] is None, "Abrupt fixture should not report reference peaks"
    )


def run_suite(large_bytes: int, small_bytes: int, timeout: int) -> dict:
    runs = {}
    results = {}
    with tempfile.TemporaryDirectory(prefix="trainlens-cuda-validation-") as temporary:
        root = Path(temporary)
        for name, script, size in (
            ("large", "allocate_peak.py", large_bytes),
            ("small", "allocate_peak.py", small_bytes),
            ("exception", "failure_after_allocate.py", small_bytes),
            ("abrupt", "abrupt_exit.py", small_bytes),
        ):
            try:
                runs[name] = run_fixture(root, name, script, size, timeout)
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                runs[name] = {"error": str(error)}
        checks = (
            (SCENARIOS[0], ("large",), lambda: check_peaks(runs["large"])),
            (SCENARIOS[1], ("large",), lambda: check_retention(runs["large"])),
            (
                SCENARIOS[2],
                ("large", "small"),
                lambda: check_isolation(runs["large"], runs["small"]),
            ),
            (
                SCENARIOS[3],
                ("exception",),
                lambda: check_peaks(runs["exception"], failed=True),
            ),
            (SCENARIOS[4], ("abrupt",), lambda: check_abrupt(runs["abrupt"])),
        )
        for name, required, check in checks:
            try:
                for key in required:
                    _require(
                        "error" not in runs[key], f"{key}: {runs[key].get('error')}"
                    )
                check()
            except ValueError as error:
                results[name] = {"status": "FAIL", "reason": str(error)}
            else:
                results[name] = {
                    "status": "PASS",
                    "reason": "Persisted record matches scenario requirements",
                }
    snapshots = {
        name: {**evidence, "record": asdict(evidence["record"])}
        if "record" in evidence
        else evidence
        for name, evidence in runs.items()
    }
    return {"scenarios": results, "runs": snapshots}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-mib", type=int, default=64)
    parser.add_argument("--small-mib", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per CLI run")
    parser.add_argument("--output", type=Path, help="Save evidence JSON to a new file")
    args = parser.parse_args(argv)
    if not (args.large_mib > args.small_mib > 0 and args.timeout > 0):
        parser.error("Require large-mib > small-mib > 0 and timeout > 0")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new evidence file")
    if args.output is not None and ".trainlens" in args.output.resolve().parts:
        parser.error("Evidence output must be outside any .trainlens store")
    probe = preflight()
    print("Environment")
    print(json.dumps(probe["environment"], indent=2, ensure_ascii=False))
    report = {"observed_at": datetime.now(timezone.utc).isoformat(), "preflight": probe}
    if probe["status"] == "PASS":
        report.update(
            run_suite(args.large_mib * 1024**2, args.small_mib * 1024**2, args.timeout)
        )
    else:
        report.update(
            scenarios={
                name: {"status": "SKIP", "reason": probe["reason"]}
                for name in SCENARIOS
            },
            runs={},
        )
    statuses = [value["status"] for value in report["scenarios"].values()]
    report["status"] = (
        "FAIL"
        if probe["status"] == "FAIL" or "FAIL" in statuses
        else ("SKIP" if "SKIP" in statuses else "PASS")
    )
    for name, value in report["scenarios"].items():
        print(f"{name}: {value['status']} — {value['reason']}")
    if report["status"] == "SKIP":
        print("REAL NVIDIA VALIDATION NOT RUN / SKIPPED")
    elif probe["status"] != "PASS":
        print(f"REAL NVIDIA VALIDATION NOT RUN — preflight FAIL: {probe['reason']}")
    else:
        print(
            f"REAL NVIDIA VALIDATION: {report['status']} (controlled suite only; not v0.1 acceptance)"
        )
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        print(f"Evidence: {args.output}")
    return {"PASS": 0, "FAIL": 1, "SKIP": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
