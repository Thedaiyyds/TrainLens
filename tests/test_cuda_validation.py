"""CPU/mock tests of validation infrastructure, NOT real NVIDIA validation."""

import importlib.util
import json
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trainlens.models import CUDADeviceMetrics, RunDiagnostics, RunMetrics, RunRecord
from trainlens.storage import RunStore

VALIDATION = Path(__file__).resolve().parents[1] / "validation" / "cuda"
spec = importlib.util.spec_from_file_location(
    "cuda_validation_runner", VALIDATION / "run_validation.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def mock_evidence(*, size=1024, pid=101, failed=False, abrupt=False):
    record = RunRecord(
        run_id=f"run-{pid}",
        name=f"run-{pid}",
        trainlens_version="test",
        command=["python", "fixture.py"],
        cwd="/isolated/validation",
        status="failed" if failed or abrupt else "succeeded",
        exit_code=7 if abrupt else (1 if failed else 0),
        metrics=RunMetrics([], unavailable_reason="finalization_missing")
        if abrupt
        else RunMetrics(
            [CUDADeviceMetrics("cuda:0", size, size + 4096)],
            runner.EXPECTED_SCOPE,
        ),
        diagnostics=RunDiagnostics(
            failure_message="RuntimeError: intentional validation failure"
            if failed
            else None
        ),
    )
    peaks = {"peak_allocated_bytes": size, "peak_reserved_bytes": size + 4096}
    direct = {
        "pid": pid,
        "device": "cuda:0",
        "requested_bytes": size,
        "allocator_backend": "native",
        "while_live": {**peaks, "current_allocated_bytes": size},
        "after_free": {**peaks, "current_allocated_bytes": 0},
    }
    return {
        "record": record,
        "direct": None if abrupt else direct,
        "cli_exit_code": 1 if failed or abrupt else 0,
        "stdout": "",
        "stderr": "Traceback (most recent call last)" if failed else "",
    }


def test_mock_exact_reference_and_retention_match():
    evidence = mock_evidence()
    runner.check_peaks(evidence)
    runner.check_retention(evidence)
    runner.check_isolation(evidence, mock_evidence(size=512, pid=102))
    runner.check_peaks(mock_evidence(failed=True), failed=True)
    runner.check_abrupt(mock_evidence(abrupt=True))


@pytest.mark.parametrize("field", ["peak_allocated_bytes", "peak_reserved_bytes"])
def test_mock_one_byte_difference_fails_without_tolerance(field):
    evidence = mock_evidence()
    evidence["direct"]["after_free"][field] += 1
    with pytest.raises(ValueError, match="!= direct"):
        runner.check_peaks(evidence)


def test_mock_zero_is_compared_as_zero_not_missing():
    evidence = mock_evidence(size=0)
    runner.check_peaks(evidence)
    evidence["record"].metrics.cuda_devices[0].peak_allocated_bytes = None
    with pytest.raises(ValueError, match="missing"):
        runner.check_peaks(evidence)


@pytest.mark.parametrize(
    "invalid", ["missing", "scope", "partial", "status", "exit", "cli", "reference"]
)
def test_mock_incomplete_or_inconsistent_measurements_fail(invalid):
    evidence = mock_evidence()
    record = evidence["record"]
    if invalid == "missing":
        record.metrics = RunMetrics([], unavailable_reason="not_collected")
    elif invalid == "scope":
        record.metrics.measurement_scope = "different"
    elif invalid == "partial":
        record.metrics.cuda_devices = [
            CUDADeviceMetrics("cuda:0", 1024, None, "query failed")
        ]
    elif invalid == "status":
        record.status = "failed"
    elif invalid == "exit":
        record.exit_code = 7
    elif invalid == "cli":
        evidence["cli_exit_code"] = 1
    else:
        evidence["direct"] = None
    with pytest.raises(ValueError):
        runner.check_peaks(evidence)


@pytest.mark.parametrize("invalid", ["peak", "not_freed", "pid", "run_id", "carryover"])
def test_mock_retention_and_fresh_process_regressions_fail(invalid):
    large, small = mock_evidence(), mock_evidence(size=512, pid=102)
    if invalid == "peak":
        large["direct"]["while_live"]["peak_reserved_bytes"] += 1
    elif invalid == "not_freed":
        large["direct"]["after_free"]["current_allocated_bytes"] = 1024
    elif invalid == "pid":
        small["direct"]["pid"] = large["direct"]["pid"]
    elif invalid == "run_id":
        small["record"].run_id = large["record"].run_id
    else:
        for key in ("while_live", "after_free"):
            small["direct"][key]["peak_allocated_bytes"] = 1024
        small["record"].metrics.cuda_devices[0].peak_allocated_bytes = 1024
    with pytest.raises(ValueError):
        runner.check_isolation(large, small)


@pytest.mark.parametrize("invalid", ["reason", "zero", "exit", "status"])
def test_mock_abrupt_exit_requires_honest_missing_data(invalid):
    evidence = mock_evidence(abrupt=True)
    record = evidence["record"]
    if invalid == "reason":
        record.metrics.unavailable_reason = "not_collected"
    elif invalid == "zero":
        record.metrics = RunMetrics([CUDADeviceMetrics("cuda:0", 0, 0)])
    elif invalid == "exit":
        record.exit_code = 1
    else:
        record.status = "succeeded"
    with pytest.raises(ValueError):
        runner.check_abrupt(evidence)


@pytest.mark.parametrize(
    "invalid", ["json", "fields", "null", "bool", "negative", "backend"]
)
def test_mock_direct_json_is_strictly_parsed(tmp_path, invalid):
    data = mock_evidence()["direct"]
    if invalid == "fields":
        data["extra"] = 1
    elif invalid == "backend":
        data["allocator_backend"] = "cudaMallocAsync"
    elif invalid in ("null", "bool", "negative"):
        data["after_free"]["peak_allocated_bytes"] = {
            "null": None,
            "bool": True,
            "negative": -1,
        }[invalid]
    path = tmp_path / "direct.json"
    path.write_text("{" if invalid == "json" else json.dumps(data))
    with pytest.raises(ValueError):
        runner.read_reference(path)


def test_mock_reference_round_trip_preserves_zero(tmp_path):
    data = mock_evidence()["direct"]
    data["after_free"]["peak_allocated_bytes"] = 0
    path = tmp_path / "direct.json"
    path.write_text(json.dumps(data))
    assert runner.read_reference(path) == data


@pytest.fixture
def mock_preflight(monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        runner, "_diagnostic_command", lambda command: "test-diagnostic"
    )
    for name in (
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        monkeypatch.delenv(name, raising=False)
    cuda = Mock(spec=["is_available", "device_count", "get_device_name", "memory"])
    cuda.is_available.return_value = True
    cuda.device_count.return_value = 1
    cuda.get_device_name.return_value = "MOCK GPU, not real NVIDIA evidence"
    cuda.memory = SimpleNamespace(get_allocator_backend=lambda: "native")
    torch = SimpleNamespace(
        __version__="mock-torch", version=SimpleNamespace(cuda="mock-cuda"), cuda=cuda
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: torch)
    return torch


def test_mock_supported_preflight_reports_environment(mock_preflight, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    result = runner.preflight()
    assert result["status"] == "PASS"
    environment = result["environment"]
    assert environment["pytorch"] == "mock-torch"
    assert environment["cuda_build"] == "mock-cuda"
    assert environment["gpu"].startswith("MOCK GPU")
    assert environment["allocator_backend"] == "native"
    assert environment["configuration"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["driver"] and environment["trainlens_commit"]


def test_mock_preflight_rejects_installed_checkout_mismatch(
    mock_preflight, monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "REPOSITORY", tmp_path)
    result = runner.preflight()
    assert result["status"] == "FAIL"
    assert "does not match this checkout" in result["reason"]
    mock_preflight.cuda.is_available.assert_not_called()


@pytest.mark.parametrize(
    "state",
    [
        "absent",
        "cpu",
        "unavailable",
        "backend",
        "unknown",
        "disabled",
        "platform",
        "probe_error",
        "driver_missing",
        "commit_missing",
    ],
)
def test_mock_preflight_skips_unvalidated_environments(
    mock_preflight, monkeypatch, state
):
    torch = mock_preflight
    if state == "absent":
        monkeypatch.setattr(
            runner.importlib,
            "import_module",
            Mock(side_effect=ModuleNotFoundError("torch")),
        )
    elif state == "cpu":
        torch.version.cuda = None
        torch.cuda.is_available.return_value = False
    elif state == "unavailable":
        torch.cuda.is_available.return_value = False
    elif state == "backend":
        torch.cuda.memory.get_allocator_backend = lambda: "cudaMallocAsync"
    elif state == "unknown":
        torch.cuda.memory = SimpleNamespace()
    elif state == "disabled":
        monkeypatch.setenv("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
    elif state == "platform":
        monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    elif state in ("driver_missing", "commit_missing"):
        monkeypatch.setattr(
            runner,
            "_diagnostic_command",
            lambda command: (
                None
                if (command[0] == "nvidia-smi") == (state == "driver_missing")
                else "test-diagnostic"
            ),
        )
    else:
        torch.cuda.device_count.side_effect = RuntimeError("probe failed")
    result = runner.preflight()
    assert result["status"] == ("FAIL" if state == "probe_error" else "SKIP")
    assert result["reason"]
    if state in ("cpu", "unavailable"):
        torch.cuda.device_count.assert_not_called()


@pytest.mark.parametrize("status,exit_code", [("PASS", 0), ("FAIL", 1), ("SKIP", 2)])
def test_mock_runner_aggregation_and_evidence_output(
    tmp_path, monkeypatch, capsys, status, exit_code
):
    monkeypatch.setattr(
        runner,
        "preflight",
        lambda: {"status": "PASS", "reason": "MOCK", "environment": {"gpu": "MOCK"}},
    )
    monkeypatch.setattr(
        runner,
        "run_suite",
        lambda *args: {
            "scenarios": {
                name: {"status": status if index == 0 else "PASS", "reason": "MOCK"}
                for index, name in enumerate(runner.SCENARIOS)
            },
            "runs": {},
        },
    )
    output = tmp_path / "mock-only.json"
    assert runner.main(["--output", str(output)]) == exit_code
    assert json.loads(output.read_text())["status"] == status
    assert f"Normal allocation: {status}" in capsys.readouterr().out


def test_mock_no_gpu_skips_suite_without_claiming_pass(monkeypatch, capsys):
    monkeypatch.setattr(
        runner,
        "preflight",
        lambda: {"status": "SKIP", "reason": "CUDA unavailable", "environment": {}},
    )
    monkeypatch.setattr(
        runner, "run_suite", lambda *args: pytest.fail("suite executed")
    )
    assert runner.main([]) == 2
    output = capsys.readouterr().out
    assert "REAL NVIDIA VALIDATION NOT RUN / SKIPPED" in output
    assert "PASS" not in output


def test_mock_preflight_failure_does_not_execute_suite(monkeypatch, capsys):
    monkeypatch.setattr(
        runner,
        "preflight",
        lambda: {
            "status": "FAIL",
            "reason": "probe failed",
            "environment": {},
        },
    )
    monkeypatch.setattr(
        runner, "run_suite", lambda *args: pytest.fail("suite executed")
    )
    assert runner.main([]) == 1
    assert "REAL NVIDIA VALIDATION NOT RUN" in capsys.readouterr().out


def test_runner_refuses_to_overwrite_evidence(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("user evidence")
    monkeypatch.setattr(runner, "preflight", lambda: pytest.fail("GPU probed"))
    with pytest.raises(SystemExit) as caught:
        runner.main(["--output", str(output)])
    assert caught.value.code == 2
    assert output.read_text() == "user evidence"


def test_runner_rejects_evidence_inside_user_store(tmp_path, monkeypatch):
    output = tmp_path / ".trainlens" / "runs" / "evidence.json"
    monkeypatch.setattr(runner, "preflight", lambda: pytest.fail("GPU probed"))
    with pytest.raises(SystemExit):
        runner.main(["--output", str(output)])
    assert not output.parent.exists()


def test_mock_timeout_terminates_owned_cli_and_bootstrap_group(tmp_path, monkeypatch):
    class TimedOutProcess:
        pid = 123456
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("validation-cli", 1)
            return "", ""

    process = TimedOutProcess()
    kill = Mock()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner.os, "killpg", kill)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_fixture(tmp_path, "timeout", "allocate_peak.py", 1024, 1)
    kill.assert_called_once_with(process.pid, runner.signal.SIGKILL)
    assert process.calls == 2


def test_mock_suite_reports_launch_failure_and_continues(monkeypatch):
    seen = []

    def fixtures(root, name, script, size, timeout):
        seen.append(name)
        if name == "large":
            raise FileNotFoundError("reference missing")
        return mock_evidence(
            size=size,
            pid=102 + len(seen),
            failed=name == "exception",
            abrupt=name == "abrupt",
        )

    monkeypatch.setattr(runner, "run_fixture", fixtures)
    report = runner.run_suite(1024, 512, 10)
    assert seen == ["large", "small", "exception", "abrupt"]
    assert [report["scenarios"][name]["status"] for name in runner.SCENARIOS] == [
        "FAIL",
        "FAIL",
        "FAIL",
        "PASS",
        "PASS",
    ]
    assert "reference missing" in report["scenarios"]["Normal allocation"]["reason"]
    json.dumps(report)


def test_validation_modules_import_without_torch_or_cuda():
    source = """
import runpy, sys
sys.path.insert(0, sys.argv[1])
class RejectTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Import-time torch probe")
sys.meta_path.insert(0, RejectTorch())
from pathlib import Path
for path in Path(sys.argv[1]).glob("*.py"):
    runpy.run_path(str(path), run_name="validation_import_test")
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", source, str(VALIDATION)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_mock_full_suite_uses_persisted_cli_records_and_isolates_storage(
    tmp_path, monkeypatch
):
    root = tmp_path / "selected-venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    python = root / "bin" / "python"
    site = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    source = VALIDATION.parents[1] / "src"
    (Path(site) / "trainlens-test.pth").write_text(
        f"{source}\n{sysconfig.get_path('purelib')}\n"
    )
    (Path(site) / "torch.py").write_text("""
from types import SimpleNamespace
__version__ = "MOCK, NOT REAL GPU"
version = SimpleNamespace(cuda="mock-cuda")
uint8 = object()
class CUDA:
    initialized = False
    current = 0
    peak = 0
    reserved = 0
    memory = SimpleNamespace(get_allocator_backend=lambda: "native")
    def is_available(self): return True
    def is_initialized(self): return self.initialized
    def device_count(self):
        assert self.initialized
        return 1
    def get_device_name(self, index): return "MOCK GPU"
    def max_memory_allocated(self, index): return self.peak
    def max_memory_reserved(self, index): return self.reserved
    def memory_allocated(self, index): return self.current
cuda = CUDA()
class Tensor:
    def __init__(self, size): self.size = size
    def numel(self): return self.size
    def __del__(self): cuda.current = 0
def empty(size, *, dtype, device):
    assert dtype is uint8 and device == "cuda:0"
    cuda.initialized = True
    cuda.current = size
    cuda.peak = max(cuda.peak, size)
    cuda.reserved = size + 4096
    return Tensor(size)
""")
    store = RunStore(tmp_path)
    path = store.save(mock_evidence()["record"])
    before = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner.sys, "executable", str(python))
    report = runner.run_suite(4 * 1024**2, 1024**2, 30)
    assert all(item["status"] == "PASS" for item in report["scenarios"].values()), (
        report
    )
    assert path.read_bytes() == before
    assert len(store.list_records()) == 1
    assert (
        len({evidence["record"]["run_id"] for evidence in report["runs"].values()}) == 4
    )
    assert report["runs"]["exception"]["record"]["status"] == "failed"
    assert (
        report["runs"]["abrupt"]["record"]["metrics"]["unavailable_reason"]
        == "finalization_missing"
    )
    assert (
        report["runs"]["large"]["record"]["metrics"]["cuda_devices"][0][
            "peak_allocated_bytes"
        ]
        == report["runs"]["large"]["direct"]["after_free"]["peak_allocated_bytes"]
    )
    json.dumps(report)
