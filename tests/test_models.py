import json
from dataclasses import asdict

import pytest

from trainlens.models import (
    CUDADeviceMetrics,
    EnvironmentInfo,
    GitInfo,
    GPUInfo,
    RunDiagnostics,
    RunMetrics,
    RunRecord,
)


@pytest.fixture
def completed_run() -> RunRecord:
    return RunRecord(
        run_id="run-001",
        name="baseline",
        trainlens_version="0.1.0.dev0",
        command=["python", "train.py", "--label", "baseline run"],
        cwd="/work/project",
        python_executable="/work/project/.venv/bin/python",
        started_at="2026-09-21T00:00:00Z",
        ended_at="2026-09-21T00:00:10Z",
        runtime_seconds=10.0,
        status="succeeded",
        exit_code=0,
        environment=EnvironmentInfo(
            python_version="3.10.0",
            os="Linux",
            architecture="x86_64",
            pytorch_version="2.5.0",
            cuda_build_version="12.4",
            cuda_available=True,
            gpus=[GPUInfo(device="cuda:0", name="Test GPU")],
        ),
        git=GitInfo(commit="a" * 40, dirty=False),
        metrics=RunMetrics(
            cuda_devices=[CUDADeviceMetrics("cuda:0", 1024, 2048)],
            measurement_scope="test_allocation_window",
        ),
    )


def test_successful_run_serializes_to_json_compatible_data(
    completed_run: RunRecord,
) -> None:
    data = asdict(completed_run)

    assert data["schema_version"] == 1
    assert data["trainlens_version"] == "0.1.0.dev0"
    assert data["status"] == "succeeded"
    assert data["exit_code"] == 0
    assert data["command"] == ["python", "train.py", "--label", "baseline run"]
    assert data["runtime_seconds"] == 10.0
    assert data["started_at"] == "2026-09-21T00:00:00Z"
    assert data["git"] == {
        "commit": "a" * 40,
        "dirty": False,
        "unavailable_reason": None,
    }
    assert data["environment"]["gpus"] == [{"device": "cuda:0", "name": "Test GPU"}]
    assert data["metrics"] == {
        "cuda_devices": [
            {
                "device": "cuda:0",
                "peak_allocated_bytes": 1024,
                "peak_reserved_bytes": 2048,
                "unavailable_reason": None,
            }
        ],
        "measurement_scope": "test_allocation_window",
        "unavailable_reason": None,
    }
    assert json.loads(json.dumps(data)) == data


def test_failed_run_keeps_failure_and_collection_diagnostics(
    completed_run: RunRecord,
) -> None:
    completed_run.status = "failed"
    completed_run.exit_code = 1
    completed_run.metrics = RunMetrics(
        cuda_devices=[], unavailable_reason="finalization_missing"
    )
    completed_run.diagnostics = RunDiagnostics(
        failure_message="ValueError: invalid training configuration",
        collection_warnings=["CUDA peaks could not be collected before exit."],
    )

    data = asdict(completed_run)
    assert data["status"] == "failed"
    assert data["exit_code"] == 1
    assert data["diagnostics"] == {
        "failure_message": "ValueError: invalid training configuration",
        "collection_warnings": ["CUDA peaks could not be collected before exit."],
    }
    assert data["metrics"]["unavailable_reason"] == "finalization_missing"


@pytest.mark.parametrize(
    "pytorch_version,cuda_available,reason",
    [(None, None, "torch_unavailable"), ("2.5.0", False, "cuda_unavailable")],
)
def test_missing_environment_git_and_cuda_do_not_change_success(
    completed_run: RunRecord,
    pytorch_version: str | None,
    cuda_available: bool | None,
    reason: str,
) -> None:
    completed_run.environment = EnvironmentInfo(
        python_version="3.13.15",
        os="Darwin",
        architecture="arm64",
        pytorch_version=pytorch_version,
        cuda_available=cuda_available,
        gpus=[],
    )
    completed_run.git = GitInfo(unavailable_reason="not_a_git_repository")
    completed_run.metrics = RunMetrics(cuda_devices=[], unavailable_reason=reason)
    completed_run.diagnostics.collection_warnings.append(reason)

    data = json.loads(json.dumps(asdict(completed_run)))
    assert data["status"] == "succeeded"
    assert data["exit_code"] == 0
    assert data["environment"]["pytorch_version"] == pytorch_version
    assert data["environment"]["cuda_available"] is cuda_available
    assert data["environment"]["gpus"] == []
    assert data["git"]["commit"] is None
    assert data["git"]["dirty"] is None
    assert data["git"]["unavailable_reason"] == "not_a_git_repository"
    assert data["metrics"]["cuda_devices"] == []
    assert data["metrics"]["unavailable_reason"] == reason
    assert data["diagnostics"]["failure_message"] is None
    assert data["diagnostics"]["collection_warnings"]


def test_missing_device_peak_is_distinct_from_measured_zero() -> None:
    metrics = CUDADeviceMetrics(
        device="cuda:0",
        peak_allocated_bytes=0,
        peak_reserved_bytes=None,
        unavailable_reason="Reserved peak query failed.",
    )

    data = json.loads(json.dumps(asdict(metrics)))
    assert data["peak_allocated_bytes"] == 0
    assert data["peak_reserved_bytes"] is None
    assert data["unavailable_reason"] == "Reserved peak query failed."


def test_missing_peaks_require_a_reason() -> None:
    with pytest.raises(ValueError, match="unavailable_reason"):
        CUDADeviceMetrics("cuda:0", None, None)
    with pytest.raises(ValueError, match="unavailable_reason"):
        RunMetrics(cuda_devices=[])


@pytest.mark.parametrize("allocated,reserved", [(-1, 0), (0, -1)])
def test_negative_peaks_are_rejected(allocated: int, reserved: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CUDADeviceMetrics("cuda:0", allocated, reserved)


def test_available_peaks_cannot_be_marked_unavailable() -> None:
    with pytest.raises(ValueError, match="cannot have an unavailable_reason"):
        CUDADeviceMetrics("cuda:0", 0, 0, unavailable_reason="cuda_unavailable")


@pytest.mark.parametrize("reason", ["cuda_unavailable", ""])
def test_device_metrics_reject_an_overall_unavailable_reason(reason: str) -> None:
    with pytest.raises(ValueError, match="overall unavailable_reason"):
        RunMetrics(
            cuda_devices=[CUDADeviceMetrics("cuda:0", 0, 0)],
            unavailable_reason=reason,
        )


def test_invalid_run_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="Run status"):
        RunRecord(
            run_id="run-001",
            name="invalid-status",
            trainlens_version="0.1.0.dev0",
            command=["python", "train.py"],
            cwd="/work",
            status="unknown",
        )


def test_new_records_are_incomplete_and_do_not_share_mutable_defaults() -> None:
    first = RunRecord("run-001", "first", "0.1.0.dev0", ["python", "train.py"], "/work")
    second = RunRecord(
        "run-002", "second", "0.1.0.dev0", ["python", "train.py"], "/work"
    )
    first.diagnostics.collection_warnings.append("First run warning")
    first.metrics.cuda_devices.append(CUDADeviceMetrics("cuda:0", 0, 0))
    first.metrics.unavailable_reason = None
    first.git.commit = "b" * 40
    first.git.dirty = True
    first.git.unavailable_reason = None
    first.environment.cuda_available = False

    assert second.schema_version == 1
    assert second.status == "running"
    assert second.started_at is None
    assert second.ended_at is None
    assert second.runtime_seconds is None
    assert second.exit_code is None
    assert second.environment.cuda_available is None
    assert second.environment.gpus is None
    assert second.git.dirty is None
    assert second.git.unavailable_reason == "not_collected"
    assert second.metrics.cuda_devices == []
    assert second.metrics.unavailable_reason == "not_collected"
    assert second.diagnostics.collection_warnings == []
