import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trainlens import supervisor
from trainlens.cli import app
from trainlens.models import EnvironmentInfo, GitInfo
from trainlens.storage import RunStore


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "train.py").write_text("pass\n")
    monkeypatch.setattr(supervisor, "collect_git", lambda cwd: GitInfo())
    return tmp_path


def payload(request):
    return {
        "payload_version": 1,
        "run_id": request["run_id"],
        "python_executable": "/selected/child/python",
        "environment": asdict(EnvironmentInfo(python_version="child-only-version")),
        "warnings": ["child warning"],
        "failure_message": None,
    }


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize(
    "result", ["missing", "bad-json", "bad-environment", "wrong-id"]
)
def test_payload_failure_preserves_observed_outcome(
    project, monkeypatch, result, exit_code
):
    requests = []

    def child(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", "trainlens._bootstrap_child"]
        assert kwargs == {"cwd": project.resolve(), "check": False}
        request_path = Path(command[3])
        requests.append(request_path)
        request = json.loads(request_path.read_text())
        data = payload(request)
        if result == "bad-environment":
            data["environment"]["cuda_available"] = "not a boolean"
        if result == "wrong-id":
            data["run_id"] = "different-run"
        if result != "missing":
            Path(request["result_path"]).write_text(
                "{" if result == "bad-json" else json.dumps(data)
            )
        return subprocess.CompletedProcess(command, exit_code)

    monkeypatch.setattr(supervisor.subprocess, "run", child)
    record = supervisor.supervise_run(
        "baseline", [sys.executable, "train.py"], invocation_cwd=project
    )

    assert record.exit_code == exit_code
    assert record.status == ("succeeded" if exit_code == 0 else "failed")
    assert record.environment == EnvironmentInfo()
    assert record.python_executable is None
    assert record.diagnostics.collection_warnings
    assert "Child metadata unavailable" in record.diagnostics.collection_warnings[0]
    assert RunStore(project).load(record.run_id) == record
    assert not requests[0].parent.exists()


def test_valid_child_payload_and_monotonic_timing(project, monkeypatch):
    def child(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        Path(request["result_path"]).write_text(json.dumps(payload(request)))
        return subprocess.CompletedProcess(command, 0)

    ticks = iter([100.0, 102.5])
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(supervisor.subprocess, "run", child)
    record = supervisor.supervise_run(
        "baseline", [sys.executable, "train.py"], invocation_cwd=project
    )

    assert record.environment.python_version == "child-only-version"
    assert record.python_executable == "/selected/child/python"
    assert record.runtime_seconds == 2.5
    assert record.diagnostics.collection_warnings == ["child warning"]
    assert record.metrics.cuda_devices == []
    assert record.metrics.unavailable_reason == "not_collected"


def test_launch_failure_is_recorded_without_fabricated_exit_or_environment(
    project, monkeypatch
):
    def cannot_launch(*args, **kwargs):
        raise PermissionError("cannot execute selected Python")

    monkeypatch.setattr(supervisor.subprocess, "run", cannot_launch)
    record = supervisor.supervise_run(
        "baseline", [sys.executable, "train.py"], invocation_cwd=project
    )

    assert record.status == "failed"
    assert record.exit_code is None
    assert record.runtime_seconds is None
    assert record.python_executable is None
    assert "Launch/bootstrap failure" in record.diagnostics.failure_message
    assert RunStore(project).load(record.run_id) == record


def test_storage_failure_after_success_is_cli_failure(project, monkeypatch):
    launched = []

    def child(command, **kwargs):
        launched.append(True)
        request = json.loads(Path(command[-1]).read_text())
        Path(request["result_path"]).write_text(json.dumps(payload(request)))
        return subprocess.CompletedProcess(command, 0)

    def failed_save(self, record):
        assert record.status == "succeeded"
        raise OSError("disk full after training")

    monkeypatch.setattr(supervisor.subprocess, "run", child)
    monkeypatch.setattr(RunStore, "save", failed_save)
    monkeypatch.chdir(project)
    result = CliRunner().invoke(
        app, ["run", "--name", "baseline", "--", sys.executable, "train.py"]
    )

    assert launched == [True]
    assert result.exit_code != 0
    assert "Could not save run" in result.output
    assert "disk full after training" in result.output
    assert "TrainLens: saved" not in result.output


def test_storage_preflight_prevents_launch(project, monkeypatch):
    (project / ".trainlens").write_text("existing user content")
    monkeypatch.setattr(
        supervisor.subprocess, "run", lambda *a, **k: pytest.fail("launched")
    )
    with pytest.raises(supervisor.RunStorageError, match="before launch"):
        supervisor.supervise_run(
            "baseline", [sys.executable, "train.py"], invocation_cwd=project
        )
    assert (project / ".trainlens").read_text() == "existing user content"


def test_store_preflight_is_read_only_and_reuses_duplicate_checks(project):
    store = RunStore(project)
    store.check_available("new-run", "baseline")
    assert not store.runs_dir.exists()


def test_transport_cleanup_error_preserves_successful_child_outcome(
    project, monkeypatch
):
    original_temporary_directory = supervisor.tempfile.TemporaryDirectory

    class CleanupFailure:
        def __init__(self, **kwargs):
            self.directory = original_temporary_directory(**kwargs)

        def __enter__(self):
            return self.directory.__enter__()

        def __exit__(self, *args):
            self.directory.__exit__(*args)
            raise OSError("cleanup failed")

    def child(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        Path(request["result_path"]).write_text(json.dumps(payload(request)))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(supervisor.tempfile, "TemporaryDirectory", CleanupFailure)
    monkeypatch.setattr(supervisor.subprocess, "run", child)
    record = supervisor.supervise_run(
        "baseline", [sys.executable, "train.py"], invocation_cwd=project
    )
    assert record.status == "succeeded"
    assert record.exit_code == 0
    assert any(
        "cleanup failed" in warning
        for warning in record.diagnostics.collection_warnings
    )
    assert RunStore(project).load(record.run_id) == record
