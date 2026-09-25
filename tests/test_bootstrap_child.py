import json
from dataclasses import asdict
from pathlib import Path

import pytest

from trainlens import _bootstrap_child as child
from trainlens._run_payload import decode_result, validate_request
from trainlens.models import (
    CUDADeviceMetrics,
    EnvironmentInfo,
    GPUInfo,
    RunDiagnostics,
    RunMetrics,
)


@pytest.fixture(autouse=True)
def no_real_torch(monkeypatch):
    monkeypatch.setattr(
        child,
        "collect_cuda_metrics",
        lambda **kwargs: RunMetrics([], unavailable_reason="pytorch_unavailable"),
    )


def request_file(tmp_path, source):
    script = tmp_path / "train.py"
    script.write_text(source)
    request = {
        "run_id": "test-run",
        "script": str(script),
        "args": [],
        "cwd": str(tmp_path),
        "result_path": str(tmp_path / "result.json"),
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    return path, Path(request["result_path"])


def test_collection_failure_still_executes_user_script(tmp_path, monkeypatch):
    request, result = request_file(
        tmp_path, "from pathlib import Path\nPath('executed').touch()"
    )

    def fail_collection(**kwargs):
        raise RuntimeError("broken optional collector")

    monkeypatch.setattr(child, "collect_environment", fail_collection)
    child.run_request(request)

    decoded = decode_result(json.loads(result.read_text()), "test-run")
    assert (tmp_path / "executed").exists()
    assert decoded.environment == EnvironmentInfo()
    assert decoded.python_executable
    assert any("broken optional collector" in warning for warning in decoded.warnings)
    assert decoded.failure_message is None
    assert decoded.phase == "final"


@pytest.mark.parametrize(
    "ending,error",
    [("raise SystemExit(7)", SystemExit), ("raise RuntimeError('boom')", RuntimeError)],
)
def test_child_writes_diagnostics_and_reraises(tmp_path, monkeypatch, ending, error):
    request, result = request_file(tmp_path, ending)
    monkeypatch.setattr(
        child, "collect_environment", lambda **kwargs: EnvironmentInfo()
    )
    with pytest.raises(error):
        child.run_request(request)
    data = json.loads(result.read_text())
    assert error.__name__ in data["failure_message"]


@pytest.mark.parametrize(
    "ending,expected", [("pass", None), ("raise SystemExit(7)", SystemExit)]
)
def test_payload_write_error_does_not_change_training_outcome(
    tmp_path, monkeypatch, ending, expected
):
    request, result = request_file(
        tmp_path, "from pathlib import Path\nPath('executed').touch()\n" + ending
    )
    monkeypatch.setattr(
        child, "collect_environment", lambda **kwargs: EnvironmentInfo()
    )

    def fail_write(*args):
        raise OSError("no room for metadata")

    monkeypatch.setattr(child, "_write_result", fail_write)
    if expected:
        with pytest.raises(expected) as caught:
            child.run_request(request)
        assert caught.value.code == 7
    else:
        child.run_request(request)
    assert (tmp_path / "executed").exists()
    assert not result.exists()


def test_atomic_payload_failure_preserves_previous_json(tmp_path, monkeypatch):
    result = tmp_path / "result.json"
    result.write_text('{"original": true}')

    def fail_replace(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr(child.os, "replace", fail_replace)
    with pytest.raises(OSError):
        child._write_result(result, {"new": True})
    assert json.loads(result.read_text()) == {"original": True}
    assert list(tmp_path.iterdir()) == [result]


@pytest.mark.parametrize(
    "change",
    [
        {"payload_version": True},
        {"payload_version": 1},
        {"warnings": "invalid"},
        {"python_executable": 123},
        {"failure_message": []},
        {
            "environment": {
                **asdict(EnvironmentInfo()),
                "gpus": [{"device": 0, "name": None}],
            }
        },
        {"environment": {**asdict(EnvironmentInfo()), "python_version": 3}},
    ],
)
def test_invalid_result_data_is_rejected(change):
    value = {
        "payload_version": 2,
        "phase": "startup",
        "metrics": None,
        "run_id": "test-run",
        "python_executable": None,
        "environment": asdict(EnvironmentInfo()),
        "warnings": [],
        "failure_message": None,
    }
    value.update(change)
    with pytest.raises(ValueError):
        decode_result(value, "test-run")


def test_malformed_request_rejected_before_execution(tmp_path):
    request, result = request_file(tmp_path, "raise AssertionError('executed')")
    value = json.loads(request.read_text())
    value["args"] = "--not-an-array"
    request.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="string array"):
        child.run_request(request)
    assert not result.exists()
    with pytest.raises(ValueError):
        validate_request({})


@pytest.mark.parametrize(
    "ending,exception,exit_code",
    [
        ("pass", None, None),
        ("raise SystemExit(0)", SystemExit, 0),
        ("raise SystemExit(7)", SystemExit, 7),
        ("raise RuntimeError('boom')", RuntimeError, None),
    ],
)
@pytest.mark.parametrize(
    "collection_error", [None, RuntimeError("probe failed"), SystemExit(19)]
)
def test_finalization_preserves_script_outcome(
    tmp_path, monkeypatch, ending, exception, exit_code, collection_error
):
    request, result = request_file(
        tmp_path, "from pathlib import Path\nPath('executed').touch()\n" + ending
    )
    monkeypatch.setattr(child, "collect_environment", lambda **kw: EnvironmentInfo())
    snapshots = []
    publish = child._publish

    def capture(path, payload):
        snapshots.append(json.loads(json.dumps(payload)))
        publish(path, payload)

    measured = RunMetrics(
        [CUDADeviceMetrics("cuda:0", 0, 123)], "bootstrap_to_script_exit_v1"
    )

    def collect(**kwargs):
        assert (tmp_path / "executed").exists()
        assert snapshots[0]["phase"] == "startup"
        assert snapshots[0]["metrics"] is None
        if collection_error:
            raise collection_error
        return measured

    monkeypatch.setattr(child, "_publish", capture)
    monkeypatch.setattr(child, "collect_cuda_metrics", collect)
    if exception:
        with pytest.raises(exception) as caught:
            child.run_request(request)
        if exception is SystemExit:
            assert caught.value.code == exit_code
        else:
            assert str(caught.value) == "boom"
    else:
        child.run_request(request)
    decoded = decode_result(json.loads(result.read_text()), "test-run")
    assert [item["phase"] for item in snapshots] == ["startup", "final"]
    assert decoded.phase == "final"
    if collection_error:
        assert decoded.metrics.unavailable_reason == "collection_failed"
        assert any(
            "CUDA metric collection failed" in warning for warning in decoded.warnings
        )
    else:
        assert decoded.metrics == measured
    if exception is RuntimeError:
        assert decoded.failure_message == "RuntimeError: boom"
    elif exit_code == 7:
        assert decoded.failure_message == "SystemExit: 7"
    else:
        assert decoded.failure_message is None


@pytest.mark.parametrize(
    "final_error", [None, RuntimeError("refresh failed"), SystemExit(19)]
)
def test_environment_refresh_retains_startup_values(tmp_path, monkeypatch, final_error):
    request, result = request_file(tmp_path, "pass")
    startup = EnvironmentInfo(python_version="startup", pytorch_version="torch-startup")
    observed = iter(
        [
            startup,
            EnvironmentInfo(
                cuda_available=True, gpus=[GPUInfo("cuda:0", "observed GPU")]
            ),
        ]
    )

    def collect(**kwargs):
        value = next(observed)
        assert kwargs.get("include_cuda_inventory", True) is (value is not startup)
        if value is not startup and final_error:
            raise final_error
        return value

    monkeypatch.setattr(child, "collect_environment", collect)
    child.run_request(request)
    decoded = decode_result(json.loads(result.read_text()), "test-run")
    assert decoded.environment.python_version == "startup"
    assert decoded.environment.pytorch_version == "torch-startup"
    if final_error:
        assert decoded.environment == startup
        assert any("Final environment" in warning for warning in decoded.warnings)
    else:
        assert decoded.environment.cuda_available is True
        assert decoded.environment.gpus == [GPUInfo("cuda:0", "observed GPU")]


def test_partial_name_refresh_preserves_known_startup_name(monkeypatch):
    startup = EnvironmentInfo(gpus=[GPUInfo("cuda:0", "known name")])
    monkeypatch.setattr(
        child,
        "collect_environment",
        lambda **kw: EnvironmentInfo(
            gpus=[GPUInfo("cuda:0"), GPUInfo("cuda:1", "new name")]
        ),
    )
    refreshed = child._refresh_environment(startup, RunDiagnostics())
    assert refreshed.gpus == [
        GPUInfo("cuda:0", "known name"),
        GPUInfo("cuda:1", "new name"),
    ]


def test_failed_final_publication_leaves_startup_marker(tmp_path, monkeypatch):
    request, result = request_file(tmp_path, "pass")
    monkeypatch.setattr(child, "collect_environment", lambda **kw: EnvironmentInfo())
    write = child._write_result

    def final_write_fails(path, payload):
        if payload["phase"] == "final":
            raise OSError("disk full")
        write(path, payload)

    monkeypatch.setattr(child, "_write_result", final_write_fails)
    child.run_request(request)
    decoded = decode_result(json.loads(result.read_text()), "test-run")
    assert decoded.phase == "startup"
    assert decoded.metrics is None
