import json
from dataclasses import asdict
from pathlib import Path

import pytest

from trainlens import _bootstrap_child as child
from trainlens._run_payload import decode_result, validate_request
from trainlens.models import EnvironmentInfo


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

    env, executable, warnings, failure = decode_result(
        json.loads(result.read_text()), "test-run"
    )
    assert (tmp_path / "executed").exists()
    assert env == EnvironmentInfo()
    assert executable
    assert any("broken optional collector" in warning for warning in warnings)
    assert failure is None


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
        {"payload_version": 2},
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
        "payload_version": 1,
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
