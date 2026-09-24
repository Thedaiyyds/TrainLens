from copy import deepcopy
from dataclasses import asdict

import pytest

from trainlens._run_payload import decode_result
from trainlens.models import CUDADeviceMetrics, EnvironmentInfo, RunMetrics


def final_payload():
    return {
        "payload_version": 2,
        "run_id": "run-id",
        "phase": "final",
        "environment": asdict(EnvironmentInfo()),
        "python_executable": "/selected/python",
        "warnings": [],
        "failure_message": None,
        "metrics": asdict(
            RunMetrics(
                [CUDADeviceMetrics("cuda:0", 0, None, "reserved query failed")],
                "bootstrap_to_script_exit_v1",
            )
        ),
    }


def test_final_decode_preserves_zero_missing_and_input():
    value = final_payload()
    before = deepcopy(value)
    decoded = decode_result(value, "run-id")
    assert decoded.phase == "final"
    assert asdict(decoded.metrics) == value["metrics"]
    assert decoded.metrics.cuda_devices[0].peak_allocated_bytes == 0
    assert decoded.metrics.cuda_devices[0].peak_reserved_bytes is None
    assert value == before


@pytest.mark.parametrize(
    "change",
    [
        {"phase": "finished"},
        {"phase": None},
        {"phase": []},
        {"payload_version": 1},
        {"payload_version": 3},
        {"payload_version": True},
        {"run_id": "different"},
        {"extra": None},
        {"metrics": None},
        {"metrics": []},
        {"metrics": {}},
        {"warnings": [123]},
        {"phase": "startup"},
    ],
)
def test_invalid_final_payload_is_rejected(change):
    value = final_payload()
    value.update(change)
    with pytest.raises(ValueError):
        decode_result(value, "run-id")


@pytest.mark.parametrize("field", list(final_payload()))
def test_missing_payload_field_is_rejected(field):
    value = final_payload()
    del value[field]
    with pytest.raises(ValueError):
        decode_result(value, "run-id")


@pytest.mark.parametrize(
    "change",
    [
        {"cuda_devices": None},
        {"cuda_devices": {}},
        {"cuda_devices": [None]},
        {"cuda_devices": []},
        {"unavailable_reason": "contradicts devices"},
        {"unavailable_reason": False},
        {"measurement_scope": 1},
        {"extra": None},
    ],
)
def test_invalid_metrics_structure_or_invariants_rejected(change):
    value = final_payload()
    value["metrics"].update(change)
    with pytest.raises(ValueError):
        decode_result(value, "run-id")


@pytest.mark.parametrize(
    "change",
    [
        {"device": None},
        {"device": 0},
        {"peak_allocated_bytes": -1},
        {"peak_allocated_bytes": True},
        {"peak_allocated_bytes": 1.5},
        {"peak_reserved_bytes": "12"},
        {"unavailable_reason": None},
        {"unavailable_reason": ""},
        {"unavailable_reason": 7},
        {"peak_reserved_bytes": 12},
        {"extra": None},
    ],
)
def test_invalid_device_types_and_invariants_rejected(change):
    value = final_payload()
    value["metrics"]["cuda_devices"][0].update(change)
    with pytest.raises(ValueError):
        decode_result(value, "run-id")


@pytest.mark.parametrize("level", ["metrics", "device"])
def test_missing_nested_fields_rejected(level):
    value = final_payload()
    target = (
        value["metrics"] if level == "metrics" else value["metrics"]["cuda_devices"][0]
    )
    del target["unavailable_reason"]
    with pytest.raises(ValueError):
        decode_result(value, "run-id")


def test_duplicate_device_identifiers_rejected():
    value = final_payload()
    value["metrics"]["cuda_devices"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        decode_result(value, "run-id")
