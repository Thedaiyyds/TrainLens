import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from trainlens.collectors.cuda import MEASUREMENT_SCOPE, collect_cuda_metrics
from trainlens.models import CUDADeviceMetrics, RunDiagnostics


def fake_torch(monkeypatch, *, build="12.4", initialized=True, count=1):
    cuda = Mock(
        spec=[
            "is_initialized",
            "device_count",
            "max_memory_allocated",
            "max_memory_reserved",
        ]
    )
    cuda.is_initialized.return_value = initialized
    cuda.device_count.return_value = count
    cuda.max_memory_allocated.return_value = 123
    cuda.max_memory_reserved.return_value = 456
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(cuda=build), cuda=cuda),
    )
    return cuda


def test_torch_absent_is_unavailable_not_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    diagnostics = RunDiagnostics(failure_message="training error")
    metrics = collect_cuda_metrics(diagnostics=diagnostics)
    assert metrics.cuda_devices == []
    assert metrics.unavailable_reason == "pytorch_unavailable"
    assert metrics.measurement_scope is None
    assert diagnostics.collection_warnings
    assert diagnostics.failure_message == "training error"


@pytest.mark.parametrize(
    "build,initialized,reason,expected_calls",
    [
        (None, False, "cuda_unsupported", []),
        ("12.4", False, "cuda_not_initialized", [call.is_initialized()]),
    ],
)
def test_no_cuda_queries_without_initialized_cuda(
    monkeypatch, build, initialized, reason, expected_calls
):
    cuda = fake_torch(monkeypatch, build=build, initialized=initialized)
    metrics = collect_cuda_metrics(diagnostics=RunDiagnostics())
    assert metrics.cuda_devices == []
    assert metrics.unavailable_reason == reason
    assert cuda.mock_calls == expected_calls


@pytest.mark.parametrize("peaks", [(123, 456), (0, 0)])
def test_initialized_device_records_integer_peaks_including_zero(monkeypatch, peaks):
    cuda = fake_torch(monkeypatch)
    cuda.max_memory_allocated.return_value, cuda.max_memory_reserved.return_value = (
        peaks
    )
    diagnostics = RunDiagnostics()
    metrics = collect_cuda_metrics(diagnostics=diagnostics)
    assert metrics.cuda_devices == [CUDADeviceMetrics("cuda:0", *peaks)]
    assert metrics.measurement_scope == MEASUREMENT_SCOPE
    assert metrics.unavailable_reason is None
    assert not diagnostics.collection_warnings
    assert cuda.mock_calls == [
        call.is_initialized(),
        call.device_count(),
        call.max_memory_allocated(0),
        call.max_memory_reserved(0),
    ]


def test_multiple_devices_remain_separate_after_partial_failure(monkeypatch):
    cuda = fake_torch(monkeypatch, count=2)
    cuda.max_memory_allocated.side_effect = [123, 0]
    cuda.max_memory_reserved.side_effect = [RuntimeError("query failed"), 789]
    diagnostics = RunDiagnostics()
    metrics = collect_cuda_metrics(diagnostics=diagnostics)
    first, second = metrics.cuda_devices
    assert first.device == "cuda:0"
    assert first.peak_allocated_bytes == 123
    assert first.peak_reserved_bytes is None
    assert "max_memory_reserved: query failed" in first.unavailable_reason
    assert second == CUDADeviceMetrics("cuda:1", 0, 789)
    assert metrics.unavailable_reason is None
    assert diagnostics.collection_warnings == [first.unavailable_reason]
    assert cuda.max_memory_allocated.call_args_list == [call(0), call(1)]
    assert cuda.max_memory_reserved.call_args_list == [call(0), call(1)]


@pytest.mark.parametrize("query", ["is_initialized", "device_count"])
def test_state_or_enumeration_failure_is_explicit(monkeypatch, query):
    cuda = fake_torch(monkeypatch)
    getattr(cuda, query).side_effect = RuntimeError("probe failed")
    diagnostics = RunDiagnostics()
    metrics = collect_cuda_metrics(diagnostics=diagnostics)
    assert metrics.cuda_devices == []
    assert metrics.unavailable_reason == "collection_failed"
    assert "probe failed" in diagnostics.collection_warnings[0]
    cuda.max_memory_allocated.assert_not_called()
    cuda.max_memory_reserved.assert_not_called()


def test_both_peak_queries_fail_without_fake_zero(monkeypatch):
    cuda = fake_torch(monkeypatch)
    cuda.max_memory_allocated.side_effect = RuntimeError("allocated failed")
    cuda.max_memory_reserved.side_effect = RuntimeError("reserved failed")
    diagnostics = RunDiagnostics()
    metrics = collect_cuda_metrics(diagnostics=diagnostics)
    [device] = metrics.cuda_devices
    assert device.peak_allocated_bytes is device.peak_reserved_bytes is None
    assert "allocated failed" in device.unavailable_reason
    assert "reserved failed" in device.unavailable_reason
    assert len(diagnostics.collection_warnings) == 2


@pytest.mark.parametrize("value", [True, -1, 1.5, None])
def test_invalid_api_values_are_missing_not_coerced(monkeypatch, value):
    cuda = fake_torch(monkeypatch)
    cuda.max_memory_allocated.return_value = value
    diagnostics = RunDiagnostics()
    [device] = collect_cuda_metrics(diagnostics=diagnostics).cuda_devices
    assert device.peak_allocated_bytes is None
    assert device.peak_reserved_bytes == 456
    assert device.unavailable_reason and diagnostics.collection_warnings
