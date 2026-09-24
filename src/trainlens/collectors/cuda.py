"""Observe allocator peaks in the training interpreter after script execution."""

import importlib

from trainlens.models import CUDADeviceMetrics, RunDiagnostics, RunMetrics

MEASUREMENT_SCOPE = "bootstrap_to_script_exit_v1"


def collect_cuda_metrics(*, diagnostics: RunDiagnostics) -> RunMetrics:
    """Read per-device bytes without initializing, resetting or synchronizing CUDA.

    These are the requested PyTorch API observations, not a guarantee of equivalent
    statistics across allocator backends. No parent process should call this helper.
    """
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        diagnostics.collection_warnings.append(
            f"CUDA metrics: PyTorch unavailable: {error}"
        )
        return RunMetrics([], unavailable_reason="pytorch_unavailable")

    try:
        if torch.version.cuda is None:
            return RunMetrics([], unavailable_reason="cuda_unsupported")
        if not torch.cuda.is_initialized():
            return RunMetrics([], unavailable_reason="cuda_not_initialized")
        count = torch.cuda.device_count()
        if type(count) is not int or count <= 0:
            raise ValueError("Initialized CUDA reported no valid device count")
    except Exception as error:
        diagnostics.collection_warnings.append(
            f"CUDA metric collection failed: {error}"
        )
        return RunMetrics([], unavailable_reason="collection_failed")

    devices = []
    for index in range(count):
        peaks = []
        failures = []
        for name in ("max_memory_allocated", "max_memory_reserved"):
            try:
                value = getattr(torch.cuda, name)(index)
                if type(value) is not int or value < 0:
                    raise ValueError("Expected non-negative integer bytes")
                peaks.append(value)
            except Exception as error:
                peaks.append(None)
                message = f"cuda:{index} {name}: {error}"
                failures.append(message)
                diagnostics.collection_warnings.append(message)
        devices.append(
            CUDADeviceMetrics(
                device=f"cuda:{index}",
                peak_allocated_bytes=peaks[0],
                peak_reserved_bytes=peaks[1],
                unavailable_reason="; ".join(failures) or None,
            )
        )
    return RunMetrics(devices, measurement_scope=MEASUREMENT_SCOPE)
