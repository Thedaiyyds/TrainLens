"""Observe the calling interpreter, eventually inside the training bootstrap."""

import importlib
import platform
import sys

from trainlens.models import EnvironmentInfo, GPUInfo, RunDiagnostics


def collect_python_executable() -> str | None:
    """Value for RunRecord.python_executable in the calling Python process."""
    return sys.executable or None


def collect_environment(
    *, diagnostics: RunDiagnostics, include_cuda_inventory: bool = True
) -> EnvironmentInfo:
    """Collect current-process metadata without initializing PyTorch CUDA.

    Call from the selected training interpreter, not its CLI supervisor. CUDA
    builds defer availability/device queries until CUDA is already initialized;
    a later call after script execution can fill those fields. The bootstrap also
    disables inventory explicitly at startup, even if a Python startup hook has
    already initialized CUDA. Missing metadata
    adds collection warnings without changing the training failure message.
    """
    result = EnvironmentInfo()
    for field, query in (
        ("python_version", platform.python_version),
        ("os", platform.platform),
        ("architecture", platform.machine),
    ):
        try:
            value = query()
            setattr(result, field, value or None)
            if not value:
                diagnostics.collection_warnings.append(f"{field}: unavailable")
        except Exception as error:
            diagnostics.collection_warnings.append(f"{field}: {error}")

    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        diagnostics.collection_warnings.append(f"PyTorch import unavailable: {error}")
        return result

    try:
        result.pytorch_version = str(torch.__version__)
        result.cuda_build_version = torch.version.cuda
    except Exception as error:
        diagnostics.collection_warnings.append(f"PyTorch build metadata: {error}")
        return result

    if result.cuda_build_version is None:
        result.cuda_available = False
        result.gpus = []
        return result

    if not include_cuda_inventory:
        diagnostics.collection_warnings.append(
            "Startup CUDA availability and GPU inventory deferred until script exit."
        )
        return result

    try:
        if not torch.cuda.is_initialized():
            diagnostics.collection_warnings.append(
                "CUDA availability and GPU inventory deferred: CUDA is not initialized."
            )
            return result
        result.cuda_available = torch.cuda.is_available()
        if not result.cuda_available:
            result.gpus = []
            return result
        count = torch.cuda.device_count()
        result.gpus = []
        for index in range(count):
            gpu = GPUInfo(device=f"cuda:{index}")
            result.gpus.append(gpu)
            try:
                gpu.name = torch.cuda.get_device_name(index)
            except Exception as error:
                diagnostics.collection_warnings.append(f"{gpu.device} name: {error}")
    except Exception as error:
        diagnostics.collection_warnings.append(f"CUDA environment metadata: {error}")
    return result
