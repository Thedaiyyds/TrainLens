"""Private, ephemeral child transport; independent of the persisted run schema."""

from dataclasses import dataclass, fields
from pathlib import Path

from trainlens.models import CUDADeviceMetrics, EnvironmentInfo, GPUInfo, RunMetrics


@dataclass
class ChildResult:
    environment: EnvironmentInfo
    python_executable: str | None
    warnings: list[str]
    failure_message: str | None
    phase: str
    metrics: RunMetrics | None


def _decode_metrics(value: object) -> RunMetrics:
    if not isinstance(value, dict) or value.keys() != {
        f.name for f in fields(RunMetrics)
    }:
        raise ValueError("Invalid child metrics fields")
    for key in ("measurement_scope", "unavailable_reason"):
        if value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"Invalid child metrics {key}")
    if not isinstance(value["cuda_devices"], list):
        raise ValueError("Invalid child CUDA metrics list")
    devices = []
    identifiers = set()
    for device in value["cuda_devices"]:
        if not isinstance(device, dict) or device.keys() != {
            f.name for f in fields(CUDADeviceMetrics)
        }:
            raise ValueError("Invalid child CUDA metric fields")
        if not isinstance(device["device"], str) or device["device"] in identifiers:
            raise ValueError("Invalid or duplicate child CUDA device identifier")
        identifiers.add(device["device"])
        reason = device["unavailable_reason"]
        if reason is not None and not isinstance(reason, str):
            raise ValueError("Invalid child CUDA metric reason")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if type(device[key]) not in (int, type(None)):
                raise ValueError(f"Invalid child CUDA {key}")
        devices.append(CUDADeviceMetrics(**device))
    return RunMetrics(**{**value, "cuda_devices": devices})


def _strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_request(value: object) -> dict:
    expected = {"run_id", "script", "args", "cwd", "result_path"}
    if not isinstance(value, dict) or value.keys() != expected:
        raise ValueError("Invalid bootstrap request fields")
    if not all(isinstance(value[key], str) for key in expected - {"args"}):
        raise ValueError("Invalid bootstrap request strings")
    if not _strings(value["args"]):
        raise ValueError("Bootstrap arguments must be a string array")
    if not all(Path(value[key]).is_absolute() for key in ("cwd", "result_path")):
        raise ValueError("Bootstrap cwd and result path must be absolute")
    return value


def decode_result(value: object, run_id: str) -> ChildResult:
    expected = {
        "payload_version",
        "run_id",
        "environment",
        "python_executable",
        "warnings",
        "failure_message",
        "phase",
        "metrics",
    }
    if not isinstance(value, dict) or value.keys() != expected:
        raise ValueError("Invalid child result fields")
    if type(value["payload_version"]) is not int or value["payload_version"] != 2:
        raise ValueError("Unsupported child payload version")
    if value["run_id"] != run_id:
        raise ValueError("Child result belongs to a different run")
    if value["phase"] not in ("startup", "final"):
        raise ValueError("Invalid child result phase")
    metrics = None
    if value["phase"] == "final":
        metrics = _decode_metrics(value["metrics"])
    elif value["metrics"] is not None or value["failure_message"] is not None:
        raise ValueError("Startup payload cannot contain final metrics or failure")
    if not _strings(value["warnings"]):
        raise ValueError("Invalid child warnings")
    for key in ("python_executable", "failure_message"):
        if value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"Invalid child {key}")
    env = value["environment"]
    if not isinstance(env, dict) or env.keys() != {
        f.name for f in fields(EnvironmentInfo)
    }:
        raise ValueError("Invalid child environment fields")
    for key in env.keys() - {"cuda_available", "gpus"}:
        if env[key] is not None and not isinstance(env[key], str):
            raise ValueError(f"Invalid child environment {key}")
    if type(env["cuda_available"]) not in (bool, type(None)):
        raise ValueError("Invalid child CUDA availability")
    gpus = env["gpus"]
    restored = None
    if gpus is not None:
        if not isinstance(gpus, list):
            raise ValueError("Invalid child GPU list")
        restored = []
        for gpu in gpus:
            if (
                not isinstance(gpu, dict)
                or gpu.keys() != {"device", "name"}
                or not isinstance(gpu["device"], str)
                or (gpu["name"] is not None and not isinstance(gpu["name"], str))
            ):
                raise ValueError("Invalid child GPU entry")
            restored.append(GPUInfo(**gpu))
    return ChildResult(
        EnvironmentInfo(**{**env, "gpus": restored}),
        value["python_executable"],
        list(value["warnings"]),
        value["failure_message"],
        value["phase"],
        metrics,
    )
