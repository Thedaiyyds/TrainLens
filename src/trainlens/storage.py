"""Local schema-1 RunRecord storage without collection or lifecycle management."""

import json
import math
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainlens.models import (
    CUDADeviceMetrics,
    EnvironmentInfo,
    GitInfo,
    GPUInfo,
    RunDiagnostics,
    RunMetrics,
    RunRecord,
)


def _object(value: object, model: type) -> dict[str, Any]:
    """Require the complete existing model fields; never silently discard data."""
    if not isinstance(value, dict):
        raise ValueError(f"Invalid RunRecord data: {model.__name__} must be an object.")
    expected = {item.name for item in fields(model)}
    if value.keys() != expected:
        raise ValueError(
            f"Invalid RunRecord data: incorrect fields for {model.__name__}."
        )
    return dict(value)


def _expect(value: object, expected: type | tuple[type, ...], field: str) -> None:
    types = expected if isinstance(expected, tuple) else (expected,)
    # JSON booleans must not pass as integers, even though bool subclasses int.
    if type(value) not in types:
        raise ValueError(f"Invalid RunRecord data: incorrect type for {field}.")


def _strings(values: dict[str, Any], names: tuple[str, ...], nullable: bool) -> None:
    for name in names:
        _expect(values[name], (str, type(None)) if nullable else str, name)


def _string_list(value: object, field: str) -> None:
    _expect(value, list, field)
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"Invalid RunRecord data: {field} must contain strings.")


def _restore_record(value: object) -> RunRecord:
    """Restore only the current schema with explicit nested model construction."""
    if not isinstance(value, dict):
        raise ValueError("Invalid RunRecord data: RunRecord must be an object.")
    version = value.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError(f"Unsupported RunRecord schema_version: {version!r}.")
    data = _object(value, RunRecord)
    data.pop("schema_version")

    _strings(data, ("run_id", "name", "trainlens_version", "cwd", "status"), False)
    _strings(data, ("python_executable", "started_at", "ended_at"), True)
    _string_list(data["command"], "command")
    _expect(data["runtime_seconds"], (int, float, type(None)), "runtime_seconds")
    if isinstance(data["runtime_seconds"], float) and not math.isfinite(
        data["runtime_seconds"]
    ):
        raise ValueError("Invalid RunRecord data: runtime_seconds must be finite.")
    _expect(data["exit_code"], (int, type(None)), "exit_code")

    environment = _object(data.pop("environment"), EnvironmentInfo)
    _strings(
        environment,
        (
            "python_version",
            "os",
            "architecture",
            "pytorch_version",
            "cuda_build_version",
        ),
        True,
    )
    _expect(environment["cuda_available"], (bool, type(None)), "cuda_available")
    gpus = environment["gpus"]
    if gpus is not None:
        _expect(gpus, list, "gpus")
        restored_gpus = []
        for value in gpus:
            gpu = _object(value, GPUInfo)
            _strings(gpu, ("device",), False)
            _strings(gpu, ("name",), True)
            restored_gpus.append(GPUInfo(**gpu))
        environment["gpus"] = restored_gpus

    git = _object(data.pop("git"), GitInfo)
    _strings(git, ("commit", "unavailable_reason"), True)
    _expect(git["dirty"], (bool, type(None)), "git.dirty")

    metrics = _object(data.pop("metrics"), RunMetrics)
    _strings(metrics, ("measurement_scope", "unavailable_reason"), True)
    _expect(metrics["cuda_devices"], list, "cuda_devices")
    devices = []
    for value in metrics["cuda_devices"]:
        device = _object(value, CUDADeviceMetrics)
        _strings(device, ("device",), False)
        _strings(device, ("unavailable_reason",), True)
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            _expect(device[name], (int, type(None)), name)
        devices.append(CUDADeviceMetrics(**device))
    metrics["cuda_devices"] = devices

    diagnostics = _object(data.pop("diagnostics"), RunDiagnostics)
    _strings(diagnostics, ("failure_message",), True)
    _string_list(diagnostics["collection_warnings"], "collection_warnings")

    return RunRecord(
        **data,
        environment=EnvironmentInfo(**environment),
        git=GitInfo(**git),
        metrics=RunMetrics(**metrics),
        diagnostics=RunDiagnostics(**diagnostics),
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}.")


def _start_time(record: RunRecord) -> tuple[bool, datetime]:
    if record.started_at is None:
        return False, datetime.min.replace(tzinfo=timezone.utc)
    try:
        # Accept UTC's Z spelling on Python 3.10 as well as numeric offsets.
        started = datetime.fromisoformat(record.started_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"Invalid started_at for run {record.run_id!r}: {record.started_at!r}."
        ) from error
    # Saved timestamps represent UTC; never infer the reader's local timezone.
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return True, started


class RunStore:
    """Save new runs and load by ID under project_dir/.trainlens/runs.

    The project directory defaults to the construction-time working directory.
    Missing reads never create directories. Save rejects existing IDs and names;
    it does not update records or coordinate concurrent writers. FileNotFoundError
    means a missing run/store, FileExistsError a duplicate, JSONDecodeError invalid
    JSON syntax, and ValueError an unsafe identity or unrecoverable record schema.
    Other filesystem errors propagate as OSError with their original path details.
    """

    def __init__(self, project_dir: str | Path | None = None) -> None:
        self.project_dir = Path(project_dir if project_dir is not None else Path.cwd())
        self.project_dir = self.project_dir.resolve()
        self.runs_dir = self.project_dir / ".trainlens" / "runs"

    def _path(self, run_id: str) -> Path:
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in (".", "..")
            or any(character in run_id for character in ("/", "\\", "\0"))
        ):
            raise ValueError("Run ID must be a non-empty filename component.")
        directory = self.runs_dir.resolve()
        if not directory.is_relative_to(self.project_dir):
            raise ValueError(
                "Storage directory resolves outside the project directory."
            )
        path = self.runs_dir / f"{run_id}.json"
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Run path resolves outside the storage directory.")
        return path

    def save(self, record: RunRecord) -> Path:
        """Save a complete UTF-8 JSON record; never replace an existing record."""
        data = asdict(record)
        # Recheck mutable dataclasses before touching disk, without changing them.
        snapshot = _restore_record(data)
        text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        text.encode("utf-8")
        path = self._path(snapshot.run_id)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.check_available(snapshot.run_id, snapshot.name)

        # Exclusive creation also rejects a duplicate ID appearing after the check.
        stream = path.open("x", encoding="utf-8")
        try:
            with stream:
                stream.write(text)
        except OSError:
            path.unlink(missing_ok=True)
            raise
        return path

    def check_available(self, run_id: str, name: str) -> None:
        """Read-only preflight; not a reservation or a concurrency guarantee."""
        path = self._path(run_id)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Run ID already exists: {run_id!r} ({path}).")
        if not self.runs_dir.exists():
            return
        for existing in sorted(self.runs_dir.iterdir()):
            if existing.suffix != ".json":
                continue
            if self.load(existing.stem).name == name:
                raise FileExistsError(f"Run name already exists: {name!r}.")

    def load(self, run_id: str) -> RunRecord:
        """Load schema 1 as nested dataclasses, without modifying the saved file."""
        path = self._path(run_id)
        with path.open(encoding="utf-8") as stream:
            try:
                data = json.load(stream, parse_constant=_reject_constant)
            except json.JSONDecodeError as error:
                raise json.JSONDecodeError(
                    f"Invalid JSON in {path}: {error.msg}", error.doc, error.pos
                ) from error
        try:
            record = _restore_record(data)
        except ValueError as error:
            raise ValueError(
                f"Cannot restore RunRecord from {path}: {error}"
            ) from error
        if record.run_id != run_id:
            raise ValueError(f"Run identity in {path} does not match {run_id!r}.")
        return record

    def list_records(self) -> list[RunRecord]:
        """Read runs newest first, unknown starts last, breaking ties by run ID.

        An absent store is empty. Invalid records and filesystem errors propagate;
        enumeration never creates directories or repairs saved data.
        """
        if not self.runs_dir.resolve().is_relative_to(self.project_dir):
            raise ValueError(
                "Storage directory resolves outside the project directory."
            )
        try:
            paths = sorted(self.runs_dir.iterdir())
        except FileNotFoundError:
            return []
        records = [self.load(path.stem) for path in paths if path.suffix == ".json"]
        # Stable sorting keeps filename/run-ID order for equal or missing starts.
        return sorted(records, key=_start_time, reverse=True)
