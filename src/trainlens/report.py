"""Pure Markdown rendering of saved records; no collection or execution."""

import html
import json

from trainlens.models import CUDADeviceMetrics, RunRecord


def _cell(value: object) -> str:
    if value is None:
        return "N/A"
    text = html.escape(str(value), quote=False)
    for character in ("\\", "|", "`", "*", "_", "[", "]"):
        text = text.replace(character, "\\" + character)
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
        .replace("\t", "\\t")
    )


def _table(headers: list[str], rows: list[tuple]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_cell(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _number(value: int | float, *, signed: bool = False) -> str:
    text = str(value) if isinstance(value, int) else format(value, ".6g")
    return ("+" if signed and value > 0 else "") + text


def _numeric(
    baseline: int | float | None,
    experiment: int | float | None,
    unit: str,
    incompatible: str | None = None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    values = tuple(
        None if value is None else f"{_number(value)} {unit}"
        for value in (baseline, experiment)
    )
    if incompatible or baseline is None or experiment is None:
        return (*values, None, None, incompatible or "Value unavailable")
    delta = experiment - baseline
    percentage = (
        None if baseline == 0 else f"{_number(delta / baseline * 100, signed=True)}%"
    )
    note = "Percentage unavailable: baseline is zero" if baseline == 0 else None
    return (*values, f"{_number(delta, signed=True)} {unit}", percentage, note)


def _devices(record: RunRecord) -> dict[str, CUDADeviceMetrics]:
    devices = {}
    for entry in record.metrics.cuda_devices:
        if entry.device in devices:
            raise ValueError(
                f"Duplicate CUDA device identifier {entry.device!r} in run {record.run_id!r}."
            )
        devices[entry.device] = entry
    return devices


def _cuda_rows(baseline: RunRecord, experiment: RunRecord) -> list[tuple]:
    left, right = _devices(baseline), _devices(experiment)
    rows = []
    for device in sorted(left.keys() | right.keys()) or [None]:
        first, second = left.get(device), right.get(device)
        scopes = (
            baseline.metrics.measurement_scope,
            experiment.metrics.measurement_scope,
        )
        if first is None or second is None:
            incompatible = "Matching saved device entry unavailable"
        elif None in scopes:
            incompatible = "Measurement scope unavailable"
        elif scopes[0] != scopes[1]:
            incompatible = "Measurement scopes differ"
        else:
            incompatible = None
        for field, label in (
            ("peak_allocated_bytes", "Peak allocated"),
            ("peak_reserved_bytes", "Peak reserved"),
        ):
            values = _numeric(
                getattr(first, field) if first else None,
                getattr(second, field) if second else None,
                "bytes",
                incompatible,
            )
            notes = [values[-1]] if values[-1] else []
            for role, entry in (("Baseline", first), ("Experiment", second)):
                if entry is not None and entry.unavailable_reason is not None:
                    notes.append(f"{role}: {entry.unavailable_reason}")
            rows.append((device, label, *values[:-1], "; ".join(notes) or None))
    return rows


def _gpus(record: RunRecord) -> str | None:
    if record.environment.gpus is None:
        return None
    return (
        "\n".join(
            f"{gpu.device}: {gpu.name if gpu.name is not None else 'N/A'}"
            for gpu in record.environment.gpus
        )
        or "None recorded"
    )


def render_comparison(baseline: RunRecord, experiment: RunRecord) -> str:
    """Describe historical observations with experiment-minus-baseline changes."""
    pair = (baseline, experiment)
    sections = [
        "# TrainLens Comparison",
        "Columns are baseline then experiment. Delta = experiment - baseline; "
        "percentage = delta / baseline × 100. Missing or incomparable values are N/A. "
        "This report describes observations and does not select a winner.",
    ]

    def section(title: str, rows: list[tuple]) -> None:
        sections.extend(
            (f"## {title}", _table(["Field", "Baseline", "Experiment"], rows))
        )

    section(
        "Runs",
        [
            (label, *(getattr(record, field) for record in pair))
            for field, label in (
                ("run_id", "Run ID"),
                ("name", "Name"),
                ("trainlens_version", "TrainLens version"),
                ("schema_version", "Schema version"),
            )
        ],
    )
    section(
        "Execution",
        [
            (
                "Command (argument array)",
                *(json.dumps(record.command, ensure_ascii=False) for record in pair),
            ),
            ("Invocation cwd", *(record.cwd for record in pair)),
            ("Python executable", *(record.python_executable for record in pair)),
        ],
    )
    section(
        "Lifecycle",
        [
            (label, *(getattr(record, field) for record in pair))
            for field, label in (
                ("started_at", "Started"),
                ("ended_at", "Ended"),
                ("status", "Status"),
                ("exit_code", "Exit code"),
            )
        ],
    )
    environment_rows = [
        (label, *(getattr(record.environment, field) for record in pair))
        for field, label in (
            ("python_version", "Python version"),
            ("os", "OS"),
            ("architecture", "Architecture"),
            ("pytorch_version", "PyTorch version"),
            ("cuda_build_version", "CUDA build version"),
            ("cuda_available", "CUDA available"),
        )
    ]
    environment_rows.append(("GPUs", *(_gpus(record) for record in pair)))
    section("Environment", environment_rows)
    section(
        "Git",
        [
            (label, *(getattr(record.git, field) for record in pair))
            for field, label in (
                ("commit", "Commit"),
                ("dirty", "Dirty"),
                ("unavailable_reason", "Unavailable reason"),
            )
        ],
    )
    sections.extend(
        (
            "## Metrics",
            "Runtime is parent-observed process wall time, including bootstrap overhead; "
            "it is not isolated training-loop or GPU time.",
            _table(
                ["Metric", "Baseline", "Experiment", "Delta", "Change", "Notes"],
                [
                    (
                        "Runtime",
                        *_numeric(
                            baseline.runtime_seconds, experiment.runtime_seconds, "s"
                        ),
                    ),
                ],
            ),
            "### CUDA allocator peaks",
            "Rows match exact saved device identifiers only; hardware identity is not inferred. "
            "Numeric comparisons require matching known measurement scopes. "
            "Device peaks are not aggregated.",
            _table(
                ["Field", "Baseline", "Experiment"],
                [
                    (
                        "Measurement scope",
                        *(record.metrics.measurement_scope for record in pair),
                    ),
                    (
                        "Unavailable reason",
                        *(record.metrics.unavailable_reason for record in pair),
                    ),
                ],
            ),
            _table(
                [
                    "Device",
                    "Metric",
                    "Baseline",
                    "Experiment",
                    "Delta",
                    "Change",
                    "Notes",
                ],
                _cuda_rows(baseline, experiment),
            ),
        )
    )
    section(
        "Diagnostics",
        [
            (
                "Failure message",
                *(record.diagnostics.failure_message for record in pair),
            ),
            (
                "Collection warnings",
                *(
                    "\n".join(record.diagnostics.collection_warnings) or "None"
                    for record in pair
                ),
            ),
        ],
    )
    return "\n\n".join(sections) + "\n"
