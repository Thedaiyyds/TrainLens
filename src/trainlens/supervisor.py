"""Parent-owned timing, outcome and persistence for the script-file workflow."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from trainlens._run_payload import decode_result
from trainlens.collectors.git import collect_git
from trainlens.models import RunRecord
from trainlens.storage import RunStore


class RunStorageError(RuntimeError):
    """Recording failed; callers must not report a normally recorded run."""


def _validate_command(command: list[str]) -> None:
    if len(command) < 2 or not all(isinstance(arg, str) for arg in command):
        raise ValueError("Expected PYTHON SCRIPT.py [SCRIPT_ARGS...] after --")
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", Path(command[0]).name):
        raise ValueError(
            "Only Python script commands are supported (python, python3, python3.x)"
        )
    if command[1].startswith("-") or Path(command[1]).suffix != ".py":
        raise ValueError(
            "Expected a .py script; -m, -c and stdin programs are unsupported"
        )
    if any("\0" in arg for arg in command):
        raise ValueError("Command arguments cannot contain NUL characters")


def _resolve_python(executable: str, cwd: Path) -> str:
    # Preserve venv symlinks: resolving to their target would select the base Python.
    if os.path.dirname(executable):
        candidate = str(cwd / executable)
        found = shutil.which(candidate)
    else:
        search_path = os.pathsep.join(str(cwd / entry) for entry in os.get_exec_path())
        found = shutil.which(executable, path=search_path)
    if found is None:
        raise FileNotFoundError(f"Selected Python executable not found: {executable}")
    return os.path.abspath(found)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def supervise_run(
    name: str, command: list[str], *, invocation_cwd: str | Path
) -> RunRecord:
    """Run one child with inherited streams, then save one final record.

    Missing metadata does not change the child's observed exit status. Storage
    failures raise RunStorageError even after successful training. No concurrency
    reservation or signal-forwarding policy is provided here.
    """
    command = list(command)
    _validate_command(command)
    if not name.strip():
        raise ValueError("Run name must not be empty")
    cwd = Path(invocation_cwd).resolve()
    record = RunRecord(
        run_id=str(uuid.uuid4()),
        name=name,
        trainlens_version=version("trainlens"),
        command=command,
        cwd=str(cwd),
        git=collect_git(cwd),
    )
    store = RunStore(cwd)
    try:
        store.check_available(record.run_id, name)
        store.runs_dir.mkdir(parents=True, exist_ok=True)
        # Check actual write access rather than trusting permission bits alone.
        with tempfile.TemporaryFile(dir=store.runs_dir):
            pass
    except (OSError, ValueError) as error:
        raise RunStorageError(
            f"Storage preflight failed before launch: {error}"
        ) from error

    record.started_at = _now()
    try:
        python = _resolve_python(command[0], cwd)
        record.python_executable = python
        script = cwd / command[1]
        if not script.is_file():
            raise FileNotFoundError(f"Training script not found: {script}")
        with tempfile.TemporaryDirectory(prefix="trainlens-") as temporary:
            request_path = Path(temporary) / "request.json"
            result_path = Path(temporary) / "result.json"
            request_path.write_text(
                json.dumps(
                    {
                        "run_id": record.run_id,
                        "script": command[1],
                        "args": command[2:],
                        "cwd": str(cwd),
                        "result_path": str(result_path),
                    }
                ),
                encoding="utf-8",
            )
            started = time.monotonic()
            completed = subprocess.run(
                [python, "-m", "trainlens._bootstrap_child", str(request_path)],
                cwd=cwd,
                check=False,
            )
            record.runtime_seconds = time.monotonic() - started
            record.ended_at = _now()
            record.exit_code = completed.returncode
            record.status = "succeeded" if completed.returncode == 0 else "failed"
            try:
                payload = decode_result(
                    json.loads(result_path.read_text(encoding="utf-8")), record.run_id
                )
            except (OSError, ValueError) as error:
                record.diagnostics.collection_warnings.append(
                    f"Child metadata unavailable: {error}. The selected interpreter must "
                    "have TrainLens importable; bootstrap startup or payload delivery "
                    "may have failed."
                )
            else:
                environment, executable, warnings, failure_message = payload
                record.environment = environment
                if executable is not None:
                    record.python_executable = executable
                record.diagnostics.collection_warnings.extend(warnings)
                record.diagnostics.failure_message = failure_message
            if record.status == "failed" and not record.diagnostics.failure_message:
                record.diagnostics.failure_message = (
                    f"Child exited with code {record.exit_code}"
                )
    except (OSError, ValueError) as error:
        if record.exit_code is None:
            record.status = "failed"
            record.diagnostics.failure_message = f"Launch/bootstrap failure: {error}"
        else:
            record.diagnostics.collection_warnings.append(
                f"Temporary transport cleanup failed: {error}"
            )
    finally:
        if record.ended_at is None:
            record.ended_at = _now()
    try:
        store.save(record)
    except (OSError, ValueError) as error:
        raise RunStorageError(
            f"Could not save run {record.run_id} (child exit={record.exit_code}): {error}"
        ) from error
    return record
