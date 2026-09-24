"""Private module entry point, executed by the selected training interpreter."""

import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from trainlens._run_payload import validate_request
from trainlens.bootstrap import execute_script
from trainlens.collectors.environment import (
    collect_environment,
    collect_python_executable,
)
from trainlens.models import EnvironmentInfo, RunDiagnostics


def _write_result(path: Path, payload: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish(path: Path, payload: dict) -> None:
    # Optional transport must never replace a script exception or exit outcome.
    try:
        _write_result(path, payload)
    except Exception as error:
        try:
            print(
                f"TrainLens: could not write child metadata: {error}", file=sys.stderr
            )
        except Exception:
            pass  # The user may have closed stderr; the parent detects missing data.


def run_request(request_path: str | Path) -> None:
    request = validate_request(
        json.loads(Path(request_path).read_text(encoding="utf-8"))
    )
    diagnostics = RunDiagnostics()
    environment = EnvironmentInfo()
    try:
        environment = collect_environment(diagnostics=diagnostics)
    except Exception as error:
        diagnostics.collection_warnings.append(
            f"Environment collection failed: {error}"
        )
    payload = {
        "payload_version": 1,
        "run_id": request["run_id"],
        "python_executable": collect_python_executable(),
        "environment": asdict(environment),
        "warnings": diagnostics.collection_warnings,
        "failure_message": None,
    }
    result_path = Path(request["result_path"])
    _publish(result_path, payload)
    try:
        execute_script(
            request["script"], request["args"], invocation_cwd=request["cwd"]
        )
    except SystemExit as error:
        if error.code is not None and error.code != 0:
            payload["failure_message"] = f"SystemExit: {error.code}"
        raise
    except BaseException as error:
        payload["failure_message"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _publish(result_path, payload)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("TrainLens bootstrap expects one private request path")
    run_request(sys.argv[1])
