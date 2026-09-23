"""Internal script execution in the future instrumentation process."""

import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path


def execute_script(
    script_path: str | Path,
    args: Sequence[str],
    *,
    invocation_cwd: str | Path,
) -> None:
    """Execute a Python source script as __main__ in this interpreter.

    Relative script paths are resolved against invocation_cwd. Arguments remain
    separate strings. SystemExit and user exceptions propagate unchanged; normal
    return produces None. Standard streams are left alone.

    This primitive is for a one-shot bootstrap, not concurrent or isolated runs.
    Temporary argv, import path and cwd changes are restored on unwinding, but
    other script side effects (including imported modules) remain in this process.
    """
    cwd = Path(invocation_cwd).resolve()
    script = (cwd / script_path).resolve()
    if script.suffix != ".py":
        raise ValueError("Bootstrap expects a Python .py script.")
    if not script.is_file():
        raise FileNotFoundError(f"Python script is not a file: {script}")

    previous_cwd = Path.cwd()
    previous_argv = sys.argv
    previous_path = sys.path
    try:
        os.chdir(cwd)
        sys.argv = [str(script), *args]
        sys.path = [str(script.parent), *previous_path]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path = previous_path
        os.chdir(previous_cwd)
