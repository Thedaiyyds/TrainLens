"""Read pre-run Git metadata from an explicitly supplied invocation directory."""

import os
import subprocess
from pathlib import Path

from trainlens.models import GitInfo


def _git(cwd: Path, *args: str) -> str:
    # Inherited Git overrides must not redirect collection to another repository
    # or reinterpret our literal pathspec. No caller environment is modified.
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(LC_ALL="C", GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        timeout=10,
        env=env,
    ).stdout


def collect_git(invocation_cwd: str | Path) -> GitInfo:
    """Collect commit and whole-worktree status, excluding this run's store.

    Tracked, staged and untracked changes count as dirty. Only the supplied
    invocation directory's .trainlens subtree is excluded. Failures preserve
    whatever was collected and explain unknown fields through unavailable_reason.
    """
    result = GitInfo()
    try:
        cwd = Path(invocation_cwd).resolve()
        root = Path(_git(cwd, "rev-parse", "--show-toplevel").rstrip("\n")).resolve()
        store = (cwd.relative_to(root) / ".trainlens").as_posix()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result.unavailable_reason = f"Git repository unavailable: {error}"
        return result

    failures = []
    try:
        result.commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    except (OSError, subprocess.SubprocessError) as error:
        failures.append(f"Git commit unavailable: {error}")
    try:
        result.dirty = bool(
            _git(
                root,
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
                "--",
                ".",
                f":(top,exclude,literal){store}",
            )
        )
    except (OSError, subprocess.SubprocessError) as error:
        failures.append(f"Git status unavailable: {error}")
    result.unavailable_reason = "; ".join(failures) or None
    return result
