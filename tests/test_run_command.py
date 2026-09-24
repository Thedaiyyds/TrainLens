"""Exercise the public run command across the real parent/child boundary."""

import json
import os
import subprocess
import sys
import venv
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import pytest

from trainlens.models import RunRecord
from trainlens.storage import RunStore


def clean_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_") and key not in ("PYTHONPATH", "PYTHONHOME")
    }
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return env


def cli(cwd: Path, *args: str, stdin: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "from trainlens.cli import app; app()", *args],
        cwd=cwd,
        env=clean_env(),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )


def records(cwd: Path) -> list[RunRecord]:
    store = RunStore(cwd)
    return [store.load(path.stem) for path in sorted(store.runs_dir.glob("*.json"))]


def test_success_preserves_command_cwd_arguments_and_standard_streams(tmp_path):
    script = tmp_path / "training script.py"
    script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'input': input(), 'pid': os.getpid()}))\n"
        "print('training stderr', file=sys.stderr)\n"
        "Path('changed-cwd').mkdir()\n"
        "os.chdir('changed-cwd')\n",
        encoding="utf-8",
    )
    args = ["--epochs", "5", "hello world", "", "--help", "--name", "script name"]
    command = [sys.executable, script.name, *args]

    result = cli(
        tmp_path, "run", "--name", "baseline 名称", "--", *command, stdin="user input\n"
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["args"] == args
    assert output["cwd"] == str(tmp_path)
    assert output["input"] == "user input"
    assert output["pid"] != os.getpid()
    assert "training stderr" in result.stderr
    assert not (tmp_path / "changed-cwd" / ".trainlens").exists()
    [record] = records(tmp_path)
    assert record.command == command
    assert record.cwd == str(tmp_path)
    assert record.name == "baseline 名称"
    assert record.run_id
    assert record.schema_version == 1
    assert record.trainlens_version == version("trainlens")
    assert record.status == "succeeded"
    assert record.exit_code == 0
    assert record.runtime_seconds >= 0
    start, end = map(datetime.fromisoformat, (record.started_at, record.ended_at))
    assert start.utcoffset().total_seconds() == end.utcoffset().total_seconds() == 0
    assert end >= start
    assert record.python_executable == sys.executable
    assert record.environment.python_version
    assert record.environment.os
    assert record.environment.architecture
    assert record.metrics.cuda_devices == []
    assert record.metrics.unavailable_reason


@pytest.mark.parametrize(
    ("source", "exit_code", "status", "error"),
    [
        ("raise SystemExit(0)", 0, "succeeded", None),
        ("raise SystemExit(7)", 7, "failed", None),
        ("raise RuntimeError('training exploded')", 1, "failed", "training exploded"),
    ],
)
def test_script_outcome_and_traceback_are_preserved(
    tmp_path, source, exit_code, status, error
):
    (tmp_path / "train.py").write_text(source, encoding="utf-8")

    result = cli(tmp_path, "run", "--name", "outcome", "--", sys.executable, "train.py")

    assert (result.returncode == 0) == (exit_code == 0)
    [record] = records(tmp_path)
    assert record.status == status
    assert record.exit_code == exit_code
    assert record.environment.python_version
    if status == "failed":
        assert record.diagnostics.failure_message
    if error:
        assert "Traceback (most recent call last)" in result.stderr
        assert error in result.stderr
        assert error in record.diagnostics.failure_message


def test_duplicate_name_prevents_execution_and_distinct_runs_keep_distinct_ids(
    tmp_path,
):
    (tmp_path / "train.py").write_text(
        "from pathlib import Path\n"
        "with Path('executions').open('a') as stream: stream.write('run\\n')\n",
        encoding="utf-8",
    )
    args = ["--", sys.executable, "train.py"]
    first = cli(tmp_path, "run", "--name", "baseline", *args)
    assert first.returncode == 0, first.stderr
    saved = records(tmp_path)

    duplicate = cli(tmp_path, "run", "--name", "baseline", *args)

    assert duplicate.returncode != 0
    assert "baseline" in duplicate.stderr
    assert records(tmp_path) == saved
    assert (tmp_path / "executions").read_text() == "run\n"
    second = cli(tmp_path, "run", "--name", "experiment", *args)
    assert second.returncode == 0, second.stderr
    runs = records(tmp_path)
    assert len({record.run_id for record in runs}) == 2
    assert {record.name for record in runs} == {"baseline", "experiment"}
    assert saved[0] in runs


def test_git_state_is_collected_before_the_training_script_changes_files(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
            env=clean_env(),
        ).stdout.strip()

    (tmp_path / "train.py").write_text(
        "from pathlib import Path\nPath('tracked.txt').write_text('changed')\n",
        encoding="utf-8",
    )
    (tmp_path / "tracked.txt").write_text("original", encoding="utf-8")
    git("init")
    git("config", "user.name", "TrainLens Test")
    git("config", "user.email", "test@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("add", ".")
    git("commit", "-m", "fixture")
    commit = git("rev-parse", "HEAD")

    result = cli(
        tmp_path, "run", "--name", "git-state", "--", sys.executable, "train.py"
    )

    assert result.returncode == 0, result.stderr
    [record] = records(tmp_path)
    assert record.git.commit == commit
    assert record.git.dirty is False
    assert (tmp_path / "tracked.txt").read_text() == "changed"


@pytest.fixture
def selected_python(tmp_path):
    root = tmp_path / "selected venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    python = root / "bin" / "python"
    site = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        check=True,
        capture_output=True,
        text=True,
        env=clean_env(),
    ).stdout.strip()
    return python, Path(site)


def test_selected_virtual_environment_supplies_execution_and_metadata(
    tmp_path, selected_python
):
    python, site = selected_python
    source = Path(__file__).resolve().parents[1] / "src"
    (site / "trainlens-test.pth").write_text(f"{source}\n", encoding="utf-8")
    (tmp_path / "train.py").write_text(
        "import json, sys\n"
        "print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix}))\n",
        encoding="utf-8",
    )
    relative_python = str(python.relative_to(tmp_path))

    result = cli(
        tmp_path, "run", "--name", "selected", "--", relative_python, "train.py"
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["executable"] == str(python)
    assert observed["prefix"] == str(python.parent.parent)
    [record] = records(tmp_path)
    assert record.python_executable == str(python)
    assert record.command == [relative_python, "train.py"]
    assert record.environment.pytorch_version is None
    assert record.environment.cuda_available is None
    assert any(
        "PyTorch" in warning for warning in record.diagnostics.collection_warnings
    )
    assert record.status == "succeeded"
    assert record.metrics.unavailable_reason == "pytorch_unavailable"
    assert record.metrics.cuda_devices == []


def test_selected_interpreter_without_trainlens_records_bootstrap_failure(
    tmp_path, selected_python
):
    python, _ = selected_python
    (tmp_path / "train.py").write_text(
        "from pathlib import Path\nPath('executed').touch()\n", encoding="utf-8"
    )

    result = cli(
        tmp_path, "run", "--name", "missing-package", "--", str(python), "train.py"
    )

    assert result.returncode != 0
    assert "trainlens" in result.stderr.lower()
    assert not (tmp_path / "executed").exists()
    [record] = records(tmp_path)
    assert record.status == "failed"
    assert record.exit_code != 0
    assert record.python_executable == str(python)
    assert record.environment.python_version is None
    assert record.diagnostics.collection_warnings


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-m", "train"],
        [sys.executable, "-c", "from pathlib import Path; Path('executed').touch()"],
        [sys.executable, "-"],
        ["sh", "train.py"],
        ["torchrun", "train.py"],
    ],
)
def test_unsupported_commands_are_rejected_without_executing_user_code(
    tmp_path, command
):
    (tmp_path / "train.py").write_text(
        "from pathlib import Path\nPath('executed').touch()\n", encoding="utf-8"
    )

    result = cli(tmp_path, "run", "--name", "unsupported", "--", *command)

    assert result.returncode != 0
    assert result.stderr
    assert not (tmp_path / "executed").exists()


def test_separator_is_required_before_training_command(tmp_path):
    (tmp_path / "train.py").write_text(
        "from pathlib import Path\nPath('executed').touch()\n", encoding="utf-8"
    )

    result = cli(
        tmp_path, "run", "--name", "missing-separator", sys.executable, "train.py"
    )

    assert result.returncode != 0
    assert "--" in result.stderr
    assert not (tmp_path / "executed").exists()


def test_abrupt_child_exit_preserves_startup_but_has_no_final_metrics(tmp_path):
    (tmp_path / "train.py").write_text("import os\nos._exit(7)\n")
    result = cli(tmp_path, "run", "--name", "abrupt", "--", sys.executable, "train.py")
    assert result.returncode != 0
    [record] = records(tmp_path)
    assert record.status == "failed"
    assert record.exit_code == 7
    assert record.environment.python_version
    assert record.python_executable == sys.executable
    assert record.metrics.cuda_devices == []
    assert record.metrics.unavailable_reason == "finalization_missing"
    assert any(
        "Only startup" in warning for warning in record.diagnostics.collection_warnings
    )


@pytest.mark.parametrize(
    "ending,code",
    [
        ("pass", 0),
        ("raise SystemExit(0)", 0),
        ("raise SystemExit(7)", 7),
        ("raise RuntimeError('boom')", 1),
    ],
)
@pytest.mark.parametrize("query_failure", [False, True])
def test_mock_cuda_in_real_selected_child_reaches_saved_record_and_report(
    tmp_path, selected_python, ending, code, query_failure
):
    python, site = selected_python
    source = Path(__file__).resolve().parents[1] / "src"
    (site / "trainlens-test.pth").write_text(f"{source}\n", encoding="utf-8")
    # A fake installed only in the selected interpreter proves the parent cannot
    # supply the metrics. Queries also require state set by the training script.
    (site / "torch.py").write_text(
        """
import os
from types import SimpleNamespace
__version__ = "fake-test-torch"
version = SimpleNamespace(cuda="fake-test-cuda")
class FakeCUDA:
    initialized = False
    training_pid = None
    fail = False
    allocated = None
    def is_initialized(self):
        return self.initialized
    def guard(self):
        assert self.initialized and self.training_pid == os.getpid()
    def is_available(self):
        self.guard()
        return True
    def device_count(self):
        self.guard()
        return 1
    def get_device_name(self, index):
        self.guard()
        return "Fake GPU"
    def max_memory_allocated(self, index):
        self.guard()
        return self.allocated
    def max_memory_reserved(self, index):
        self.guard()
        if self.fail:
            raise RuntimeError("reserved query failed")
        return 4096
cuda = FakeCUDA()
""",
        encoding="utf-8",
    )
    (tmp_path / "train.py").write_text(
        "import os, torch\n"
        "assert not torch.cuda.initialized\n"
        "torch.cuda.initialized = True\n"
        "torch.cuda.training_pid = os.getpid()\n"
        "torch.cuda.allocated = 0\n"
        f"torch.cuda.fail = {query_failure!r}\n" + ending + "\n",
        encoding="utf-8",
    )
    result = cli(tmp_path, "run", "--name", "measured", "--", str(python), "train.py")
    assert (result.returncode == 0) == (code == 0), result.stderr
    [record] = records(tmp_path)
    assert record.schema_version == 1
    assert record.status == ("succeeded" if code == 0 else "failed")
    assert record.exit_code == code
    assert record.environment.cuda_available is True
    assert record.environment.gpus[0].name == "Fake GPU"
    assert record.environment.python_version
    assert record.metrics.measurement_scope == "bootstrap_to_script_exit_v1"
    [device] = record.metrics.cuda_devices
    assert device.device == "cuda:0"
    assert device.peak_allocated_bytes == 0
    assert device.peak_reserved_bytes == (None if query_failure else 4096)
    if query_failure:
        assert "reserved query failed" in device.unavailable_reason
        assert any(
            "reserved query failed" in item
            for item in record.diagnostics.collection_warnings
        )
    if code == 1:
        assert "Traceback (most recent call last)" in result.stderr
        assert "RuntimeError: boom" in result.stderr
        assert record.diagnostics.failure_message == "RuntimeError: boom"
    elif code == 7:
        assert record.diagnostics.failure_message == "SystemExit: 7"
    else:
        assert record.diagnostics.failure_message is None

    # Existing schema-1 records remain readable alongside newly measured records.
    store = RunStore(tmp_path)
    store.save(
        RunRecord(
            run_id="historical",
            name="historical",
            trainlens_version="0.1.0.dev0",
            command=["python", "old.py"],
            cwd=str(tmp_path),
        )
    )
    listed = cli(tmp_path, "list")
    assert listed.returncode == 0, listed.stderr
    assert "measured" in listed.stdout and "historical" in listed.stdout
    report = cli(tmp_path, "diff", "historical", "measured")
    assert report.returncode == 0, report.stderr
    assert "0 bytes" in report.stdout
    assert r"not\_collected" in report.stdout
    assert ("4096 bytes" in report.stdout) == (not query_failure)
