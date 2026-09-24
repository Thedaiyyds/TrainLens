import json
import os
import subprocess
import sys
from dataclasses import asdict, replace

import pytest
from typer.testing import CliRunner

from trainlens.cli import app
from trainlens.models import CUDADeviceMetrics, RunDiagnostics, RunMetrics, RunRecord
from trainlens.storage import RunStore


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = RunStore(tmp_path)
    first = RunRecord(
        run_id="base-id",
        name="baseline",
        trainlens_version="0.1.0.dev0",
        command=["/unavailable/python", "missing.py", "hello world"],
        cwd="/historical/project",
        runtime_seconds=10,
        status="succeeded",
        exit_code=0,
    )
    second = replace(
        first,
        run_id="exp-id",
        name="experiment",
        runtime_seconds=12.5,
        status="failed",
        exit_code=7,
        diagnostics=RunDiagnostics(
            failure_message="historical failure",
            collection_warnings=["historical warning"],
        ),
    )
    paths = [store.save(record) for record in (first, second)]
    return store, first, second, paths


@pytest.mark.parametrize(
    "selectors",
    [
        ("baseline", "experiment"),
        ("base-id", "exp-id"),
        ("baseline", "exp-id"),
    ],
)
def test_diff_resolves_selectors_and_emits_clean_markdown(saved, selectors):
    _, _, _, paths = saved
    before = {path: path.read_bytes() for path in paths}
    result = CliRunner().invoke(app, ["diff", *selectors])

    assert result.exit_code == 0, result.stderr
    assert result.stdout.startswith("# TrainLens Comparison\n")
    assert result.stderr == ""
    for text in (
        "base-id",
        "exp-id",
        "hello world",
        "/historical/project",
        "Lifecycle",
        "Environment",
        "Git",
        "Runtime",
        "CUDA",
        "Diagnostics",
        "+2.5 s",
        "+25%",
        "historical warning",
        "historical failure",
    ):
        assert text in result.stdout
    assert "TrainLens:" not in result.stdout
    assert {path: path.read_bytes() for path in paths} == before


def test_missing_selector_has_no_partial_report_or_writes(saved):
    _, _, _, paths = saved
    before = {path: path.read_bytes() for path in paths}
    result = CliRunner().invoke(app, ["diff", "baseline", "does-not-exist"])
    assert result.exit_code != 0
    assert result.stdout == ""
    assert "No saved run" in result.stderr and "does-not-exist" in result.stderr
    assert {path: path.read_bytes() for path in paths} == before


def test_ambiguous_selector_is_not_resolved_by_priority(saved):
    store, first, _, paths = saved
    paths.append(store.save(replace(first, run_id="baseline", name="other")))
    before = {path: path.read_bytes() for path in paths}
    result = CliRunner().invoke(app, ["diff", "baseline", "experiment"])
    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Ambiguous" in result.stderr and "baseline" in result.stderr
    assert {path: path.read_bytes() for path in paths} == before


def test_empty_store_fails_without_creating_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["diff", "x", "y"])
    assert result.exit_code != 0
    assert result.stdout == ""
    assert "No saved run" in result.stderr
    assert not (tmp_path / ".trainlens").exists()


@pytest.mark.parametrize("corruption", ["json", "schema", "data"])
def test_corrupt_storage_is_not_skipped_or_rewritten(saved, corruption):
    _, first, _, paths = saved
    path = paths[0]
    data = asdict(first)
    data.update(
        {"schema_version": 999} if corruption == "schema" else {"command": "invalid"}
    )
    path.write_text("{" if corruption == "json" else json.dumps(data))
    before = {path: path.read_bytes() for path in paths}
    result = CliRunner().invoke(app, ["diff", "baseline", "experiment"])
    assert result.exit_code != 0
    assert result.stdout == ""
    assert "TrainLens: could not compare runs" in result.stderr
    assert path.name in result.stderr
    assert {path: path.read_bytes() for path in paths} == before


def test_report_failure_emits_no_partial_markdown(saved):
    store, first, _, _ = saved
    duplicate_devices = RunMetrics(
        [CUDADeviceMetrics("cuda:0", 1, 2), CUDADeviceMetrics("cuda:0", 3, 4)],
        "whole_run",
    )
    path = store.save(
        replace(
            first, run_id="duplicate-id", name="duplicate", metrics=duplicate_devices
        )
    )
    before = path.read_bytes()
    result = CliRunner().invoke(app, ["diff", "duplicate", "experiment"])
    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Duplicate CUDA device" in result.stderr
    assert path.read_bytes() == before


def test_diff_redirection_in_fresh_interpreter_needs_no_training_environment(
    saved, tmp_path
):
    _, _, _, paths = saved
    before = {path: path.read_bytes() for path in paths}
    source = """
import sys
blocked = ('torch', 'cuda', 'pynvml', 'trainlens.supervisor',
           'trainlens.bootstrap', 'trainlens._bootstrap_child', 'trainlens.collectors')
class RejectTrainingImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, RejectTrainingImports())
def no_processes(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec', 'os.fork'):
        raise AssertionError('Unexpected execution: ' + event)
sys.addaudithook(no_processes)
from trainlens.cli import app
app(args=['diff', 'baseline', 'experiment'], standalone_mode=False)
assert not any(name in sys.modules for name in blocked)
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME")
    }
    destination = tmp_path / "comparison.md"
    with destination.open("w") as output:
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=tmp_path,
            env=env,
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = destination.read_text()
    assert report.startswith("# TrainLens Comparison\n")
    assert "historical warning" in report
    assert {path: path.read_bytes() for path in paths} == before
