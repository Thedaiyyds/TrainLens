"""Saved-run listing must remain a read-only operation on historical data."""

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, replace

import pytest
from typer.testing import CliRunner

from trainlens.cli import app
from trainlens.models import RunRecord
from trainlens.storage import RunStore


@pytest.fixture
def record():
    return RunRecord(
        run_id="saved-run",
        name="historical run",
        trainlens_version="0.1.0.dev0",
        command=["/missing/python", "missing-training.py"],
        cwd="/historical/project",
        started_at="2026-09-21T00:00:00Z",
        status="succeeded",
        runtime_seconds=2.5,
        exit_code=0,
    )


def test_empty_list_succeeds_with_header_only_and_no_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["list"])

    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.stdout.split() == [
        "ID",
        "NAME",
        "STARTED",
        "STATUS",
        "RUNTIME",
        "EXIT",
        "CODE",
    ]
    assert len(result.stdout.splitlines()) == 1
    assert not (tmp_path / ".trainlens").exists()


def test_list_displays_saved_fields_order_and_missing_vs_zero(
    tmp_path, monkeypatch, record
):
    monkeypatch.chdir(tmp_path)
    store = RunStore(tmp_path)
    oldest = record
    newest = replace(
        record,
        run_id="newest",
        name="failed experiment",
        status="failed",
        started_at="2026-09-22T00:00:00+00:00",
        runtime_seconds=0,
        exit_code=7,
    )
    unknown = replace(
        record,
        run_id="unknown",
        name="unfinished",
        status="running",
        started_at=None,
        runtime_seconds=None,
        exit_code=None,
    )
    paths = [store.save(item) for item in (unknown, oldest, newest)]
    before = {path: path.read_bytes() for path in paths}

    result = CliRunner().invoke(app, ["list"])

    assert result.exit_code == 0
    assert result.stderr == ""
    rows = result.stdout.splitlines()[1:]
    assert [row.split()[0] for row in rows] == ["newest", "saved-run", "unknown"]
    for row, saved in zip(rows, (newest, oldest, unknown)):
        assert saved.name in row
        assert saved.status in row
        if saved.started_at is not None:
            assert saved.started_at in row
    assert re.search(r"\b0(?:\.0+)? s\b", rows[0])
    assert rows[0].rstrip().endswith("7")
    assert "2.5 s" in rows[1]
    assert rows[1].rstrip().endswith("0")
    assert "N/A" not in rows[0] + rows[1]
    assert rows[2].count("N/A") == 3
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize("corruption", ["json", "schema", "status", "identity", "time"])
def test_list_fails_clearly_without_changing_bad_records(
    tmp_path, monkeypatch, record, corruption
):
    monkeypatch.chdir(tmp_path)
    path = RunStore(tmp_path).save(record)
    data = asdict(record)
    changes = {
        "schema": {"schema_version": 999},
        "status": {"status": "not-a-status"},
        "identity": {"run_id": "wrong-id"},
        "time": {"started_at": "not-a-timestamp"},
    }
    data.update(changes.get(corruption, {}))
    path.write_text("{" if corruption == "json" else json.dumps(data))
    original = path.read_bytes()

    result = CliRunner().invoke(app, ["list"])

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "TrainLens: could not list runs" in result.stderr
    assert record.run_id in result.stderr
    assert path.read_bytes() == original


def test_list_in_fresh_interpreter_without_execution_or_collection_imports(
    tmp_path, record
):
    path = RunStore(tmp_path).save(record)
    before = path.read_bytes()
    source = """
import sys

blocked = (
    'trainlens.collectors', 'trainlens.supervisor', 'trainlens.bootstrap',
    'trainlens._bootstrap_child', 'torch', 'cuda', 'pynvml', 'nvidia_smi',
)
class RejectTrainingImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Unexpected training import: ' + fullname)
sys.meta_path.insert(0, RejectTrainingImports())

def no_processes(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec', 'os.fork'):
        raise AssertionError('Unexpected process execution: ' + event)
sys.addaudithook(no_processes)

from trainlens.cli import app
app(args=['list'], standalone_mode=False)
assert not any(name in sys.modules for name in blocked)
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME")
    }
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert record.name in result.stdout
    assert result.stderr == ""
    assert path.read_bytes() == before
