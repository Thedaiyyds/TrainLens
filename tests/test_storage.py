import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from trainlens.models import (
    CUDADeviceMetrics,
    EnvironmentInfo,
    GitInfo,
    GPUInfo,
    RunDiagnostics,
    RunMetrics,
    RunRecord,
)
from trainlens.storage import RunStore


@pytest.fixture
def record() -> RunRecord:
    return RunRecord(
        run_id="run-001",
        name="基线运行",
        trainlens_version="0.1.0.dev0",
        command=["python", "train.py", "--label", "训练 baseline"],
        cwd="/work/project",
        python_executable="/work/project/.venv/bin/python",
        started_at="2026-09-21T00:00:00Z",
        ended_at="2026-09-21T00:00:10Z",
        runtime_seconds=10.0,
        status="succeeded",
        exit_code=0,
        environment=EnvironmentInfo(
            python_version="3.10.0",
            os="Linux",
            architecture="x86_64",
            pytorch_version="2.5.0",
            cuda_build_version="12.4",
            cuda_available=True,
            gpus=[GPUInfo(device="cuda:0", name="Test GPU")],
        ),
        git=GitInfo(commit="a" * 40, dirty=False),
        metrics=RunMetrics(
            cuda_devices=[CUDADeviceMetrics("cuda:0", 1024, 2048)],
            measurement_scope="test_allocation_window",
        ),
        diagnostics=RunDiagnostics(collection_warnings=["测试 warning"]),
    )


def test_save_creates_directories_and_round_trips_full_record(
    tmp_path: Path, record: RunRecord
) -> None:
    store = RunStore(tmp_path)
    assert not (tmp_path / ".trainlens").exists()

    path = store.save(record)

    assert path == tmp_path / ".trainlens" / "runs" / "run-001.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == asdict(record)
    assert data["schema_version"] == 1
    assert data["name"] == "基线运行"
    assert store.load(record.run_id) == record


def test_failed_run_preserves_outcome_and_diagnostics(
    tmp_path: Path, record: RunRecord
) -> None:
    failed = replace(
        record,
        status="failed",
        exit_code=1,
        metrics=RunMetrics(cuda_devices=[], unavailable_reason="finalization_missing"),
        diagnostics=RunDiagnostics(
            failure_message="ValueError: 训练失败",
            collection_warnings=["CUDA peaks unavailable after failure."],
        ),
    )
    store = RunStore(tmp_path)

    store.save(failed)
    loaded = store.load(failed.run_id)

    assert loaded == failed
    assert loaded.status == "failed"
    assert loaded.diagnostics.failure_message == "ValueError: 训练失败"


def test_unknown_values_and_partial_peak_remain_distinct_from_zero(
    tmp_path: Path, record: RunRecord
) -> None:
    record.environment = EnvironmentInfo()
    record.git = GitInfo(unavailable_reason="not_a_git_repository")
    record.runtime_seconds = None
    record.metrics = RunMetrics(
        cuda_devices=[
            CUDADeviceMetrics("cuda:0", 0, None, "Reserved peak query failed.")
        ]
    )
    store = RunStore(tmp_path)

    store.save(record)
    loaded = store.load(record.run_id)

    assert loaded == record
    assert loaded.runtime_seconds is None
    assert loaded.environment.gpus is None
    assert loaded.git.dirty is None
    assert loaded.metrics.cuda_devices[0].peak_allocated_bytes == 0
    assert loaded.metrics.cuda_devices[0].peak_reserved_bytes is None


def test_loaded_nested_objects_are_typed_and_independent(
    tmp_path: Path, record: RunRecord
) -> None:
    store = RunStore(tmp_path)
    store.save(record)

    loaded = store.load(record.run_id)
    loaded.command.append("--changed")
    loaded.environment.gpus[0].name = "Changed GPU"
    loaded.metrics.cuda_devices[0].peak_allocated_bytes = 999
    loaded.git.dirty = True
    loaded.diagnostics.collection_warnings.append("New warning")

    assert store.load(record.run_id) == record
    assert loaded != record


@pytest.mark.parametrize("duplicate", ["id", "name"])
def test_duplicate_identity_does_not_overwrite_existing_record(
    tmp_path: Path, record: RunRecord, duplicate: str
) -> None:
    store = RunStore(tmp_path)
    path = store.save(record)
    original = path.read_bytes()
    second = replace(
        record,
        run_id=record.run_id if duplicate == "id" else "run-002",
        name=record.name if duplicate == "name" else "experiment",
        runtime_seconds=20.0,
    )

    with pytest.raises(FileExistsError):
        store.save(second)

    assert path.read_bytes() == original
    assert list(path.parent.glob("*.json")) == [path]


def test_default_store_keeps_original_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: RunRecord
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    store = RunStore()
    monkeypatch.chdir(tmp_path)

    path = store.save(record)

    assert path == project / ".trainlens" / "runs" / "run-001.json"
    assert store.load(record.run_id) == record
    assert not (tmp_path / ".trainlens").exists()


def test_missing_store_load_does_not_create_directories(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RunStore(tmp_path).load("missing")

    assert not (tmp_path / ".trainlens").exists()


def test_missing_run_in_existing_store(tmp_path: Path, record: RunRecord) -> None:
    store = RunStore(tmp_path)
    store.save(record)

    with pytest.raises(FileNotFoundError):
        store.load("missing")


def test_corrupt_json_is_reported_and_preserved(
    tmp_path: Path, record: RunRecord
) -> None:
    store = RunStore(tmp_path)
    path = store.save(record)
    path.write_text('{"schema_version":', encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(json.JSONDecodeError):
        store.load(record.run_id)

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 999},
        {"environment": {**asdict(EnvironmentInfo()), "gpus": "not a list"}},
        {"status": "unknown"},
        {"run_id": "different-id"},
    ],
)
def test_unrestorable_record_is_reported_and_preserved(
    tmp_path: Path, record: RunRecord, change: dict
) -> None:
    store = RunStore(tmp_path)
    path = store.save(record)
    data = asdict(record)
    data.update(change)
    path.write_text(json.dumps(data), encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ValueError):
        store.load(record.run_id)

    assert path.read_bytes() == original


@pytest.mark.parametrize("run_id", ["../outside", "/absolute", "nested/run"])
def test_run_id_cannot_escape_store(
    tmp_path: Path, record: RunRecord, run_id: str
) -> None:
    store = RunStore(tmp_path)

    with pytest.raises(ValueError):
        store.save(replace(record, run_id=run_id))
    with pytest.raises(ValueError):
        store.load(run_id)


def test_symlinked_storage_directory_does_not_write_outside_project(
    tmp_path: Path, record: RunRecord
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / ".trainlens").symlink_to(outside, target_is_directory=True)

    with pytest.raises((ValueError, OSError)):
        RunStore(project).save(record)

    assert list(outside.iterdir()) == []


def test_symlinked_record_does_not_read_outside_store(
    tmp_path: Path, record: RunRecord
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(asdict(record)), encoding="utf-8")
    runs = tmp_path / "project" / ".trainlens" / "runs"
    runs.mkdir(parents=True)
    (runs / "run-001.json").symlink_to(outside)

    with pytest.raises((ValueError, OSError)):
        RunStore(tmp_path / "project").load(record.run_id)


def test_storage_path_that_is_a_file_reports_filesystem_error(
    tmp_path: Path, record: RunRecord
) -> None:
    occupied = tmp_path / ".trainlens"
    occupied.write_text("existing content", encoding="utf-8")

    with pytest.raises(OSError):
        RunStore(tmp_path).save(record)

    assert occupied.read_text(encoding="utf-8") == "existing content"


def test_save_revalidates_mutated_record_before_creating_directories(
    tmp_path: Path, record: RunRecord
) -> None:
    record.status = "unknown"

    with pytest.raises(ValueError, match="Run status"):
        RunStore(tmp_path).save(record)

    assert not (tmp_path / ".trainlens").exists()


def test_directory_scan_failure_prevents_unchecked_save(
    tmp_path: Path, record: RunRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path)
    existing = store.save(record)
    original = existing.read_bytes()
    second = replace(record, run_id="run-002", name="experiment")
    error = PermissionError("Cannot scan existing run names")

    def fail_scan(path: Path):
        raise error

    monkeypatch.setattr(Path, "iterdir", fail_scan)

    with pytest.raises(PermissionError) as caught:
        store.save(second)

    assert caught.value is error
    assert not existing.with_name("run-002.json").exists()
    assert existing.read_bytes() == original


def test_partial_write_failure_removes_new_record_and_preserves_error(
    tmp_path: Path, record: RunRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path)
    existing = store.save(record)
    original = existing.read_bytes()
    second = replace(record, run_id="run-002", name="experiment")
    target = existing.with_name("run-002.json")
    error = OSError(28, "Simulated disk full", str(target))
    original_open = Path.open

    def open_with_partial_failure(path: Path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == target:
            original_write = stream.write

            def write_partially(text: str) -> None:
                original_write(text[:10])
                raise error

            monkeypatch.setattr(stream, "write", write_partially)
        return stream

    monkeypatch.setattr(Path, "open", open_with_partial_failure)

    with pytest.raises(OSError) as caught:
        store.save(second)

    assert caught.value is error
    assert not target.exists()
    assert existing.read_bytes() == original


@pytest.mark.parametrize("existing_directory", [None, ".trainlens", ".trainlens/runs"])
def test_list_empty_store_does_not_create_directories(tmp_path, existing_directory):
    if existing_directory is not None:
        (tmp_path / existing_directory).mkdir(parents=True)
    before = sorted(tmp_path.rglob("*"))

    assert RunStore(tmp_path).list_records() == []
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("fraction", ["1", "1234", "123456"])
def test_list_restores_records_by_start_time_without_modifying_files(
    tmp_path, record, fraction
):
    store = RunStore(tmp_path)
    starts = [
        ("z-unknown", None),
        ("b-tie", "2026-09-21T00:00:00Z"),
        ("newest", f"2026-09-21T00:00:00.{fraction}+00:00"),
        ("a-tie", "2026-09-21T00:00:00.000000+00:00"),
        ("oldest", "2026-09-20T23:59:59+00:00"),
        ("a-unknown", None),
    ]
    originals = {}
    for index, (run_id, started_at) in enumerate(starts):
        saved = replace(record, run_id=run_id, name=run_id, started_at=started_at)
        originals[run_id] = saved
        path = store.save(saved)
        os.utime(path, (100 + index, 100 + index))
    (store.runs_dir / "notes.txt").write_text("not JSON")
    (store.runs_dir / "nested").mkdir()
    (store.runs_dir / "nested" / "ignored.json").write_text("not JSON")
    files = [path for path in store.runs_dir.rglob("*") if path.is_file()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}

    for _ in range(2):
        loaded = store.list_records()
        assert [item.run_id for item in loaded] == [
            "newest",
            "a-tie",
            "b-tie",
            "oldest",
            "a-unknown",
            "z-unknown",
        ]
        assert all(item == originals[item.run_id] for item in loaded)
        assert all(isinstance(item.environment, EnvironmentInfo) for item in loaded)
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files
    } == before


def test_list_rejects_storage_and_record_paths_outside_store(tmp_path, record):
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    link = project / ".trainlens"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        RunStore(project).list_records()
    link.unlink()

    store = RunStore(project)
    path = store.save(record)
    original = path.read_bytes()
    external = outside / "record.json"
    external.write_bytes(original)
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError, match="outside"):
        store.list_records()
    assert external.read_bytes() == original


def test_list_propagates_directory_scan_error(tmp_path, monkeypatch):
    def fail_scan(path):
        raise PermissionError("Cannot read saved runs")

    monkeypatch.setattr(Path, "iterdir", fail_scan)
    with pytest.raises(PermissionError, match="Cannot read saved runs"):
        RunStore(tmp_path).list_records()


def test_resolve_exact_name_or_id_and_same_record_matching_both(tmp_path, record):
    store = RunStore(tmp_path)
    first = replace(record, run_id="baseline-id", name="baseline")
    second = replace(record, run_id="experiment", name="experiment")
    paths = [store.save(item) for item in (first, second)]
    before = {path: path.read_bytes() for path in paths}

    assert store.resolve("baseline") == first
    assert store.resolve("baseline-id") == first
    assert store.resolve("experiment") == second
    for missing in ("base", "BASELINE", " baseline", "nonexistent"):
        with pytest.raises(FileNotFoundError, match="No saved run"):
            store.resolve(missing)
    assert {path: path.read_bytes() for path in paths} == before


def test_resolve_name_id_collision_is_ambiguous(tmp_path, record):
    store = RunStore(tmp_path)
    store.save(replace(record, run_id="first", name="collision"))
    store.save(replace(record, run_id="collision", name="second"))

    with pytest.raises(ValueError, match="Ambiguous.*collision"):
        store.resolve("collision")


def test_resolve_empty_store_does_not_create_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match="No saved run"):
        RunStore(tmp_path).resolve("missing")
    assert not (tmp_path / ".trainlens").exists()


def test_resolve_does_not_skip_corrupt_records(tmp_path, record):
    store = RunStore(tmp_path)
    store.save(record)
    bad = store.runs_dir / "broken.json"
    bad.write_text("{")

    with pytest.raises(json.JSONDecodeError):
        store.resolve(record.run_id)
    assert bad.read_text() == "{"
