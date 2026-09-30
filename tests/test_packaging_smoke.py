"""Offline unit checks; the real wheel-install smoke is a separate CI step."""

import importlib.util
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest

from trainlens.models import EnvironmentInfo, RunMetrics, RunRecord

PATH = Path(__file__).resolve().parents[1] / "validation/packaging/run_smoke.py"
spec = importlib.util.spec_from_file_location("packaging_smoke", PATH)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.fixture
def wheel_files():
    info = "trainlens-0.1.0.dev0.dist-info"
    metadata = (
        "Metadata-Version: 2.4\nName: trainlens\nVersion: 0.1.0.dev0\n"
        "Requires-Python: >=3.10\nDescription-Content-Type: text/markdown\n"
        "License-Expression: Apache-2.0\nLicense-File: LICENSE\n\n"
        + (smoke.REPOSITORY / "README.md").read_text(encoding="utf-8")
    )
    return {
        "trainlens/__init__.py": b"",
        "trainlens/cli.py": b"",
        "trainlens/_bootstrap_child.py": b"",
        f"{info}/METADATA": metadata.encode(),
        f"{info}/licenses/LICENSE": (smoke.REPOSITORY / "LICENSE").read_bytes(),
        f"{info}/entry_points.txt": b"[console_scripts]\ntrainlens = trainlens.cli:app\n",
    }


@pytest.mark.parametrize(
    "broken", [None, "version", "entry_point", "license", "readme", "module", "junk"]
)
def test_wheel_inspection_rejects_incomplete_artifacts(tmp_path, wheel_files, broken):
    info = "trainlens-0.1.0.dev0.dist-info"
    if broken == "version":
        wheel_files[f"{info}/METADATA"] = wheel_files[f"{info}/METADATA"].replace(
            b"Version: 0.1.0.dev0", b"Version: 9.9"
        )
    elif broken == "entry_point":
        wheel_files[f"{info}/entry_points.txt"] = (
            b"[console_scripts]\ntrainlens = other:app\n"
        )
    elif broken == "license":
        wheel_files[f"{info}/licenses/LICENSE"] = b"wrong license"
    elif broken == "readme":
        wheel_files[f"{info}/METADATA"] = (
            wheel_files[f"{info}/METADATA"].split(b"\n\n")[0] + b"\n\nwrong README"
        )
    elif broken == "module":
        del wheel_files["trainlens/_bootstrap_child.py"]
    elif broken == "junk":
        wheel_files[".trainlens/runs/private.json"] = b"{}"
    wheel = tmp_path / "trainlens-0.1.0.dev0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in wheel_files.items():
            archive.writestr(name, data)
    if broken:
        with pytest.raises(ValueError):
            smoke.inspect_wheel(wheel, "0.1.0.dev0")
    else:
        evidence = smoke.inspect_wheel(wheel, "0.1.0.dev0")
        assert evidence["artifact"] == wheel.name
        assert evidence["version"] == "0.1.0.dev0"


def test_smoke_removes_inherited_source_import_paths(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(smoke.REPOSITORY / "src"))
    monkeypatch.setenv("PYTHONHOME", "/outside-python")
    monkeypatch.setenv("VIRTUAL_ENV", "/outside-venv")
    environment = smoke.clean_environment()
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & environment.keys()
    assert environment["PYTHONNOUSERSITE"] == "1"


@pytest.mark.parametrize("location", ["installed", "checkout", "outside"])
def test_installed_import_must_come_from_fresh_venv(tmp_path, location):
    root = tmp_path / "venv"
    library = root / "lib" / "site-packages"
    path = {
        "installed": library / "trainlens/__init__.py",
        "checkout": smoke.REPOSITORY / "src/trainlens/__init__.py",
        "outside": tmp_path / "other-venv/trainlens/__init__.py",
    }[location]
    probe = {
        "prefix": str(root),
        "purelib": str(library),
        "platlib": str(library),
        "package_path": str(path),
        "version": "0.1.0.dev0",
        "pytorch_present": False,
    }
    if location == "installed":
        smoke.check_installation(probe, root, "0.1.0.dev0")
    else:
        with pytest.raises(ValueError, match="site-packages"):
            smoke.check_installation(probe, root, "0.1.0.dev0")


@pytest.mark.parametrize("broken", [None, "failure", "zero", "reason", "identity"])
def test_smoke_requires_saved_success_and_honest_missing_metrics(tmp_path, broken):
    python = tmp_path / "venv/bin/python"
    records = [
        asdict(
            RunRecord(
                run_id=name,
                name=name,
                trainlens_version="0.1.0.dev0",
                command=[str(python), "train.py"],
                python_executable=str(python),
                cwd=str(tmp_path),
                status="succeeded",
                exit_code=0,
                environment=EnvironmentInfo(),
                metrics=RunMetrics([], unavailable_reason="pytorch_unavailable"),
            )
        )
        for name in ("baseline", "experiment")
    ]
    if broken == "failure":
        records[0]["status"] = "failed"
    elif broken == "zero":
        records[0]["metrics"]["cuda_devices"] = [{"peak_allocated_bytes": 0}]
    elif broken == "reason":
        records[0]["metrics"]["unavailable_reason"] = None
    elif broken == "identity":
        records[1]["run_id"] = records[0]["run_id"]
    if broken:
        with pytest.raises(ValueError):
            smoke.check_records(records, tmp_path, python, "0.1.0.dev0")
    else:
        smoke.check_records(
            json.loads(json.dumps(records)), tmp_path, python, "0.1.0.dev0"
        )
