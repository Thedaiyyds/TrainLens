"""Build and exercise an installed wheel; opt-in because pip may need a network."""

import configparser
import json
import os
import platform
import subprocess
import sys
import tempfile
import venv
import zipfile
from email.parser import Parser
from importlib.metadata import version
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]

PROBE = """
import importlib.util, json, sys, sysconfig
from importlib.metadata import version
import trainlens
print(json.dumps({
    'package_path': trainlens.__file__,
    'prefix': sys.prefix,
    'purelib': sysconfig.get_path('purelib'),
    'platlib': sysconfig.get_path('platlib'),
    'version': version('trainlens'),
    'pytorch_present': importlib.util.find_spec('torch') is not None,
}))
"""

TRAINING = """
import json, sys
import trainlens
print('hello from packaged trainlens')
print(json.dumps({'package_path': trainlens.__file__, 'prefix': sys.prefix}))
"""

READ_RECORDS = """
import json
from dataclasses import asdict
from trainlens.storage import RunStore
print(json.dumps([asdict(record) for record in RunStore().list_records()]))
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def clean_environment() -> dict[str, str]:
    # Neither the console script nor its training child may inherit repo imports.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    environment.update(PYTHONNOUSERSITE="1", NO_COLOR="1")
    return environment


def command(argv: list[str], cwd: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        argv, cwd=cwd, env=environment, capture_output=True, text=True, timeout=300
    )
    if result.returncode:
        raise ValueError(
            f"Command failed ({result.returncode}): {argv}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def inspect_wheel(path: Path, expected_version: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_files = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        require(len(metadata_files) == 1, "Expected one wheel metadata file")
        metadata_file = metadata_files[0]
        info = metadata_file.rsplit("/", 1)[0]
        require(
            all(name.startswith(("trainlens/", f"{info}/")) for name in names),
            "Unexpected files outside package and distribution metadata",
        )
        for module in ("__init__.py", "cli.py", "_bootstrap_child.py"):
            require(f"trainlens/{module}" in names, f"Missing package module: {module}")
        metadata = Parser().parsestr(archive.read(metadata_file).decode("utf-8"))
        require(metadata["Name"] == "trainlens", "Wrong distribution name")
        require(metadata["Version"] == expected_version, "Wrong wheel version")
        require(
            path.name.startswith(f"trainlens-{expected_version}-"),
            "Wheel filename/version mismatch",
        )
        require(
            metadata["Requires-Python"] == ">=3.10", "Unexpected Python requirement"
        )
        require(
            metadata["Description-Content-Type"] == "text/markdown"
            and metadata.get_payload().strip()
            == (REPOSITORY / "README.md").read_text(encoding="utf-8").strip(),
            "README missing or changed in package metadata",
        )
        require(
            metadata["License-Expression"] == "Apache-2.0", "Wrong license metadata"
        )
        require(
            "LICENSE" in metadata.get_all("License-File", []),
            "Missing license metadata",
        )
        require(
            archive.read(f"{info}/licenses/LICENSE")
            == (REPOSITORY / "LICENSE").read_bytes(),
            "LICENSE contents mismatch",
        )
        entries = configparser.ConfigParser()
        entries.read_string(archive.read(f"{info}/entry_points.txt").decode("utf-8"))
        require(
            entries.get("console_scripts", "trainlens", fallback=None)
            == "trainlens.cli:app",
            "Missing or incorrect console entry point",
        )
        return {
            "artifact": path.name,
            "version": metadata["Version"],
            "requires_python": metadata["Requires-Python"],
            "requires_dist": metadata.get_all("Requires-Dist", []),
            "files": sorted(names),
        }


def check_installation(probe: dict, root: Path, expected_version: str) -> None:
    package = Path(probe["package_path"]).resolve()
    root = root.resolve()
    libraries = [Path(probe[key]).resolve() for key in ("purelib", "platlib")]
    require(Path(probe["prefix"]).resolve() == root, "Wrong Python environment")
    require(
        any(
            library.is_relative_to(root) and package.is_relative_to(library)
            for library in libraries
        ),
        "Import did not come from temporary venv site-packages",
    )
    require(
        not package.is_relative_to(REPOSITORY / "src"), "Source-tree import leakage"
    )
    require(probe["version"] == expected_version, "Installed version mismatch")
    require(
        probe["pytorch_present"] is False, "Smoke environment unexpectedly has PyTorch"
    )


def check_records(
    records: list[dict], project: Path, python: Path, expected_version: str
) -> None:
    require(len(records) == 2, "Expected two persisted runs")
    require(
        {record["name"] for record in records} == {"baseline", "experiment"},
        "Wrong run names",
    )
    require(len({record["run_id"] for record in records}) == 2, "Run IDs reused")
    for record in records:
        require(
            record["status"] == "succeeded" and record["exit_code"] == 0,
            "Training failed",
        )
        require(record["cwd"] == str(project), "Wrong invocation directory")
        require(record["python_executable"] == str(python), "Wrong selected Python")
        require(
            record["trainlens_version"] == expected_version,
            "Wrong saved package version",
        )
        require(
            record["environment"]["pytorch_version"] is None,
            "Invented PyTorch metadata",
        )
        metrics = record["metrics"]
        require(metrics["cuda_devices"] == [], "Unexpected CUDA metrics")
        require(
            metrics["unavailable_reason"] == "pytorch_unavailable",
            "Missing unavailable reason",
        )
        require(metrics["measurement_scope"] is None, "Invented CUDA measurement scope")


def run_smoke() -> dict:
    expected_version = version("trainlens")  # Install this checkout before running.
    environment = clean_environment()
    with tempfile.TemporaryDirectory(prefix="trainlens-packaging-") as temporary:
        root = Path(temporary).resolve()
        wheels = root / "wheels"
        wheels.mkdir()
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(REPOSITORY),
                "--no-deps",
                "--wheel-dir",
                str(wheels),
            ],
            root,
            environment,
        )
        artifacts = list(wheels.glob("*.whl"))
        require(len(artifacts) == 1, "Expected exactly one built wheel")
        artifact = inspect_wheel(artifacts[0], expected_version)
        installed = root / "venv"
        venv.EnvBuilder(with_pip=True).create(installed)
        executable_dir = installed / ("Scripts" if os.name == "nt" else "bin")
        python = executable_dir / ("python.exe" if os.name == "nt" else "python")
        cli = executable_dir / ("trainlens.exe" if os.name == "nt" else "trainlens")
        command(
            [str(python), "-m", "pip", "install", str(artifacts[0])], root, environment
        )
        project = root / "project"
        project.mkdir()
        probe = json.loads(
            command([str(python), "-I", "-c", PROBE], project, environment)
        )
        check_installation(probe, installed, expected_version)
        require(cli.is_file(), "Console script not installed in temporary venv")
        help_output = command([str(cli), "--help"], project, environment)
        require("Diff your PyTorch training runs." in help_output, "CLI help missing")
        (project / "train.py").write_text(TRAINING, encoding="utf-8")
        for name in ("baseline", "experiment"):
            output = command(
                [str(cli), "run", "--name", name, "--", str(python), "train.py"],
                project,
                environment,
            )
            require(
                "hello from packaged trainlens" in output, "Training output missing"
            )
            child = json.loads(output.splitlines()[-1])
            require(
                child["package_path"] == probe["package_path"],
                "Training child imported different code",
            )
            require(
                child["prefix"] == probe["prefix"],
                "Training child used different environment",
            )
        records = json.loads(
            command([str(python), "-I", "-c", READ_RECORDS], project, environment)
        )
        check_records(records, project, python, expected_version)
        require((project / ".trainlens").is_dir(), "Local storage not created")
        listing = command([str(cli), "list"], project, environment)
        for record in records:
            require(
                record["name"] in listing and record["run_id"] in listing,
                "List missing saved run",
            )
        report = command(
            [str(cli), "diff", "baseline", "experiment"], project, environment
        )
        require(
            report.startswith("# TrainLens Comparison\n"), "Markdown report missing"
        )
        require(
            "baseline" in report and "experiment" in report,
            "Comparison missing identities",
        )
        require(
            "N/A" in report and "pytorch_unavailable" in report.replace("\\_", "_"),
            "Missing CUDA not explained",
        )
        return {
            "status": "PASS",
            "os": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "artifact": artifact,
            "installation": probe,
            "workflow": "console help; two successful saved runs; list; Markdown diff",
            "pytorch": "absent",
            "cuda": "unavailable; not real NVIDIA validation",
        }


def main() -> int:
    try:
        evidence = run_smoke()
    except (
        OSError,
        ValueError,
        KeyError,
        zipfile.BadZipFile,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"Packaging smoke FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
