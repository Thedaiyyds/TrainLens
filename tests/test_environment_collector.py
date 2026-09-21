import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trainlens.collectors import environment
from trainlens.models import GPUInfo, RunDiagnostics


def fake_torch(monkeypatch, *, build="12.4", initialized=False):
    cuda = Mock(
        spec=["is_initialized", "is_available", "device_count", "get_device_name"]
    )
    cuda.is_initialized.return_value = initialized
    cuda.is_available.return_value = True
    cuda.device_count.return_value = 2
    cuda.get_device_name.side_effect = ["GPU A", "GPU B"]
    torch = SimpleNamespace(
        __version__="2.5.0", version=SimpleNamespace(cuda=build), cuda=cuda
    )
    monkeypatch.setattr(environment.importlib, "import_module", lambda name: torch)
    return cuda


@pytest.mark.parametrize(
    "error", [ModuleNotFoundError("torch"), OSError("bad library")]
)
def test_python_platform_survive_torch_import_failure(monkeypatch, error):
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.10.0")
    monkeypatch.setattr(environment.platform, "platform", lambda: "macOS-test")
    monkeypatch.setattr(environment.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(sys, "executable", "/test/python")

    def fail_import(name):
        assert name == "torch"
        raise error

    monkeypatch.setattr(environment.importlib, "import_module", fail_import)
    diagnostics = RunDiagnostics(failure_message="Existing training failure")

    info = environment.collect_environment(diagnostics=diagnostics)

    assert info.python_version == "3.10.0"
    assert info.os == "macOS-test"
    assert info.architecture == "arm64"
    assert environment.collect_python_executable() == "/test/python"
    assert info.pytorch_version is None
    assert info.cuda_build_version is None
    assert info.cuda_available is None
    assert info.gpus is None
    assert "PyTorch import unavailable" in diagnostics.collection_warnings[0]
    assert diagnostics.failure_message == "Existing training failure"


def test_cpu_build_needs_no_cuda_queries(monkeypatch):
    cuda = fake_torch(monkeypatch, build=None)
    diagnostics = RunDiagnostics()

    info = environment.collect_environment(diagnostics=diagnostics)

    assert info.pytorch_version == "2.5.0"
    assert info.cuda_build_version is None
    assert info.cuda_available is False
    assert info.gpus == []
    assert not cuda.mock_calls
    assert not diagnostics.collection_warnings


def test_cuda_build_defers_queries_until_initialized(monkeypatch):
    cuda = fake_torch(monkeypatch)
    diagnostics = RunDiagnostics()

    info = environment.collect_environment(diagnostics=diagnostics)

    assert info.cuda_build_version == "12.4"
    assert info.cuda_available is None
    assert info.gpus is None
    assert [call[0] for call in cuda.mock_calls] == ["is_initialized"]
    assert "deferred" in diagnostics.collection_warnings[0]


def test_initialized_cuda_inventory_is_metadata_only(monkeypatch):
    cuda = fake_torch(monkeypatch, initialized=True)
    diagnostics = RunDiagnostics()

    info = environment.collect_environment(diagnostics=diagnostics)

    assert info.cuda_available is True
    assert info.gpus == [GPUInfo("cuda:0", "GPU A"), GPUInfo("cuda:1", "GPU B")]
    assert [call[0] for call in cuda.mock_calls] == [
        "is_initialized",
        "is_available",
        "device_count",
        "get_device_name",
        "get_device_name",
    ]
    assert not diagnostics.collection_warnings


@pytest.mark.parametrize("query", ["is_initialized", "is_available", "device_count"])
def test_cuda_query_failure_retains_build_and_unknown_inventory(monkeypatch, query):
    cuda = fake_torch(monkeypatch, initialized=True)
    getattr(cuda, query).side_effect = RuntimeError("query failed")
    diagnostics = RunDiagnostics()

    info = environment.collect_environment(diagnostics=diagnostics)

    assert info.pytorch_version == "2.5.0"
    assert info.cuda_build_version == "12.4"
    assert info.gpus is None
    assert info.cuda_available is (True if query == "device_count" else None)
    assert "query failed" in diagnostics.collection_warnings[0]


def test_device_name_failure_keeps_device_identity_and_other_names(monkeypatch):
    cuda = fake_torch(monkeypatch, initialized=True)
    cuda.get_device_name.side_effect = [RuntimeError("name unavailable"), "GPU B"]
    diagnostics = RunDiagnostics()

    info = environment.collect_environment(diagnostics=diagnostics)

    assert info.gpus == [GPUInfo("cuda:0", None), GPUInfo("cuda:1", "GPU B")]
    assert "cuda:0 name" in diagnostics.collection_warnings[0]


def test_platform_failure_does_not_discard_other_metadata(monkeypatch):
    fake_torch(monkeypatch, build=None)
    monkeypatch.setattr(
        environment.platform, "platform", Mock(side_effect=OSError("OS"))
    )
    monkeypatch.setattr(sys, "executable", "")
    diagnostics = RunDiagnostics()

    info = environment.collect_environment(diagnostics=diagnostics)

    assert info.os is None
    assert info.python_version
    assert info.pytorch_version == "2.5.0"
    assert environment.collect_python_executable() is None
    assert "os:" in diagnostics.collection_warnings[0]
