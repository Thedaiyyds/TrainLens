import re
from copy import deepcopy
from dataclasses import replace

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
from trainlens.report import render_comparison


@pytest.fixture
def pair():
    baseline = RunRecord(
        run_id="base-id",
        name="baseline",
        trainlens_version="0.1.0.dev0",
        command=["python", "train.py", "--label", "hello world", ""],
        cwd="/saved/baseline",
        python_executable="/saved/bin/python",
        started_at="2026-09-21T10:00:00Z",
        ended_at="2026-09-21T10:00:10Z",
        runtime_seconds=10,
        status="succeeded",
        exit_code=0,
        environment=EnvironmentInfo(
            python_version="3.10",
            os="Linux",
            architecture="x86_64",
            pytorch_version="2.5",
            cuda_build_version="12.4",
            cuda_available=True,
            gpus=[GPUInfo("cuda:0", "Saved GPU")],
        ),
        git=GitInfo(commit="a" * 40, dirty=False),
    )
    experiment = replace(
        baseline,
        run_id="exp-id",
        name="experiment",
        cwd="/saved/experiment",
        runtime_seconds=12.5,
        status="failed",
        exit_code=7,
        environment=EnvironmentInfo(python_version="3.13", pytorch_version="2.6"),
        git=GitInfo(commit="b" * 40, dirty=True),
        diagnostics=RunDiagnostics(
            failure_message="training failed", collection_warnings=["saved warning"]
        ),
    )
    return baseline, experiment


def row(report, first, second=None):
    for line in report.splitlines():
        if not line.startswith("| "):
            continue
        cells = [part.strip() for part in re.split(r"(?<!\\)\|", line)[1:-1]]
        if cells[0] == first and (second is None or cells[1] == second):
            return cells
    raise AssertionError(f"Missing report row: {first}, {second}")


def test_report_contains_recorded_facts_and_does_not_mutate(pair):
    original = deepcopy(pair)
    report = render_comparison(*pair)
    assert report.startswith("# TrainLens Comparison\n")
    for text in (
        "base-id",
        "exp-id",
        "baseline",
        "experiment",
        "0.1.0.dev0",
        "Schema version",
        '"hello world", ""',
        "/saved/baseline",
        "/saved/experiment",
        "/saved/bin/python",
        "2026-09-21T10:00:00Z",
        "2026-09-21T10:00:10Z",
        "succeeded",
        "failed",
        "3.10",
        "3.13",
        "2.5",
        "2.6",
        "12.4",
        "Saved GPU",
        "a" * 40,
        "b" * 40,
        "training failed",
        "saved warning",
        "process wall time",
        "bootstrap overhead",
    ):
        assert text in report
    assert row(report, "Dirty")[1:] == ["False", "True"]
    assert row(report, "Exit code")[1:] == ["0", "7"]
    assert pair == original


@pytest.mark.parametrize(
    "baseline,experiment,delta,percentage",
    [
        (10, 12.5, "+2.5 s", "+25%"),
        (10, 8, "-2 s", "-20%"),
        (0, 5, "+5 s", "N/A"),
        (0, 0, "0 s", "N/A"),
        (None, 5, "N/A", "N/A"),
        (10, None, "N/A", "N/A"),
        (None, None, "N/A", "N/A"),
    ],
)
def test_runtime_numeric_semantics(pair, baseline, experiment, delta, percentage):
    first, second = pair
    report = render_comparison(
        replace(first, runtime_seconds=baseline),
        replace(second, runtime_seconds=experiment),
    )
    cells = row(report, "Runtime")
    assert cells[3:5] == [delta, percentage]
    assert (cells[1] == "N/A") == (baseline is None)
    assert (cells[2] == "N/A") == (experiment is None)
    for judgment in ("better", "worse", "regression", "improvement"):
        assert judgment not in report


def test_unavailable_cuda_and_incomplete_lifecycle_are_explicit(pair):
    first, second = pair
    second = replace(second, status="running", exit_code=None, ended_at=None)
    report = render_comparison(first, second)
    assert "not\\_collected" in report
    assert row(report, "Status")[2] == "running"
    assert row(report, "Ended")[2] == "N/A"
    for metric in ("Peak allocated", "Peak reserved"):
        assert row(report, "N/A", metric)[2:6] == ["N/A"] * 4


def cuda_pair(pair, first_scope="whole_run", second_scope="whole_run"):
    first, second = pair
    return (
        replace(
            first,
            metrics=RunMetrics([CUDADeviceMetrics("cuda:0", 100, 200)], first_scope),
        ),
        replace(
            second,
            metrics=RunMetrics([CUDADeviceMetrics("cuda:0", 150, 100)], second_scope),
        ),
    )


def test_comparable_cuda_peaks_are_compared_separately_in_bytes(pair):
    report = render_comparison(*cuda_pair(pair))
    assert row(report, "cuda:0", "Peak allocated")[2:6] == [
        "100 bytes",
        "150 bytes",
        "+50 bytes",
        "+50%",
    ]
    assert row(report, "cuda:0", "Peak reserved")[2:6] == [
        "200 bytes",
        "100 bytes",
        "-100 bytes",
        "-50%",
    ]


@pytest.mark.parametrize(
    "scopes,note",
    [
        (("whole_run", "training_loop"), "Measurement scopes differ"),
        ((None, "whole_run"), "Measurement scope unavailable"),
        ((None, None), "Measurement scope unavailable"),
    ],
)
def test_cuda_incompatible_scopes_keep_values_without_deltas(pair, scopes, note):
    report = render_comparison(*cuda_pair(pair, *scopes))
    cells = row(report, "cuda:0", "Peak allocated")
    assert cells[2:6] == ["100 bytes", "150 bytes", "N/A", "N/A"]
    assert note in cells[6]


def test_cuda_different_devices_are_not_paired_or_summed(pair):
    first, second = cuda_pair(pair)
    second.metrics.cuda_devices[0].device = "cuda:1"
    report = render_comparison(first, second)
    assert row(report, "cuda:0", "Peak allocated")[2:6] == [
        "100 bytes",
        "N/A",
        "N/A",
        "N/A",
    ]
    assert row(report, "cuda:1", "Peak allocated")[2:6] == [
        "N/A",
        "150 bytes",
        "N/A",
        "N/A",
    ]
    assert "250 bytes" not in report


def test_cuda_partial_missing_values_preserve_measured_zero_and_reason(pair):
    first, second = cuda_pair(pair)
    first.metrics.cuda_devices = [
        CUDADeviceMetrics("cuda:0", 0, None, "reserved unavailable")
    ]
    report = render_comparison(first, second)
    assert row(report, "cuda:0", "Peak allocated")[2:6] == [
        "0 bytes",
        "150 bytes",
        "+150 bytes",
        "N/A",
    ]
    cells = row(report, "cuda:0", "Peak reserved")
    assert cells[2:6] == ["N/A", "100 bytes", "N/A", "N/A"]
    assert "reserved unavailable" in cells[6]


def test_duplicate_saved_device_identifiers_fail_without_choosing_an_entry(pair):
    first, second = cuda_pair(pair)
    first.metrics.cuda_devices.append(CUDADeviceMetrics("cuda:0", 3, 4))
    with pytest.raises(ValueError, match="Duplicate CUDA device"):
        render_comparison(first, second)


def test_markdown_escaping_keeps_user_text_inside_cells(pair):
    first, second = pair
    first = replace(
        first,
        name="a|b\nrow\ttab`tick\\slash",
        cwd="/a|b/[link](x)",
        command=["python", "train.py", "hello world", "", "a|b", "line\nnext"],
        diagnostics=RunDiagnostics(failure_message="<script>bad</script>\n# heading"),
    )
    report = render_comparison(first, second)
    assert "a\\|b<br>row\\ttab\\`tick\\\\slash" in report
    assert "/a\\|b/\\[link\\](x)" in report
    assert '"hello world", ""' in report
    assert "&lt;script&gt;bad&lt;/script&gt;<br># heading" in report
    assert "\n# heading" not in report
    width = None
    for line in report.splitlines():
        if not line.startswith("| "):
            width = None
            continue
        count = len(re.split(r"(?<!\\)\|", line))
        if width is None:
            width = count
        assert count == width
