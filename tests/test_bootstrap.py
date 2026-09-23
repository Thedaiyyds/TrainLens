import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from trainlens.bootstrap import execute_script


def write_script(path: Path, source: str) -> Path:
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_script_context_arguments_and_same_process(tmp_path, capsys):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = write_script(
        scripts / "train.py",
        """
        import __main__
        import json
        import os
        import sys
        from pathlib import Path

        if __name__ == "__main__":
            print(json.dumps({
                "argv": sys.argv,
                "file": str(Path(__file__).resolve()),
                "cwd": os.getcwd(),
                "pid": os.getpid(),
                "main_file": __main__.__file__,
            }))
        """,
    )
    args = ["ordinary", "hello world", "", "--epochs", "5"]

    result = execute_script("scripts/train.py", args, invocation_cwd=tmp_path)

    data = json.loads(capsys.readouterr().out)
    assert result is None
    assert data["argv"][1:] == args
    assert Path(data["argv"][0]).resolve() == script.resolve()
    assert Path(data["file"]) == script.resolve()
    assert Path(data["main_file"]).resolve() == script.resolve()
    assert Path(data["cwd"]) == tmp_path.resolve()
    assert data["pid"] == os.getpid()


def test_script_can_change_cwd_and_import_sibling(tmp_path, capsys, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    helper = "trainlens_fixture_sibling"
    # Isolate the imported fixture module from the rest of the test process.
    monkeypatch.delitem(sys.modules, helper, raising=False)
    write_script(scripts / f"{helper}.py", "VALUE = 'sibling imported'\n")
    script = write_script(
        scripts / "train.py",
        f"""
        import os
        os.chdir({str(elsewhere)!r})
        import {helper}
        print({helper}.VALUE)
        print(os.getcwd())
        """,
    )
    try:
        execute_script(script, [], invocation_cwd=tmp_path)
        assert capsys.readouterr().out.splitlines() == [
            "sibling imported",
            str(elsewhere.resolve()),
        ]
    finally:
        sys.modules.pop(helper, None)


@pytest.mark.parametrize(
    "ending,exception,code",
    [
        ("pass", None, None),
        ("raise SystemExit(0)", SystemExit, 0),
        ("raise SystemExit(7)", SystemExit, 7),
        ("raise ValueError('training failed')", ValueError, None),
    ],
)
def test_outcomes_propagate_and_temporary_context_is_restored(
    tmp_path, ending, exception, code
):
    script = write_script(
        tmp_path / "train.py",
        "import os, sys\nsys.argv.append('script-added')\n"
        "sys.path.append('script-added')\nos.chdir('..')\n" + ending,
    )
    argv, import_path = sys.argv, sys.path
    argv_values, path_values = list(argv), list(import_path)
    cwd, main = Path.cwd(), sys.modules["__main__"]

    if exception is None:
        assert execute_script(script, [], invocation_cwd=tmp_path) is None
    else:
        with pytest.raises(exception) as caught:
            execute_script(script, [], invocation_cwd=tmp_path)
        if exception is SystemExit:
            assert caught.value.code == code
        else:
            assert str(caught.value) == "training failed"
            assert any(entry.path == script for entry in caught.traceback)

    assert sys.argv is argv and sys.argv == argv_values
    assert sys.path is import_path and sys.path == path_values
    assert Path.cwd() == cwd
    assert sys.modules["__main__"] is main


@pytest.mark.parametrize(
    "ending,exit_code,error",
    [
        ("pass", 0, ""),
        ("raise SystemExit(0)", 0, ""),
        ("raise SystemExit(7)", 7, ""),
        ("raise RuntimeError('script failed')", 1, "RuntimeError: script failed"),
    ],
)
def test_standard_streams_and_outer_process_outcome(tmp_path, ending, exit_code, error):
    script = write_script(
        tmp_path / "train.py",
        "import sys\nprint('stdin=' + input())\n"
        "print('user stderr', file=sys.stderr)\n" + ending,
    )
    # Only the test launches a process. The bootstrap executes train.py in it.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from trainlens.bootstrap import execute_script; "
            "execute_script(sys.argv[1], [], invocation_cwd=sys.argv[2]); "
            "print('returned normally')",
            str(script),
            str(tmp_path),
        ],
        input="hello from stdin\n",
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode == exit_code
    assert "stdin=hello from stdin\n" in result.stdout
    assert "user stderr\n" in result.stderr
    assert ("returned normally" in result.stdout) == (ending == "pass")
    if error:
        assert error in result.stderr
        assert str(script) in result.stderr
    else:
        assert "Traceback" not in result.stderr


def test_missing_script_does_not_change_execution_context(tmp_path):
    argv, import_path, cwd = sys.argv, sys.path, Path.cwd()

    with pytest.raises(FileNotFoundError):
        execute_script("missing.py", [], invocation_cwd=tmp_path)

    assert sys.argv is argv
    assert sys.path is import_path
    assert Path.cwd() == cwd


def test_directory_entry_point_is_not_executed(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    write_script(package / "__main__.py", "raise AssertionError('must not execute')")

    with pytest.raises(ValueError, match=".py script"):
        execute_script(package, [], invocation_cwd=tmp_path)
