import os
import subprocess
from pathlib import Path

import pytest

from trainlens.collectors import git


def command(cwd: Path, *args: str) -> str:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "project [literal] space"
    path.mkdir()
    command(path, "init")
    command(path, "config", "user.name", "TrainLens Test")
    command(path, "config", "user.email", "test@example.invalid")
    command(path, "config", "commit.gpgsign", "false")
    (path / "train.py").write_text("print('test')\n")
    command(path, "add", "train.py")
    command(path, "commit", "-m", "fixture")
    return path


def test_clean_repository_and_commit(repo):
    info = git.collect_git(repo)
    assert info.commit == command(repo, "rev-parse", "HEAD")
    assert info.dirty is False
    assert info.unavailable_reason is None


@pytest.mark.parametrize("kind", ["tracked", "staged", "untracked"])
def test_dirty_changes(repo, kind):
    path = repo / (
        "untracked file\nwith newline.txt" if kind == "untracked" else "train.py"
    )
    path.write_text("changed\n")
    if kind == "staged":
        command(repo, "add", "train.py")
    assert git.collect_git(repo).dirty is True


@pytest.mark.parametrize("nested", [False, True])
def test_generated_store_does_not_make_repo_dirty(repo, monkeypatch, tmp_path, nested):
    cwd = repo / "nested [x]" if nested else repo
    cwd.mkdir(exist_ok=True)
    records = cwd / ".trainlens" / "runs"
    records.mkdir(parents=True)
    (records / "record.json").write_text("{}")
    monkeypatch.chdir(tmp_path)

    info = git.collect_git(cwd)

    assert info.commit == command(repo, "rev-parse", "HEAD")
    assert info.dirty is False
    (repo / "train.py").write_text("changed outside nested cwd")
    assert git.collect_git(cwd).dirty is True


def test_store_exclusion_does_not_hide_other_paths(repo):
    (repo / ".trainlens-other").write_text("user data")
    assert git.collect_git(repo).dirty is True


def test_inherited_git_overrides_do_not_redirect_collection(
    repo, monkeypatch, tmp_path
):
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "missing.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    monkeypatch.setenv("GIT_LITERAL_PATHSPECS", "1")
    (repo / ".trainlens").mkdir()
    (repo / ".trainlens" / "data").write_text("generated")

    info = git.collect_git(repo)

    assert info.commit == command(repo, "rev-parse", "HEAD")
    assert info.dirty is False
    assert os.environ["GIT_WORK_TREE"] == str(tmp_path)


def test_non_git_directory(tmp_path):
    info = git.collect_git(tmp_path)
    assert info.commit is None
    assert info.dirty is None
    assert info.unavailable_reason


def test_unborn_repository_preserves_known_dirty_state(tmp_path):
    command(tmp_path, "init")
    (tmp_path / "untracked").write_text("new")
    info = git.collect_git(tmp_path)
    assert info.commit is None
    assert info.dirty is True
    assert "commit unavailable" in info.unavailable_reason


@pytest.mark.parametrize(
    "error", [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 10)]
)
def test_git_unavailable_or_timeout(tmp_path, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(git.subprocess, "run", fail)
    info = git.collect_git(tmp_path)
    assert info.commit is None
    assert info.dirty is None
    assert info.unavailable_reason


def test_status_failure_preserves_commit(repo, monkeypatch):
    real_git = git._git

    def fail_status(cwd, *args):
        if args[0] == "status":
            raise subprocess.CalledProcessError(128, "git status")
        return real_git(cwd, *args)

    monkeypatch.setattr(git, "_git", fail_status)
    info = git.collect_git(repo)
    assert info.commit == command(repo, "rev-parse", "HEAD")
    assert info.dirty is None
    assert "status unavailable" in info.unavailable_reason
