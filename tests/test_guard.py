"""The git guard, in a throwaway repository."""

import os
import subprocess

import pytest

from saha.tools import guard


def run(*args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    run("git", "init", "-q", "-b", "work", cwd=tmp_path)
    run("git", "config", "user.email", "t@example.com", cwd=tmp_path)
    run("git", "config", "user.name", "t", cwd=tmp_path)
    run("git", "config", "commit.gpgsign", "false", cwd=tmp_path)
    (tmp_path / "README.md").write_text("hello\n")
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "possession_metrics.py").write_text("x = 1\n")
    run("git", "add", "-A", cwd=tmp_path)
    run("git", "commit", "-q", "-m", "base", cwd=tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SAHA_UPSTREAM_SYNC", raising=False)
    return tmp_path


def stage(repo, path, content):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content if isinstance(content, bytes) else content.encode())
    run("git", "add", "-f", path, cwd=repo)


@pytest.mark.parametrize("path, content, reason", [
    ("data/tactical/x.txt", lambda: "1", "data/"),
    ("cache/evidence/w.parquet", lambda: "1", "data/"),
    ("submissions/SAHA.csv", lambda: "frame,label\n", "data/"),
    ("notes/debug.npy", lambda: "1", "data-dump"),
    ("scratch/BAR-ATH_tactical_atd_head.txt", lambda: "1", "match asset"),
    ("scratch/big.bin", lambda: b"\0" * (600 * 1024), "KB"),
    ("scratch/frames.csv",
     lambda: "frame,x\n" + "".join("{},0.5\n".format(i) for i in range(6000)), "frame-level"),
], ids=["data", "cache", "submissions", "npy", "asset-name", "large", "table"])
def test_staged_data_is_refused(repo, path, content, reason):
    stage(repo, path, content())
    report = guard.staged_problems(allow_upstream_sync=False)
    assert report and reason in "; ".join(report[0][1])
    assert guard.main([]) == 1


def test_small_code_passes_and_scorer_is_protected(repo, monkeypatch):
    stage(repo, "saha/x.py", "print('ok')\n")
    assert guard.staged_problems(False) == []
    stage(repo, "evaluation/possession_metrics.py", "x = 2\n")
    assert "official scorer" in guard.staged_problems(False)[0][1][0]
    assert guard.staged_problems(True) == []          # an upstream sync may change it


def test_pre_push_refuses_main_and_scans_pushed_commits(repo, monkeypatch):
    base = run("git", "rev-parse", "HEAD", cwd=repo).strip()
    stage(repo, "notes/leak.parquet", "1")
    run("git", "commit", "-q", "--no-verify", "-m", "oops", cwd=repo)
    stage(repo, "notes/leak.parquet", "")
    run("git", "rm", "-q", "--cached", "notes/leak.parquet", cwd=repo)
    run("git", "commit", "-q", "-m", "remove it again", cwd=repo)
    head = run("git", "rev-parse", "HEAD", cwd=repo).strip()

    monkeypatch.setenv("PRE_COMMIT_REMOTE_BRANCH", "refs/heads/ws2/restarts")
    monkeypatch.setenv("PRE_COMMIT_FROM_REF", base)
    monkeypatch.setenv("PRE_COMMIT_TO_REF", head)
    assert guard.main(["--pre-push"]) == 1            # deleted later, but still in the commits

    monkeypatch.setenv("PRE_COMMIT_FROM_REF", head)
    assert guard.main(["--pre-push"]) == 0
    monkeypatch.setenv("PRE_COMMIT_REMOTE_BRANCH", "refs/heads/main")
    assert guard.main(["--pre-push"]) == 1
    monkeypatch.setenv("SAHA_ALLOW_MAIN_PUSH", "1")
    assert guard.main(["--pre-push"]) == 0


def test_history_scan_finds_removed_files(repo):
    stage(repo, "cache/frame_table.parquet", "1")
    run("git", "commit", "-q", "--no-verify", "-m", "oops", cwd=repo)
    run("git", "rm", "-q", "-r", "cache", cwd=repo)
    run("git", "commit", "-q", "-m", "cleanup", cwd=repo)
    report = guard.scan_objects(["--all"])
    assert [path for path, _ in report] == ["cache/frame_table.parquet"]
    assert guard.main(["--all"]) == 0                 # the current tree is clean
    assert guard.main(["--history"]) == 1
    assert os.path.exists(repo / "README.md")
