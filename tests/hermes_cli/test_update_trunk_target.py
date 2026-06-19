"""Tests for _fleet_trunk_remote_ref: `hermes update` must target the fleet
trunk (the branch's configured @{upstream}, e.g. realfork/main) and REFUSE when
that would resolve to the public NousResearch upstream (the trunk-wipe footgun).
"""
import subprocess
import pytest
from hermes_cli.main import _fleet_trunk_remote_ref

NOUS = "https://github.com/NousResearch/hermes-agent.git"
FORK = "https://github.com/sacwooky/hermes-agent.git"


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _make_repo(tmp_path, name, *, fork=False, upstream_remote=None):
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "f").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    _git(repo, "remote", "add", "origin", NOUS)
    if fork:
        _git(repo, "remote", "add", "realfork", FORK)
    if upstream_remote:
        # fabricate the remote-tracking ref + set main's upstream (no network)
        _git(repo, "update-ref", f"refs/remotes/{upstream_remote}/main", sha)
        _git(repo, "branch", f"--set-upstream-to={upstream_remote}/main", "main")
    return str(repo)


def test_resolves_fork_trunk(tmp_path):
    # main tracks realfork/main (the fleet trunk) -> use it
    repo = _make_repo(tmp_path, "r1", fork=True, upstream_remote="realfork")
    assert _fleet_trunk_remote_ref(["git"], repo, "main") == ("realfork", "main")


def test_refuses_when_upstream_is_nousresearch(tmp_path):
    # main tracks origin, and origin is the NousResearch upstream -> REFUSE
    repo = _make_repo(tmp_path, "r2", upstream_remote="origin")
    assert _fleet_trunk_remote_ref(["git"], repo, "main") == (None, None)


def test_no_upstream_falls_back_then_guards(tmp_path):
    # no configured upstream -> fall back to origin -> origin is NousResearch -> REFUSE
    repo = _make_repo(tmp_path, "r3")
    assert _fleet_trunk_remote_ref(["git"], repo, "zzz-none") == (None, None)
