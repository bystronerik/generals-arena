"""Bot version registry: hash -> closure -> git ref -> diff, and the revert case."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import arena.records.fingerprint as fingerprint
from arena.paths import REPO_ROOT
from arena.records.registry import Registry, RegistryError, Step, closure_delta, REF_PREFIX


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """One throwaway git repo for the module — `git init` is the slow part."""
    root = tmp_path_factory.mktemp("registry") / "repo"
    bots = root / "bots" / "_common"
    bots.mkdir(parents=True)
    (bots / "__init__.py").write_text("", encoding="utf-8")
    (bots / "wire.py").write_text("VERSION = 1\n", encoding="utf-8")

    root.parent.mkdir(parents=True, exist_ok=True)
    _git(root.parent, "init", "-q", str(root))
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    return root


@dataclass
class Sandbox:
    """One test's own bot inside the shared repo, plus its own registry."""

    repo: Path
    bot_dir: Path
    registry: Registry

    def write_agent(self, body: str) -> None:
        (self.bot_dir / "agent.py").write_text(
            f"from _common import wire\n{body}\n", encoding="utf-8"
        )

    def commit(self) -> None:
        _git(self.repo, "add", "-A", str(self.bot_dir))
        _git(self.repo, "commit", "-q", "-m", f"add {self.bot_dir.name}")

    def register(self, **kwargs):
        return self.registry.register(self.bot_dir, **kwargs)

    def closure_names(self) -> set[str]:
        return {
            f"bots/{self.bot_dir.name}/agent.py",
            f"bots/{self.bot_dir.name}/run.sh",
            "bots/_common/__init__.py",
            "bots/_common/wire.py",
        }


@pytest.fixture
def sandbox(repo, request, monkeypatch):
    """
    A fresh bot per test, so tests can edit sources without colliding.

    Left uncommitted — only the two dirty-tree tests care, and `git commit` is
    expensive enough per test to matter against the suite's 7 s budget.
    """
    name = f"toy_{abs(hash(request.node.name)) % 10**8}"
    bot_dir = repo / "bots" / name
    bot_dir.mkdir()
    (bot_dir / "run.sh").write_text("#!/bin/sh\nexec python main.py\n", encoding="utf-8")

    monkeypatch.setattr(fingerprint, "REPO_ROOT", repo)
    monkeypatch.setattr(fingerprint, "BOTS_DIR", repo / "bots")

    box = Sandbox(
        repo=repo,
        bot_dir=bot_dir,
        registry=Registry(repo / "data" / "bot_versions" / name),
    )
    box.write_agent("MOVE = 'up'")
    return box


# --- T10: registry round-trip (hash -> closure -> diff) ---------------------


def test_register_records_closure_and_anchors_a_ref(sandbox):
    version, is_new = sandbox.register()

    assert is_new is True
    assert version.content_hash == fingerprint.content_hash_for_dir(sandbox.bot_dir)
    assert set(version.paths()) == sandbox.closure_names()
    # Recorded digests match both hash schemes.
    for entry in version.files:
        assert entry.blob == _git(sandbox.repo, "hash-object", str(sandbox.repo / entry.path))
    assert version.closure_ref == f"{REF_PREFIX}{version.content_hash}"
    listed = _git(sandbox.repo, "ls-tree", "-r", "--name-only", version.closure_ref).split()
    assert listed == sorted(version.paths())


def test_register_is_idempotent(sandbox):
    first, is_new = sandbox.register()
    again, is_new_again = sandbox.register()

    assert (is_new, is_new_again) == (True, False)
    assert first == again
    entry = sandbox.registry.load(sandbox.bot_dir.name)
    assert len(entry.versions) == 1
    assert len(entry.steps) == 1


def test_edit_makes_a_new_version_and_the_refs_diff_shows_it(sandbox):
    before, _ = sandbox.register()
    sandbox.write_agent("MOVE = 'down'")
    after, is_new = sandbox.register()

    assert is_new is True
    assert after.content_hash != before.content_hash
    assert closure_delta(before, after) == {
        "added": [],
        "removed": [],
        "changed": [f"bots/{sandbox.bot_dir.name}/agent.py"],
    }
    diff = _git(sandbox.repo, "diff", before.closure_ref, after.closure_ref)
    assert f"bots/{sandbox.bot_dir.name}/agent.py" in diff
    assert "bots/_common/wire.py" not in diff


def test_revert_pools_the_entity_and_appends_a_step(sandbox):
    """A reverted hash is the same program: one entity, three lineage steps."""
    first, _ = sandbox.register()
    sandbox.write_agent("MOVE = 'down'")
    second, _ = sandbox.register()
    sandbox.write_agent("MOVE = 'up'")
    third, is_new = sandbox.register()

    assert is_new is False
    assert third == first  # the entry is never mutated

    entry = sandbox.registry.load(sandbox.bot_dir.name)
    assert [v.content_hash for v in entry.versions] == [
        first.content_hash,
        second.content_hash,
    ]
    assert [(s.seq, s.content_hash) for s in entry.steps] == [
        (1, first.content_hash),
        (2, second.content_hash),
        (3, first.content_hash),
    ]
    assert entry.steps[2].note == "revert to seq 1"


def test_registry_file_survives_a_reload(sandbox):
    version, _ = sandbox.register()
    fresh = Registry(sandbox.registry.directory)

    assert fresh.is_registered(sandbox.bot_dir.name, version.content_hash)
    assert fresh.version(sandbox.bot_dir.name, version.content_hash) == version
    assert fresh.bot_ids() == [sandbox.bot_dir.name]
    assert fresh.find_hash(version.content_hash) == (sandbox.bot_dir.name, version)


def test_require_registered_raises_for_an_unknown_hash(sandbox):
    sandbox.register()
    with pytest.raises(RegistryError, match="not registered"):
        sandbox.registry.require_registered(sandbox.bot_dir.name, "deadbeefcafe")


def test_strict_refuses_a_dirty_closure(sandbox):
    sandbox.write_agent("X = 2")  # edited but not committed

    with pytest.raises(RegistryError, match="--strict"):
        sandbox.register(strict=True)

    version, _ = sandbox.register()
    assert version.git_dirty is True


def test_clean_closure_is_not_dirty(sandbox):
    sandbox.commit()
    version, _ = sandbox.register(strict=True)
    assert version.git_dirty is False
    assert version.git_commit == _git(sandbox.repo, "rev-parse", "HEAD")


# --- T11: the registry is committed, not ignored ---------------------------


def test_registry_dir_is_not_gitignored():
    """A future `data/*` rule must not silently hide the version history."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "data/bot_versions/x.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "data/bot_versions/ must stay committed"


# --- T12 (registry half): verify never changes a rating --------------------


def test_verify_is_clean_after_registration(sandbox):
    sandbox.register()
    assert sandbox.registry.verify() == []


def test_verify_flags_an_unreachable_commit_without_touching_versions(sandbox):
    version, _ = sandbox.register()
    entry = sandbox.registry.load(sandbox.bot_dir.name)
    entry.versions[0] = replace(version, git_commit="0" * 40)
    sandbox.registry.save(entry)

    issues = sandbox.registry.verify()
    assert [i["kind"] for i in issues] == ["git-unresolvable"]
    assert "unreachable" in issues[0]["detail"]
    # The rated identity is unchanged: content, not git, defines it.
    assert sandbox.registry.is_registered(sandbox.bot_dir.name, version.content_hash)


def test_verify_flags_a_step_pointing_at_an_unknown_hash(sandbox):
    sandbox.register()
    entry = sandbox.registry.load(sandbox.bot_dir.name)
    entry.steps.append(Step(seq=2, content_hash="ffffffffffff", at="2026-01-01T00:00:00Z"))
    sandbox.registry.save(entry)

    assert "step-unknown-hash" in [i["kind"] for i in sandbox.registry.verify()]


# --- step 4: workers assert, they never write ------------------------------


def test_worker_refuses_a_match_whose_bots_are_unregistered(sandbox, monkeypatch):
    """A pool worker must fail the match rather than store an unrateable game."""
    import arena.records.registry as registry_module
    from arena.tournaments.worker import run_one_worker

    monkeypatch.setattr(registry_module, "REGISTRY_DIR", sandbox.registry.directory)
    payload = {
        "bot_a_run": str(sandbox.bot_dir / "run.sh"),
        "bot_b_run": str(sandbox.bot_dir / "run.sh"),
        "seed": 0,
        "games_dir": str(sandbox.repo / "games"),
        "round": "roundT",
        "engine_version": "9e3b9d1",
        "bot_a_content_hash": "deadbeefcafe",
        "bot_b_content_hash": "deadbeefcafe",
    }
    with pytest.raises(RegistryError, match="not registered"):
        run_one_worker(payload)


@pytest.mark.parametrize(
    "key", ["round", "engine_version", "bot_a_content_hash", "bot_b_content_hash"]
)
def test_worker_refuses_a_payload_missing_identity(sandbox, key):
    """The parent supplies identity; a worker never invents it."""
    from arena.tournaments.worker import run_one_worker

    payload = {
        "bot_a_run": str(sandbox.bot_dir / "run.sh"),
        "bot_b_run": str(sandbox.bot_dir / "run.sh"),
        "seed": 0,
        "games_dir": str(sandbox.repo / "games"),
        "round": "roundT",
        "engine_version": "9e3b9d1",
        "bot_a_content_hash": "deadbeefcafe",
        "bot_b_content_hash": "deadbeefcafe",
    }
    del payload[key]
    with pytest.raises(ValueError, match=key):
        run_one_worker(payload)


def test_expander_python_is_not_registerable(sandbox):
    """It lives outside bots/, so its closure — and its hash — mean nothing."""
    from arena.records.registry import is_registerable

    assert is_registerable(sandbox.bot_dir) is True
    assert is_registerable(sandbox.repo / "elsewhere" / "expander_python") is False

    with pytest.raises(RegistryError, match="not under"):
        sandbox.registry.register_run_scripts(
            [sandbox.repo / "elsewhere" / "expander_python" / "run.sh"]
        )
