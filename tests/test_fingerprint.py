"""Bot content-hash closure: what it covers, and what it ignores."""
from __future__ import annotations

import pytest

from arena.records.fingerprint import (
    BOTS_DIR,
    HASH_LENGTH,
    UnhashableBotError,
    bot_content_hash,
    bot_content_hashes,
    bot_source_closure,
    content_hash_for_dir,
)


def closure_names(bot_id: str) -> set[str]:
    return {
        p.relative_to(BOTS_DIR).as_posix() for p in bot_source_closure(BOTS_DIR / bot_id)
    }


def test_hash_is_short_hex():
    digest = bot_content_hash(BOTS_DIR / "smoke" / "run.sh")
    assert len(digest) == HASH_LENGTH
    assert all(c in "0123456789abcdef" for c in digest)


def test_hash_is_stable_across_calls():
    run_sh = BOTS_DIR / "smoke" / "run.sh"
    assert bot_content_hash(run_sh) == content_hash_for_dir(run_sh.parent)


def test_closure_includes_own_sources_and_shared_wire():
    names = closure_names("smoke")
    assert {"smoke/agent.py", "smoke/main.py", "smoke/run.sh", "_common/wire.py"} <= names


def test_closure_excludes_pycache():
    assert not any("__pycache__" in name for name in closure_names("smoke"))


def test_closure_follows_transitive_shared_imports():
    # aegis -> _common.oppmodel -> _common.tactics -> _common.strategy_common
    names = closure_names("aegis")
    assert "_common/oppmodel.py" in names
    assert "_common/tactics.py" in names
    assert "_common/strategy_common.py" in names


def test_closure_crosses_bot_directories():
    """proteus dispatches to other bots; editing them changes proteus."""
    names = closure_names("proteus")
    for dependency in ("blitz", "boom"):
        assert f"{dependency}/agent.py" in names, dependency
    # Dropped from the counter map, so no longer in the closure: a core
    # proteus cannot select must not be able to fork proteus's hash.
    for dropped in ("aegis", "metro"):
        assert f"{dropped}/agent.py" not in names, dropped
    assert "proteus/switcher.py" in names
    assert "proteus/classifier.py" in names


def test_closure_follows_shell_source_directives():
    """cm_* launchers `source` a shared body no AST walk would find."""
    assert "_common/cm_run.sh" in closure_names("cm_random")


def test_closure_stops_at_third_party_imports():
    """`generals` and jax are pinned by the lockfile, not by this hash."""
    assert all(
        not name.startswith(("generals", "jax", "numpy")) for name in closure_names("cm_random")
    )


def test_unrelated_bots_have_different_hashes():
    hashes = bot_content_hashes(
        [BOTS_DIR / name / "run.sh" for name in ("smoke", "aegis", "proteus")]
    )
    assert len(set(hashes.values())) == 3
    assert set(hashes) == {"smoke", "aegis", "proteus"}


def test_missing_bot_dir_raises_rather_than_returning_a_sentinel():
    """A `"unknown"` hash would pool unrelated programs into one rated entity."""
    with pytest.raises(UnhashableBotError):
        bot_content_hash(BOTS_DIR / "does_not_exist" / "run.sh")


@pytest.mark.parametrize(
    "target, dependents, unaffected",
    [
        ("_common/wire.py", ("smoke", "aegis", "cm_random"), ()),
        ("_common/tactics.py", ("aegis",), ("smoke",)),
        ("blitz/agent.py", ("blitz", "proteus"), ("smoke", "aegis")),
        ("_common/cm_adapter.py", ("cm_random",), ("smoke", "aegis", "proteus")),
    ],
)
def test_edit_propagates_to_exactly_the_dependents(
    tmp_path, monkeypatch, target, dependents, unaffected
):
    """Touching a shared file must move its dependents' hashes and nothing else."""
    import shutil

    import arena.records.fingerprint as fingerprint

    sandbox = tmp_path / "bots"
    shutil.copytree(BOTS_DIR, sandbox, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(fingerprint, "BOTS_DIR", sandbox)
    monkeypatch.setattr(fingerprint, "REPO_ROOT", tmp_path)

    watched = list(dependents) + list(unaffected)
    before = {name: fingerprint.content_hash_for_dir(sandbox / name) for name in watched}

    edited = sandbox / target
    edited.write_text(
        edited.read_text(encoding="utf-8") + "\n# behaviour change\n", encoding="utf-8"
    )

    after = {name: fingerprint.content_hash_for_dir(sandbox / name) for name in watched}
    for name in dependents:
        assert before[name] != after[name], f"{name} should depend on {target}"
    for name in unaffected:
        assert before[name] == after[name], f"{name} should not depend on {target}"


# --- probe.py: outside the closure, unreachable from inside it ---------------


def test_probe_is_absent_from_a_real_closure():
    """
    A probe is arena-owned introspection; it never plays, so it never hashes.

    proteus alone is the whole invariant on one closure walk: it carries a
    probe *and* cross-imports blitz and boom, each of which carries one too —
    so this covers the cross-directory leak, not just the bot's own
    directory.
    """
    assert (BOTS_DIR / "proteus" / "probe.py").is_file()
    assert (BOTS_DIR / "boom" / "probe.py").is_file()
    names = closure_names("proteus")
    assert "boom/agent.py" in names  # the cross-import is live
    assert not any(name.endswith("probe.py") for name in names)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """
    Two throwaway bots, so the closure rules can be exercised in milliseconds.

    Deliberately synthetic rather than a copy of `bots/`: these tests are about
    the hashing rule, not about any real bot's imports.
    """
    import arena.records.fingerprint as fingerprint

    root = tmp_path / "bots"
    for name, agent_src in (("one", "VALUE = 1\n"), ("two", "from one.agent import VALUE\n")):
        bot = root / name
        bot.mkdir(parents=True)
        (bot / "agent.py").write_text(agent_src, encoding="utf-8")
        (bot / "main.py").write_text("import agent\n", encoding="utf-8")
        (bot / "run.sh").write_text("exec python main.py\n", encoding="utf-8")
    monkeypatch.setattr(fingerprint, "BOTS_DIR", root)
    monkeypatch.setattr(fingerprint, "REPO_ROOT", tmp_path)
    return fingerprint, root


def test_adding_or_editing_a_probe_does_not_move_the_hash(sandbox):
    """The point of the exclusion: instrumentation stops being hash-coupled."""
    fingerprint, root = sandbox
    before = fingerprint.content_hash_for_dir(root / "one")

    probe = root / "one" / "probe.py"
    probe.write_text("def extras(agent):\n    return {}\n", encoding="utf-8")
    assert fingerprint.content_hash_for_dir(root / "one") == before

    probe.write_text("def extras(agent):\n    return {'phase': 'x'}\n", encoding="utf-8")
    assert fingerprint.content_hash_for_dir(root / "one") == before
    assert "one/probe.py" not in {
        p.relative_to(root).as_posix() for p in fingerprint.bot_source_closure(root / "one")
    }


def test_bot_local_tests_dir_does_not_move_the_hash(sandbox):
    """Unit fixtures under bots/<name>/tests/ stay outside the content hash."""
    fingerprint, root = sandbox
    before = fingerprint.content_hash_for_dir(root / "one")

    tests = root / "one" / "tests"
    tests.mkdir()
    (tests / "test_logic.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    assert fingerprint.content_hash_for_dir(root / "one") == before
    assert not any(
        "tests" in p.relative_to(root).parts
        for p in fingerprint.bot_source_closure(root / "one")
    )


def test_sosipolis_tests_stay_out_of_the_live_closure():
    """Real bots/<name>/tests/ must not enter the hashed source closure."""
    tests_dir = BOTS_DIR / "sosipolis" / "tests"
    assert tests_dir.is_dir(), "sosipolis tests/ is the live fixture for this guard"
    names = closure_names("sosipolis")
    assert not any("tests" in name.split("/") for name in names)
    assert bot_content_hash(BOTS_DIR / "sosipolis" / "run.sh") == content_hash_for_dir(
        BOTS_DIR / "sosipolis"
    )


def test_a_closure_module_importing_probe_raises(sandbox):
    """
    Unhashed code must be unreachable from the hashed program.

    A probe the agent imports could change how the bot plays while leaving its
    rating identity untouched — a silent under-hash. Fail loudly instead.
    """
    fingerprint, root = sandbox
    (root / "one" / "probe.py").write_text("def extras(agent):\n    return {}\n", encoding="utf-8")
    agent = root / "one" / "agent.py"
    agent.write_text("import probe\n" + agent.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(fingerprint.ProbeInClosureError, match="probe"):
        fingerprint.content_hash_for_dir(root / "one")


def test_the_guard_also_catches_a_cross_bot_probe_import(sandbox):
    fingerprint, root = sandbox
    (root / "one" / "probe.py").write_text("def extras(agent):\n    return {}\n", encoding="utf-8")
    agent = root / "two" / "agent.py"
    agent.write_text("from one import probe\n" + agent.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(fingerprint.ProbeInClosureError, match="one/probe.py"):
        fingerprint.content_hash_for_dir(root / "two")


# --- compiled bots ----------------------------------------------------------


def test_a_compiled_bots_identity_is_its_sources_not_its_build():
    """
    morpheus-rs is Rust: the same sources produce `target/` on every build, and
    `vendor/` is a copy of crates.io that `Cargo.lock` already pins exactly.
    Hashing either would fork the rating identity on every `cargo build` and
    put ten thousand vendored files behind one bot's hash (rewrite-plan §10).
    """
    names = closure_names("morpheus-rs")
    assert names, "morpheus-rs is the live fixture for the compiled-bot rules"
    assert not any("target" in name.split("/") for name in names)
    assert not any("vendor" in name.split("/") for name in names)
    # What *is* the identity: the Rust sources, the lock file, the launcher,
    # and the pins that decide how they compile.
    assert "morpheus-rs/run.sh" in names
    assert "morpheus-rs/Cargo.lock" in names
    assert any(name.endswith(".rs") for name in names)


def test_developer_tooling_does_not_fork_a_rating_identity():
    """
    Same rule as `probe.py`, one directory up: the capture module the arena
    loads for a corpus run and the submission packager ship nowhere and never
    play, so editing them must not re-identify the bot.
    """
    assert (BOTS_DIR / "morpheus-rs" / "tools").is_dir()
    assert not any("tools" in name.split("/") for name in closure_names("morpheus-rs"))


def test_a_shell_comment_cannot_drag_in_another_bot():
    """
    `_SHELL_REF_RE` scans shell sources for bot-relative paths and cannot tell
    a comment from a `source` line, so naming another bot's launcher in prose
    would put that bot inside this one's hash — and editing Python morpheus
    would silently re-identify the Rust one. This caught exactly that.
    """
    names = closure_names("morpheus-rs")
    assert not any(name.startswith("morpheus/") for name in names)
