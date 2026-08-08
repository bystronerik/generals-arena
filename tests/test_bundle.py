"""Submission bundler: closure selection, zip layout, limits, standalone run."""
from __future__ import annotations

import ast
import hashlib
import tempfile
import zipfile
from pathlib import Path

import pytest

from arena.bundle import (
    BundleError,
    audit_imports,
    bundle_members,
    check_limits,
    default_output_path,
    minify_source,
    smoke_check,
    third_party_imports,
    write_bundle,
)
from arena.records.fingerprint import bot_content_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
BOTS_DIR = REPO_ROOT / "bots"


def _names(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def test_smoke_bot_bundle_layout(tmp_path):
    info = write_bundle("smoke", tmp_path / "smoke.zip")
    names = set(_names(info.zip_path))

    assert {"run.sh", "bots/smoke/main.py",
            "bots/smoke/agent.py", "bots/_common/wire.py"} <= names
    # Nothing outside the closure: no repo dirs, caches, or other bots.
    assert all(n == "run.sh" or n.startswith(("bots/smoke/", "bots/_common/")) for n in names)
    assert not any("__pycache__" in n for n in names)
    # The repo-harness launcher stays out; the generated root run.sh is the
    # exact shape the competition server's validation accepts.
    assert "bots/smoke/run.sh" not in names
    with zipfile.ZipFile(info.zip_path) as zf:
        root = zf.read("run.sh").decode("ascii")
    assert root == "#!/usr/bin/env bash\nexec python -u bots/smoke/main.py\n"


def test_shell_scripts_keep_executable_bit(tmp_path):
    info = write_bundle("smoke", tmp_path / "smoke.zip")
    with zipfile.ZipFile(info.zip_path) as zf:
        for entry in zf.infolist():
            mode = (entry.external_attr >> 16) & 0o777
            expected = 0o755 if entry.filename.endswith(".sh") else 0o644
            assert mode == expected, entry.filename


def test_cm_bot_is_refused(tmp_path):
    with pytest.raises(BundleError, match="competition-module"):
        write_bundle("cm_random", tmp_path / "cm.zip")


def test_unknown_bot_is_refused(tmp_path):
    with pytest.raises(BundleError, match="missing run.sh"):
        write_bundle("does_not_exist", tmp_path / "x.zip")


def test_proteus_bundle_carries_cross_imported_bots():
    # Member arcnames are exactly what write_bundle zips; skipping the build
    # keeps the closure-crossing case cheap.
    names = {arcname for _, arcname in bundle_members("proteus")}
    for bot in ("blitz", "boom"):
        assert f"bots/{bot}/agent.py" in names
    for dropped in ("aegis", "metro"):
        assert f"bots/{dropped}/agent.py" not in names


def test_bundles_are_deterministic(tmp_path):
    a = write_bundle("smoke", tmp_path / "a.zip").zip_path.read_bytes()
    b = write_bundle("smoke", tmp_path / "b.zip").zip_path.read_bytes()
    assert a == b


def test_existing_output_needs_force(tmp_path):
    out = tmp_path / "smoke.zip"
    write_bundle("smoke", out)
    with pytest.raises(BundleError, match="--force"):
        write_bundle("smoke", out)
    write_bundle("smoke", out, force=True)


def test_limits_reject_oversized_bundle(tmp_path):
    info = write_bundle("smoke", tmp_path / "smoke.zip")
    with pytest.raises(BundleError, match="zip bytes"):
        check_limits(info.zip_path, max_zip_bytes=16)
    with pytest.raises(BundleError, match="unpacked bytes"):
        check_limits(info.zip_path, max_unpacked_bytes=16)
    with pytest.raises(BundleError, match="files"):
        check_limits(info.zip_path, max_files=1)


def test_import_audit_flags_unknown_package(tmp_path):
    offender = tmp_path / "agent.py"
    offender.write_text("import numpy\nimport made_up_pkg\n", encoding="utf-8")
    assert third_party_imports([offender]) == {"numpy", "made_up_pkg"}
    with pytest.raises(BundleError, match="made_up_pkg"):
        audit_imports([offender])


def test_default_output_path_carries_content_hash():
    digest = bot_content_hash(BOTS_DIR / "smoke" / "run.sh")
    path = default_output_path("smoke")
    assert path.name == f"smoke-{digest}.zip"
    assert path.parent == REPO_ROOT / "data" / "bundles"


@pytest.mark.slow
def test_bundle_runs_standalone(tmp_path):
    info = write_bundle("smoke", tmp_path / "smoke.zip")
    # Digest-keyed extract dir survives across runs so macOS's first-exec
    # script assessment (~0.4 s) is a cold-cache cost, not part of the warm
    # 12 s budget. Keyed on the zip bytes, not the source hash: the bundler
    # itself changing must invalidate the cache too.
    digest = hashlib.sha256(info.zip_path.read_bytes()).hexdigest()[:12]
    extract_dir = Path(tempfile.gettempdir()) / f"generals-bundle-smoke-{digest}"
    action = smoke_check(info.zip_path, extract_dir, reuse_extracted=True)
    assert len(action) == 5
    kind, row, col, direction, split = action
    assert kind in (0, 1, 2)
    assert 0 <= row < 5 and 0 <= col < 5
    assert direction in (0, 1, 2, 3)
    assert split in (0, 1)


def test_the_bundle_carries_no_instrumentation(tmp_path):
    """
    The submitted program is game logic only.

    Probes and the instrumented runner live outside every closure, so neither
    can reach a zip — and `wire.py` is a bare protocol loop again.
    """
    info = write_bundle("metro", tmp_path / "metro.zip")
    with zipfile.ZipFile(info.zip_path) as zf:
        names = zf.namelist()
        wire = zf.read("bots/_common/wire.py").decode("utf-8")

    assert not any(n.endswith("probe.py") for n in names)
    assert not any(n.startswith("arena/") for n in names)
    for symbol in ("telemetry", "_telemetry_line", "telemetry_extras"):
        assert symbol not in wire


def test_minify_drops_prose_and_keeps_behaviour():
    source = (
        '''"""Module doc."""\n'''
        "import os  # trailing comment\n"
        "\n"
        "LIMIT = 3\n"
        '''"""Attribute prose: not an AST docstring, still prose."""\n'''
        "\n"
        "def area(width: int, height: int) -> int:\n"
        '''    """Docstring."""\n'''
        "    scale_factor = 2\n"
        "    return width * height * scale_factor\n"
    )
    out = minify_source(source, "sample.py")

    assert "#" not in out and '"""' not in out
    assert "scale_factor" not in out          # locals are renamed
    assert "def area(" in out                 # globals are not: imports rely on them
    assert "LIMIT=3" in out.replace(" ", "")

    scope: dict = {}
    exec(compile(out, "sample.py", "exec"), scope)
    assert scope["area"](3, 4) == 24
    assert scope["LIMIT"] == 3
    assert scope["area"].__doc__ is None


def test_minify_leaves_string_values_alone():
    out = minify_source('HELP = "a value, not prose"\n', "v.py")
    scope: dict = {}
    exec(compile(out, "v.py", "exec"), scope)
    assert scope["HELP"] == "a value, not prose"


def test_minify_rejects_unparseable_source():
    with pytest.raises(BundleError, match="cannot minify"):
        minify_source("def f(\n", "broken.py")


def test_bundle_ships_minified_python_and_leaves_the_repo_intact(tmp_path):
    agent_src = BOTS_DIR / "smoke" / "agent.py"
    before = agent_src.read_bytes()
    assert b'"""' in before, "the fixture must have prose to strip"

    info = write_bundle("smoke", tmp_path / "smoke.zip")
    with zipfile.ZipFile(info.zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".py"):
                continue
            text = zf.read(name).decode("utf-8")
            ast.parse(text)  # the minified member is still valid Python
            assert not text.startswith(("\n", " ", "\t")), f"{name}: opens with blank lines"
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    assert ast.get_docstring(node) is None, f"{name}: docstring survived"

    assert agent_src.read_bytes() == before, "bundling must not rewrite repo sources"


def test_minified_bundle_is_smaller_than_the_verbatim_one(tmp_path):
    small = write_bundle("smoke", tmp_path / "small.zip")
    plain = write_bundle("smoke", tmp_path / "plain.zip", minify=False)
    assert small.unpacked_bytes < plain.unpacked_bytes


def test_no_minify_ships_sources_verbatim(tmp_path):
    info = write_bundle("smoke", tmp_path / "verbatim.zip", minify=False)
    with zipfile.ZipFile(info.zip_path) as zf:
        assert zf.read("bots/smoke/agent.py") == (BOTS_DIR / "smoke" / "agent.py").read_bytes()


def test_a_probe_beside_the_agent_stays_out_of_the_bundle():
    """metro has a probe on disk; the bundle must not notice."""
    assert (BOTS_DIR / "metro" / "probe.py").is_file()
    names = {arcname for _, arcname in bundle_members("metro")}
    assert "bots/metro/probe.py" not in names


def test_bot_local_tests_stay_out_of_the_bundle():
    """sosipolis ships a real tests/; the submission zip must omit it."""
    assert (BOTS_DIR / "sosipolis" / "tests").is_dir()
    names = {arcname for _, arcname in bundle_members("sosipolis")}
    assert not any(
        arcname == "bots/sosipolis/tests"
        or arcname.startswith("bots/sosipolis/tests/")
        for arcname in names
    )
    # Bundle membership is the hashed closure (minus run.sh); keep that link tight.
    assert "bots/sosipolis/brain.py" in names
    assert "bots/sosipolis/probe.py" not in names
