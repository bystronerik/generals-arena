"""Submission bundler: closure selection, zip layout, limits, standalone run."""
from __future__ import annotations

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

    assert {"run.sh", "bots/smoke/run.sh", "bots/smoke/main.py",
            "bots/smoke/agent.py", "bots/_common/wire.py"} <= names
    # Nothing outside the closure: no repo dirs, caches, or other bots.
    assert all(n == "run.sh" or n.startswith(("bots/smoke/", "bots/_common/")) for n in names)
    assert not any("__pycache__" in n for n in names)


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
    for bot in ("aegis", "blitz", "boom", "metro"):
        assert f"bots/{bot}/agent.py" in names


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
    # Hash-keyed extract dir survives across runs so macOS's first-exec script
    # assessment (~0.4 s) is a cold-cache cost, not part of the warm 7 s budget.
    extract_dir = Path(tempfile.gettempdir()) / f"generals-bundle-smoke-{info.content_hash}"
    action = smoke_check(info.zip_path, extract_dir, reuse_extracted=True)
    assert len(action) == 5
    kind, row, col, direction, split = action
    assert kind in (0, 1, 2)
    assert 0 <= row < 5 and 0 <= col < 5
    assert direction in (0, 1, 2, 3)
    assert split in (0, 1)
