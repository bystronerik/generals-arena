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
    _is_prose,
    audit_imports,
    bundle_members,
    check_limits,
    default_output_path,
    smoke_check,
    strip_comments,
    strip_prose_strings,
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
    # 7 s budget. Keyed on the zip bytes, not the source hash: the bundler
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


def test_strip_comments_keeps_code_strings_and_line_numbers():
    source = (
        "#!/usr/bin/env python\n"
        "# leading comment\n"
        'BASE = "#4 not a comment"  # trailing comment\n'
        "\n"
        "def f():\n"
        "    return 1  # tail\n"
    )
    stripped = strip_comments(source)

    assert stripped == (
        "#!/usr/bin/env python\n"      # shebang is an exec directive, not a comment
        "\n"
        'BASE = "#4 not a comment"\n'  # the `#` inside the literal survives
        "\n"
        "def f():\n"
        "    return 1\n"
    )
    # Line numbers are preserved, so judge tracebacks still point at the source.
    assert len(stripped.splitlines()) == len(source.splitlines())


def test_strip_prose_blanks_docstrings_and_keeps_line_numbers():
    source = (
        '"""Module doc.\n\nSecond paragraph.\n"""\n'
        "import os\n"
        "\n"
        "class C:\n"
        '    """Class doc."""\n'
        "    def f(self):\n"
        '        """Only statement in the body."""\n'
        "    def g(self):\n"
        '        """Doc, then code."""\n'
        '        return "a real string"\n'
    )
    stripped = strip_prose_strings(source)

    assert stripped == (
        "\n\n\n\n"
        "import os\n"
        "\n"
        "class C:\n"
        "\n"                      # class keeps a body: f and g follow
        "    def f(self):\n"
        "        pass\n"          # emptied body needs a statement
        "    def g(self):\n"
        "\n"
        '        return "a real string"\n'  # a used string value is not prose
    )
    assert len(stripped.splitlines()) == len(source.splitlines())


def test_strip_prose_catches_attribute_docstrings():
    """PEP 258 prose after an assignment is not an AST docstring, but is prose."""
    source = (
        "LIMIT = 3\n"
        '"""How many hops a chain may stall for."""\n'
        "NAME = str(LIMIT)\n"
        'HELP = "a value, not prose"\n'
        'f"{LIMIT} interpolated"\n'
    )
    assert strip_prose_strings(source) == (
        "LIMIT = 3\n"
        "\n"
        "NAME = str(LIMIT)\n"
        'HELP = "a value, not prose"\n'
        'f"{LIMIT} interpolated"\n'  # an f-string can run code; never touched
    )


def test_strip_prose_keeps_an_all_prose_block_legal():
    source = "if x:\n    'only'\n    'prose'\nelse:\n    y = 1\n"
    stripped = strip_prose_strings(source)
    assert stripped == "if x:\n    pass\n\nelse:\n    y = 1\n"
    compile(stripped, "<t>", "exec")


def test_strip_prose_leaves_a_shared_line_alone():
    """Blanking in place cannot fix these without moving line numbers."""
    for source in ('def f(): "doc"\n', 'def f():\n    "doc"; x = 1\n    return x\n'):
        assert strip_prose_strings(source) == source


def test_strip_prose_rejects_unparseable_source():
    with pytest.raises(BundleError, match="cannot parse"):
        strip_prose_strings("def f(\n", "broken.py")


def test_strip_comments_survives_a_comment_inside_a_continuation():
    source = "x = (1 +  # add\n     2)\n"
    assert strip_comments(source) == "x = (1 +\n     2)\n"
    scope: dict = {}
    exec(compile(strip_comments(source), "<t>", "exec"), scope)
    assert scope["x"] == 3


def test_bundle_ships_python_without_prose_and_leaves_the_repo_intact(tmp_path):
    agent_src = BOTS_DIR / "smoke" / "agent.py"
    before = agent_src.read_bytes()
    assert b'"""' in before, "the fixture must have prose to strip"

    info = write_bundle("smoke", tmp_path / "smoke.zip")
    with zipfile.ZipFile(info.zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".py"):
                continue
            text = zf.read(name).decode("utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if lineno == 1 and line.startswith("#!"):
                    continue
                assert "#" not in _outside_strings(line), f"{name}:{lineno}: {line}"
            tree = ast.parse(text)  # the stripped member is still valid Python
            for node in ast.walk(tree):
                for field in ("body", "orelse", "finalbody"):
                    stmts = getattr(node, field, None)  # Lambda.body is an expression
                    if not isinstance(stmts, list):
                        continue
                    for stmt in stmts:
                        assert not _is_prose(stmt), f"{name}:{stmt.lineno}: prose survived"

    assert agent_src.read_bytes() == before, "bundling must not rewrite repo sources"


def _outside_strings(line: str) -> str:
    """The part of a line with quoted spans blanked out, for a crude `#` scan."""
    out, quote = [], ""
    for ch in line:
        if quote:
            out.append(" ")
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def test_keep_comments_ships_sources_verbatim(tmp_path):
    info = write_bundle("smoke", tmp_path / "verbatim.zip", strip_comments_in_zip=False)
    with zipfile.ZipFile(info.zip_path) as zf:
        assert zf.read("bots/smoke/agent.py") == (BOTS_DIR / "smoke" / "agent.py").read_bytes()


def test_a_probe_beside_the_agent_stays_out_of_the_bundle():
    """metro has a probe on disk; the bundle must not notice."""
    assert (BOTS_DIR / "metro" / "probe.py").is_file()
    names = {arcname for _, arcname in bundle_members("metro")}
    assert "bots/metro/probe.py" not in names
