"""
Package one bot from ``bots/<name>/`` into a standalone submission zip for
generals.bot.

The zip mirrors the repo-root layout, so every path a bot resolves at runtime
(`main.py` finds ``bots/`` via ``Path(__file__).parent.parent``) keeps working
with the repo gone:

    run.sh              # generated: `exec python -u bots/<name>/main.py`
    bots/_common/...    # only the source closure
    bots/<name>/...     # without its run.sh — that launcher is repo harness

The root `run.sh` uses a relative path and bare `python`: the competition
server's validation wants that exact shape, and the docs guarantee the bot
"runs from within its submission directory". The per-bot `run.sh` is excluded
because the server's validation rejects bundles that ship it.

What goes in is exactly `arena.records.fingerprint.bot_source_closure` — the
file set behind the bot's rating identity — so the bundle of hash X is the
program rated as X. cm_* bots are refused: they depend on the
competition-module submodule, which we do not ship.

Python sources are stripped of prose — both `#` comments and docstrings — **on
the way into the zip only**; the repo files are never touched. Every line
number is preserved (stripped lines go blank rather than disappearing), so a
judge traceback still points at the right line of the original file.
`--keep-comments` disables the step. Nothing under `bots/` reads `__doc__`, so
dropping docstrings costs nothing there; a bot that starts to would need this
step made selective.

Judge environment facts (https://www.generals.bot/docs): `run.sh` is the sole
entrypoint, executed from within the submission directory; zip <= 50 MB,
unpacked <= 512 MB and <= 10,000 files; CPython 3.12 with the packages pinned
in competition-module/competition/requirements.txt preinstalled; no network.
"""

from __future__ import annotations

import ast
import io
import os
import shutil
import subprocess
import sys
import tokenize
import zipfile
from dataclasses import dataclass
from pathlib import Path

from arena.records.fingerprint import (
    BOTS_DIR,
    _imported_names,
    _module_file,
    bot_content_hash,
    bot_source_closure,
)
from arena.records.store import REPO_ROOT

# Hard judge limits ("zip ≤ 50 MB", "unpacked ≤ 512 MB and ≤ 10,000 files").
MAX_ZIP_BYTES = 50 * 2**20
MAX_UNPACKED_BYTES = 512 * 2**20
MAX_FILES = 10_000

# Import names of the packages preinstalled on the judge (docs #environment,
# mirrored in competition-module/competition/requirements.txt). Anything else
# a bot imports would need wheels in the zip, which we deliberately don't do.
JUDGE_PACKAGES = {
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "torch",
    "jax",
    "jaxlib",
    "numba",
    "networkx",
    "safetensors",
    "gymnasium",
}

# Files whose presence in a closure marks a competition-module dependency.
_CM_MARKERS = {"_common/cm_adapter.py", "_common/cm_run.sh"}

BUNDLES_DIR = REPO_ROOT / "data" / "bundles"

# Fixed timestamp so identical closures produce byte-identical zips.
_ZIP_DATE = (2020, 1, 1, 0, 0, 0)

_SMOKE_TIMEOUT_SECONDS = 30


def _path_with_python(path: str, workdir: Path) -> str:
    """
    PATH that resolves bare `python`, as the generated run.sh and the judge
    expect. macOS often ships only `python3`; shim it beside the extract dir
    rather than rewriting the bundle, so the smoke runs the submitted bytes.
    """
    if shutil.which("python", path=path):
        return path
    python3 = shutil.which("python3", path=path)
    if python3 is None:
        raise BundleError("neither `python` nor `python3` on PATH for the smoke run")
    shim_dir = workdir.with_name(workdir.name + "-bin")
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "python"
    shim.unlink(missing_ok=True)
    shim.symlink_to(python3)
    return f"{shim_dir}{os.pathsep}{path}"


class BundleError(RuntimeError):
    """The bot cannot be bundled, or the bundle failed validation."""


@dataclass
class BundleInfo:
    bot_id: str
    content_hash: str
    zip_path: Path
    zip_bytes: int
    unpacked_bytes: int
    file_count: int


def _root_run_sh(bot_id: str) -> str:
    return f"#!/usr/bin/env bash\nexec python -u bots/{bot_id}/main.py\n"


def bundle_members(bot_id: str, extra_includes: list[str] | None = None) -> list[tuple[Path, str]]:
    """(source path, arcname) pairs for the bot's closure, cm bots refused."""
    bot_dir = BOTS_DIR / bot_id
    if not (bot_dir / "run.sh").is_file():
        raise BundleError(f"no bot at bots/{bot_id}/ (missing run.sh)")

    members: list[tuple[Path, str]] = []
    for path in bot_source_closure(bot_dir):
        try:
            arcname = path.relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            raise BundleError(f"closure file outside the repo: {path}") from exc
        if arcname.removeprefix("bots/") in _CM_MARKERS:
            raise BundleError(
                f"{bot_id} depends on the competition-module submodule "
                "(closure contains _common/cm_adapter.py); cm bots are not bundleable"
            )
        if arcname == f"bots/{bot_id}/run.sh":
            continue  # repo harness launcher; the generated root run.sh replaces it
        members.append((path, arcname))

    for rel in extra_includes or []:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise BundleError(f"--include {rel}: no such file under the repo root")
        members.append((path, Path(rel).as_posix()))

    return sorted(members, key=lambda m: m[1])


def third_party_imports(files: list[Path]) -> set[str]:
    """Top-level imported packages that are neither stdlib nor under bots/."""
    found: set[str] = set()
    for path in files:
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        for module in _imported_names(source, path):
            top = module.split(".", 1)[0]
            if top in sys.stdlib_module_names:
                continue
            if _module_file(module, [path.parent, BOTS_DIR]) is not None:
                continue
            if _module_file(top, [path.parent, BOTS_DIR]) is not None:
                continue
            # Bot dirs are namespace packages (no __init__.py): a bare
            # directory under a search root still resolves locally.
            if any((base / top).is_dir() for base in (path.parent, BOTS_DIR)):
                continue
            found.add(top)
    return found


def audit_imports(files: list[Path]) -> None:
    """Fail when the closure imports a package the judge does not preinstall."""
    offenders = third_party_imports(files) - JUDGE_PACKAGES
    if offenders:
        raise BundleError(
            "closure imports packages not preinstalled on the judge: "
            + ", ".join(sorted(offenders))
        )


def check_limits(
    zip_path: Path,
    *,
    max_zip_bytes: int = MAX_ZIP_BYTES,
    max_unpacked_bytes: int = MAX_UNPACKED_BYTES,
    max_files: int = MAX_FILES,
) -> tuple[int, int, int]:
    """Enforce the judge's archive limits; returns (zip, unpacked, count)."""
    zip_bytes = zip_path.stat().st_size
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
    unpacked_bytes = sum(info.file_size for info in infos)
    file_count = len(infos)
    for actual, cap, label in (
        (zip_bytes, max_zip_bytes, "zip bytes"),
        (unpacked_bytes, max_unpacked_bytes, "unpacked bytes"),
        (file_count, max_files, "files"),
    ):
        if actual > cap:
            raise BundleError(f"bundle exceeds the judge limit: {actual} {label} > {cap}")
    return zip_bytes, unpacked_bytes, file_count


def strip_comments(source: str, origin: str = "<source>") -> str:
    """
    Drop `#` comments from Python source, preserving every line number.

    Token-based, so a `#` inside a string literal is untouched. Only the
    comment span is erased: what precedes it on the line stays, and the newline
    stays, so a comment-only line turns into a blank line and line numbers
    survive into judge tracebacks. A first-line shebang is kept: it is an exec
    directive. Docstrings are prose too, but they are string expressions rather
    than comment tokens — `strip_prose_strings` handles those.
    """
    try:
        comment_starts = {
            tok.start[0]: tok.start[1]
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == tokenize.COMMENT
            and not (tok.start[0] == 1 and tok.start[1] == 0 and tok.string.startswith("#!"))
        }
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        raise BundleError(f"cannot tokenize {origin}: {exc}") from exc

    out: list[str] = []
    for lineno, line in enumerate(source.splitlines(keepends=True), start=1):
        col = comment_starts.get(lineno)
        if col is not None:
            newline = line[len(line.rstrip("\r\n")) :]
            line = line[:col].rstrip() + newline
        out.append(line)
    return "".join(out)


def _is_prose(stmt: ast.stmt) -> bool:
    """A statement that is nothing but a string literal: evaluated, discarded."""
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _statement_lists(tree: ast.AST):
    """Every statement list in the tree, with the node that owns it."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                yield node, stmts


def strip_prose_strings(source: str, origin: str = "<source>") -> str:
    """
    Drop every bare string-literal statement, preserving each line number.

    That covers module/class/function docstrings *and* the PEP 258 attribute
    docstrings that trail an assignment — both are `Expr(Constant(str))`, a
    constant evaluated and thrown away, so removing either cannot change what
    the program does. String *values* the program uses are untouched, as are
    f-strings, which can run code.

    Spans are blanked in place. When a block's statements were all prose,
    `pass` takes the first one's place at the same column, so the block stays
    legal and nothing below it shifts line.

    Prose sharing a line with code (`def f(): "doc"`, or a trailing `; x = 1`)
    is left alone — blanking it there would either break the syntax or need a
    reflow that moves line numbers.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise BundleError(f"cannot parse {origin}: {exc}") from exc

    lines = source.splitlines(keepends=True)
    edits: dict[int, tuple[int, str]] = {}  # first line -> (column, replacement)
    blanked: set[int] = set()

    for owner, stmts in _statement_lists(tree):
        prose = [s for s in stmts if _is_prose(s)]
        # Emptying a block is a syntax error; an empty module is fine.
        needs_pass = len(prose) == len(stmts) and not isinstance(owner, ast.Module)
        for stmt in prose:
            start, end = stmt.lineno, stmt.end_lineno or stmt.lineno
            if lines[start - 1][: stmt.col_offset].strip():
                continue  # code precedes it on the line
            if lines[end - 1][stmt.end_col_offset or 0 :].strip():
                continue  # code follows it on the line
            edits[start] = (stmt.col_offset, "pass" if needs_pass else "")
            blanked.update(range(start + 1, end + 1))
            needs_pass = False  # at most one filler per block

    out: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        newline = line[len(line.rstrip("\r\n")) :]
        if lineno in edits:
            col, replacement = edits[lineno]
            line = (line[:col] + replacement).rstrip() + newline
        elif lineno in blanked:
            line = newline
        out.append(line)
    return "".join(out)


def _member_bytes(path: Path, arcname: str, *, strip: bool) -> bytes:
    """Bytes to zip for one closure file: verbatim, or prose-stripped."""
    data = path.read_bytes()
    if not strip or not arcname.endswith(".py"):
        return data
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError:
        return data  # not a UTF-8 source; ship it byte-for-byte
    return strip_comments(strip_prose_strings(source, arcname), arcname).encode("utf-8")


def write_bundle(
    bot_id: str,
    out_path: Path | None = None,
    *,
    extra_includes: list[str] | None = None,
    force: bool = False,
    strip_comments_in_zip: bool = True,
) -> BundleInfo:
    """
    Build, audit, and limit-check the submission zip for `bot_id`.

    `strip_comments_in_zip` removes comments and docstrings from the Python
    sources as they are written into the archive. Files on disk are read-only
    here — the repo copies keep their prose.
    """
    members = bundle_members(bot_id, extra_includes)
    audit_imports([path for path, _ in members])
    content_hash = bot_content_hash(BOTS_DIR / bot_id / "run.sh")

    if out_path is None:
        out_path = default_output_path(bot_id, content_hash)
    if out_path.exists() and not force:
        raise BundleError(f"{out_path} exists (pass --force to overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_entry(zf, "run.sh", _root_run_sh(bot_id).encode("ascii"))
        for path, arcname in members:
            _write_entry(zf, arcname, _member_bytes(path, arcname, strip=strip_comments_in_zip))

    zip_bytes, unpacked_bytes, file_count = check_limits(out_path)
    return BundleInfo(
        bot_id=bot_id,
        content_hash=content_hash,
        zip_path=out_path,
        zip_bytes=zip_bytes,
        unpacked_bytes=unpacked_bytes,
        file_count=file_count,
    )


def _write_entry(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=_ZIP_DATE)
    mode = 0o755 if arcname.endswith(".sh") else 0o644
    info.external_attr = mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, data)


def default_output_path(bot_id: str, content_hash: str | None = None) -> Path:
    if content_hash is None:
        content_hash = bot_content_hash(BOTS_DIR / bot_id / "run.sh")
    return BUNDLES_DIR / f"{bot_id}-{content_hash}.zip"


# One wire-protocol frame (see _common/wire.py): 5x5 board, our general with 10
# army at (0,0), the opponent's visible with 10 at (4,4), plains elsewhere.
_SMOKE_HANDSHAKE = "0 5 5\n"
_SMOKE_FRAME = (
    "10 1 10 1 10\n"
    "4 1 1 1 1\n" + "1 1 1 1 1\n" * 3 + "1 1 1 1 4\n"
    "1 0 0 0 0\n" + "0 0 0 0 0\n" * 3 + "0 0 0 0 2\n"
    "10 0 0 0 0\n" + "0 0 0 0 0\n" * 3 + "0 0 0 0 10\n"
)


def smoke_check(
    zip_path: Path, extract_dir: Path, *, reuse_extracted: bool = False
) -> tuple[int, int, int, int, int]:
    """
    Prove the bundle runs standalone: unpack outside the repo, execute its
    `run.sh` with a scrubbed environment, and require one well-formed action
    for one observation frame. Returns the action.

    `reuse_extracted` skips extraction when `extract_dir` is already populated.
    macOS assesses each freshly written script on first exec (~0.2 s per file);
    a caller that keys `extract_dir` on the bundle's content hash pays that
    only when the bundle actually changed. Bundles are deterministic, so a
    hash-matched extract dir holds byte-identical files.
    """
    if not (reuse_extracted and (extract_dir / "run.sh").is_file()):
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                target = Path(zf.extract(info, extract_dir))
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    target.chmod(mode)

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHON", "VIRTUAL_ENV")}
    env["PATH"] = _path_with_python(env.get("PATH", os.defpath), extract_dir)
    proc = subprocess.Popen(
        ["./run.sh"],
        cwd=extract_dir,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(_SMOKE_HANDSHAKE + _SMOKE_FRAME, timeout=_SMOKE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise BundleError(f"smoke run of {zip_path.name} timed out after {_SMOKE_TIMEOUT_SECONDS}s")

    lines = stdout.splitlines()
    if proc.returncode != 0 or len(lines) != 1:
        raise BundleError(
            f"smoke run of {zip_path.name} failed "
            f"(exit {proc.returncode}, {len(lines)} stdout line(s)): {stderr.strip()[-500:]}"
        )
    try:
        action = tuple(int(x) for x in lines[0].split())
    except ValueError:
        action = ()
    if len(action) != 5:
        raise BundleError(f"smoke run of {zip_path.name} replied {lines[0]!r}, want 5 ints")
    return action  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="Package one bot into a standalone generals.bot submission zip."
    )
    parser.add_argument("bot", help="bot id (a directory under bots/ with a run.sh)")
    parser.add_argument("-o", "--output", type=Path, help=f"output zip (default: {BUNDLES_DIR}/<bot>-<hash>.zip)")
    parser.add_argument("--include", action="append", default=[], metavar="PATH",
                        help="extra repo-root-relative file the import walk cannot see")
    parser.add_argument("--no-smoke", action="store_true", help="skip the standalone smoke run")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output zip")
    parser.add_argument("--keep-comments", action="store_true",
                        help="ship the Python sources verbatim "
                             "(default: strip comments and docstrings in the zip)")
    args = parser.parse_args(argv)

    try:
        info = write_bundle(
            args.bot,
            args.output,
            extra_includes=args.include,
            force=args.force,
            strip_comments_in_zip=not args.keep_comments,
        )
        if not args.no_smoke:
            with tempfile.TemporaryDirectory(prefix="bundle-smoke-") as tmp:
                action = smoke_check(info.zip_path, Path(tmp))
            print(f"smoke ok: action {' '.join(str(x) for x in action)}")
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{info.zip_path} ({info.file_count} files, "
        f"{info.zip_bytes} B zipped / {info.unpacked_bytes} B unpacked, hash {info.content_hash})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
