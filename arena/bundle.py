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
program rated as X. That closure already drops `probe.py` and bot-local
`tests/`, so neither reaches the zip. cm_* bots are refused: they depend on
the competition-module submodule, which we do not ship.

Python members are minified **on the way into the zip only**; the repo files
are never touched. python-minifier does the rewrite: comments, docstrings,
annotations, and blank lines go, and locals are renamed, so a submitted bot
carries its logic but not its strategy notes. `--no-minify` ships the sources
verbatim.

Two consequences worth knowing. Line numbers no longer correspond to the repo
source, so a judge traceback locates a fault in the minified file, not in
`bots/`; re-bundle with `--no-minify` when you need to read one. And nothing
under `bots/` may rely on `__doc__` or on local variable names surviving
(`getattr` by a name that minifies away) — neither does today.

Judge environment facts (https://www.generals.bot/docs): `run.sh` is the sole
entrypoint, executed from within the submission directory; zip <= 50 MB,
unpacked <= 512 MB and <= 10,000 files; CPython 3.12 with the packages pinned
in competition-module/competition/requirements.txt preinstalled; no network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


# Minifier settings. `rename_globals` stays off: the closure spans modules that
# import each other by name, and renaming across that boundary would break the
# imports. Everything else is on, so the zip carries no comments, no
# docstrings, and no local names worth reading.
_MINIFY_OPTIONS = dict(
    remove_literal_statements=True,   # docstrings and PEP 258 attribute prose
    rename_globals=False,
)


def minify_source(source: str, origin: str = "<source>") -> str:
    """
    Rewrite one Python source into its minified equivalent for the zip.

    Delegates to python-minifier (pinned in requirements-dev.txt), which does
    what three hand-rolled passes here used to do and more: comments,
    docstrings, annotations, and blank lines go, locals are renamed, literals
    are hoisted. Line numbers no longer correspond to the repo source — the
    trade the aggressive setting buys is that a submitted bot no longer ships
    its strategy in readable form.
    """
    try:
        import python_minifier
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise BundleError(
            "python-minifier is not installed; `pip install -r requirements-dev.txt` "
            "or bundle with --no-minify"
        ) from exc
    try:
        return python_minifier.minify(source, filename=origin, **_MINIFY_OPTIONS)
    except SyntaxError as exc:
        raise BundleError(f"cannot minify {origin}: {exc}") from exc


def _member_bytes(path: Path, arcname: str, *, minify: bool) -> bytes:
    """Bytes to zip for one closure file: verbatim, or minified."""
    data = path.read_bytes()
    if not minify or not arcname.endswith(".py"):
        return data
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError:
        return data  # not a UTF-8 source; ship it byte-for-byte
    return minify_source(source, arcname).encode("utf-8")


def write_bundle(
    bot_id: str,
    out_path: Path | None = None,
    *,
    extra_includes: list[str] | None = None,
    force: bool = False,
    minify: bool = True,
) -> BundleInfo:
    """
    Build, audit, and limit-check the submission zip for `bot_id`.

    `minify` rewrites the Python sources as they are written into the archive.
    Files on disk are read-only here — the repo copies are untouched.
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
            _write_entry(zf, arcname, _member_bytes(path, arcname, minify=minify))

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
    parser.add_argument("--no-minify", action="store_true",
                        help="ship the Python sources verbatim "
                             "(default: minify each member into the zip)")
    args = parser.parse_args(argv)

    try:
        info = write_bundle(
            args.bot,
            args.output,
            extra_includes=args.include,
            force=args.force,
            minify=not args.no_minify,
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
