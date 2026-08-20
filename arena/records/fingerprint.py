"""
Content hash of a bot's source closure, for rating identity.

A repo HEAD pin (`store.git_commit_or_tag`) moves whenever *anything* in the
repo is committed and stays put when a bot is edited without committing, which
makes it useless for deciding whether two stored games were played by the same
program. Game records dropped it at schema v4; this hash is the rating
identity, and `arena/records/registry.py` is what maps it back to source.

The hash here covers exactly the files a bot's behaviour depends on: every file
in its own directory, plus every module under `bots/` it imports, transitively.
That closure can cross bot directories: a bot that imports another bot's
`agent` module takes that bot's source into its own hash. No bot in `bots/`
does so today — proteus, which dispatched to four of them, was removed on
2026-08-18. Every bot pulls in `_common/wire.py` via its `main.py`; the
heuristics also pull in `_common/strategy_common.py` and friends.

Imports that do not resolve under `bots/` (stdlib, `jax`, `generals`) are
outside the closure: pinning third-party versions is the lockfile's job.

Resolution mirrors how a bot actually runs — `bots/<id>/main.py` puts `bots/`
on `sys.path` and imports a bare `agent`, so bare names resolve against the bot
directory first, then against `bots/`.

One name is excluded by rule: `probe.py`. A probe is arena-owned per-turn
introspection that only `arena.instrument.runner` ever loads (see
docs/arena/trajectories.md); it never plays, never ships in a bundle, and must
not fork a rating identity when it is edited. Bot-local `tests/` directories are
also excluded: unit fixtures sit beside source but do not play.

What keeps that honest is the invariant **unhashed code must be unreachable
from the hashed program**: if any module in the closure imports `probe`, the
walk raises rather than under-hashing a program that a probe can influence.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from arena.records.store import REPO_ROOT, bot_id_from_run_sh

BOTS_DIR = REPO_ROOT / "bots"

HASH_LENGTH = 12

# Directories and files that are build output or non-play code, not source.
# `tests/` sits beside bot source for unit fixtures; it must not fork the
# content hash (same rule as probe.py — unhashed code stays unreachable).
_SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "tests",
    # Compiled bots (morpheus-rs) keep build output and vendored crates inside
    # the bot directory. Neither is source: `target/` is what the sources
    # compile *to*, and `vendor/` is a copy of crates.io that `Cargo.lock`
    # already pins exactly. Hashing either would fork the rating identity on
    # every build and put ten thousand vendored files behind one bot's hash.
    "target",
    "vendor",
    # Developer tooling that ships nowhere and never plays: artifact
    # converters, submission packagers, the capture module the arena loads for
    # a corpus run. Same rule as `probe.py` — arena-owned introspection and
    # build scripts must not fork a rating identity when they are edited. Note
    # this is a *directory* rule, so a bot that put play code under `tools/`
    # and imported it would go unhashed; nothing does, and the closure walk
    # below would not resolve such an import to a hashed file anyway.
    "tools",
}
_SKIP_SUFFIXES = {".pyc", ".pyo"}
# Filesystem-browser droppings, not source: a Finder visit writes .DS_Store
# into the bot directory and would fork the rating identity without one
# byte of play code moving.
_SKIP_FILES = {".DS_Store", "Thumbs.db"}

# Per-turn introspection, loaded only by `arena.instrument.runner`. Outside the
# closure, so probe edits move no hash and no bundle.
PROBE_FILENAME = "probe.py"
PROBE_MODULE = "probe"

# Shell launchers source siblings by path (`bots/cm_*/run.sh` -> `_common/cm_run.sh`),
# which no AST walk would find. Best-effort: pick paths that look like bots/ files.
_SHELL_REF_RE = re.compile(r"[\w./-]*?([\w-]+/[\w.-]+\.(?:sh|py))")


def _is_source(path: Path) -> bool:
    if path.suffix in _SKIP_SUFFIXES:
        return False
    if path.name == PROBE_FILENAME or path.name in _SKIP_FILES:
        return False
    return not any(part in _SKIP_DIRS for part in path.parts)


def _bot_dir_files(bot_dir: Path) -> list[Path]:
    return sorted(p for p in bot_dir.rglob("*") if p.is_file() and _is_source(p))


def _module_file(module: str, search_dirs: list[Path]) -> Path | None:
    """Resolve a dotted module name to a file under one of `search_dirs`."""
    parts = module.split(".")
    if not all(parts):
        return None
    for base in search_dirs:
        candidate = base.joinpath(*parts)
        module_py = candidate.parent / f"{candidate.name}.py"
        if module_py.is_file():
            return module_py
        package_init = candidate / "__init__.py"
        if package_init.is_file():
            return package_init
    return None


def _package_inits(module: str, search_dirs: list[Path]) -> list[Path]:
    """`__init__.py` of every package on the way to `module` — they run on import."""
    parts = module.split(".")[:-1]
    found: list[Path] = []
    for base in search_dirs:
        for depth in range(1, len(parts) + 1):
            init = base.joinpath(*parts[:depth]) / "__init__.py"
            if init.is_file():
                found.append(init)
    return found


def _imported_names(source: str, path: Path) -> set[str]:
    """Dotted module names a Python file imports, absolute form only."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # No bot uses relative imports today; resolve anyway rather than
                # leaving a silent hole in the closure.
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                try:
                    prefix = base.relative_to(BOTS_DIR).as_posix().replace("/", ".")
                except ValueError:
                    continue
                head = f"{prefix}.{node.module}" if node.module else prefix
            elif node.module:
                head = node.module
            else:
                continue
            names.add(head)
            # `from pkg import submodule` — the alias may itself be a module.
            for alias in node.names:
                names.add(f"{head}.{alias.name}")
    return names


def _shell_referenced(source: str) -> set[str]:
    return {match.group(1) for match in _SHELL_REF_RE.finditer(source)}


class ProbeInClosureError(RuntimeError):
    """
    A hashed module imports a `probe.py`, which is not hashed.

    That would make the rating identity blind to code the program can execute:
    editing the probe would change how the bot plays without moving its hash.
    A loud failure here is the price of keeping probes out of the closure.
    """


def bot_source_closure(bot_dir: Path) -> list[Path]:
    """
    Every source file the bot's behaviour depends on, sorted and deduplicated.

    Seeded with the bot's own directory, then expanded over `bots/` imports
    (and shell `source` references) until it stops growing. `probe.py` is
    excluded by rule and may not be imported from inside the closure.
    """
    bot_dir = bot_dir.resolve()
    search_dirs = [bot_dir, BOTS_DIR]

    seen: set[Path] = set()
    queue = _bot_dir_files(bot_dir)

    while queue:
        path = queue.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary asset: hashed, but nothing to walk

        if path.suffix == ".py":
            for module in _imported_names(source, path):
                resolved = _module_file(module, search_dirs)
                if resolved is not None and resolved.name == PROBE_FILENAME:
                    raise ProbeInClosureError(
                        f"{relative_label(path)} imports {module!r}, which resolves "
                        f"to the unhashed {relative_label(resolved)}; a probe must "
                        f"never be reachable from the program it observes"
                    )
                if resolved is not None and resolved not in seen:
                    queue.append(resolved)
                for init in _package_inits(module, search_dirs):
                    if init not in seen:
                        queue.append(init)
        elif path.suffix == ".sh":
            for ref in _shell_referenced(source):
                candidate = BOTS_DIR / ref
                if candidate.is_file() and candidate not in seen:
                    queue.append(candidate)

    return sorted(seen)


def relative_label(path: Path) -> str:
    """Machine-independent path label so hashes match across checkouts."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def content_hash_for_dir(bot_dir: Path) -> str:
    """Short hex digest over the bot's source closure."""
    digest = hashlib.sha256()
    for path in bot_source_closure(bot_dir):
        digest.update(relative_label(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:HASH_LENGTH]


class UnhashableBotError(RuntimeError):
    """A bot's source closure could not be read, so it has no rating identity."""


def bot_content_hash(run_sh: Path) -> str:
    """
    Content hash for the bot owning `run_sh`.

    Raises rather than returning a sentinel: under hash-keyed rating identity a
    `"unknown"` hash silently pools unrelated programs into one rated entity.

    Deliberately **not** memoized. The previous per-process `lru_cache` keyed on
    the directory went stale the moment a bot was edited under a long-lived pool
    worker, which then labelled every later match with the pre-edit hash. A
    fresh closure walk costs ~9 ms against a ~3 s match; a wrong identity costs
    the experiment. Callers that hash a whole roster do it once, up front.
    """
    try:
        bot_dir = run_sh.resolve().parent
        if not bot_dir.is_dir():
            raise UnhashableBotError(f"no bot directory beside {run_sh}")
        return content_hash_for_dir(bot_dir)
    except OSError as exc:
        raise UnhashableBotError(f"cannot read the source closure of {run_sh}: {exc}") from exc


def bot_content_hashes(run_scripts: list[Path]) -> dict[str, str]:
    """Map bot id -> content hash, for callers that hash a roster up front."""
    return {bot_id_from_run_sh(p): bot_content_hash(p) for p in run_scripts}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the content hash and source closure of one or more bots."
    )
    parser.add_argument(
        "bots",
        nargs="*",
        help="bot ids (default: every directory under bots/ with a run.sh)",
    )
    parser.add_argument(
        "--files",
        action="store_true",
        help="also list the source closure behind each hash",
    )
    args = parser.parse_args(argv)

    if args.bots:
        run_scripts = [BOTS_DIR / name / "run.sh" for name in args.bots]
    else:
        run_scripts = sorted(BOTS_DIR.glob("*/run.sh"))

    for run_sh in run_scripts:
        if not run_sh.exists():
            print(f"{run_sh.parent.name:<18} <no run.sh at {run_sh}>")
            continue
        bot_id = bot_id_from_run_sh(run_sh)
        closure = bot_source_closure(run_sh.resolve().parent)
        print(f"{bot_id:<18} {bot_content_hash(run_sh)}  ({len(closure)} file(s))")
        if args.files:
            for path in closure:
                print(f"    {relative_label(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
