"""
Bot version registry: `data/bot_versions/<bot_id>.json`, committed to git.

A content hash (`arena/records/fingerprint.py`) names the exact program that
played a game. The registry is what turns that opaque digest back into
something a human can read: the file closure behind it, the commit that was
checked out when it ran, and — via `refs/bot-versions/<hash>` — a real git tree
you can `git diff` even when the closure was never committed.

Two structures, deliberately different shapes (see the plan's C1):

- `versions` is a **set**, unique by content hash, append-only, never mutated.
  This is what the rating fit keys on. A reverted hash is the same program, so
  its games pool into one rated entity.
- `steps` is an ordered **sequence** that may repeat a hash. This is the
  lineage: what ran, in the order it ran. A revert appends a third step
  pointing at the first hash.

Lineage order comes from `steps` — an append-only log written when a program
actually ran — never from `finished_at` and never from git. That keeps it
independent of everything the fit's order-independence guarantee cares about.

Registration happens in the **parent** process at hash time (the tournament
runner hashes its roster once before the pool starts). Pool workers never
write; they call `require_registered` and fail the match if a hash is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from arena.paths import REPO_ROOT
from arena.records import fingerprint
from arena.records.store import utc_now_iso

REGISTRY_DIR = REPO_ROOT / "data" / "bot_versions"

# One ref per content hash. `git diff refs/bot-versions/<a> refs/bot-versions/<b>`
# then works for hashes that live in no commit.
REF_PREFIX = "refs/bot-versions/"

# Fixed identity and date for closure commits, so re-anchoring the same closure
# on another machine yields the same commit object instead of a new one.
_ANCHOR_ENV = {
    "GIT_AUTHOR_NAME": "arena-registry",
    "GIT_AUTHOR_EMAIL": "arena-registry@localhost",
    "GIT_AUTHOR_DATE": "1970-01-01T00:00:00+0000",
    "GIT_COMMITTER_NAME": "arena-registry",
    "GIT_COMMITTER_EMAIL": "arena-registry@localhost",
    "GIT_COMMITTER_DATE": "1970-01-01T00:00:00+0000",
}

FILE_VERSION = 1


class RegistryError(RuntimeError):
    """Registration or lookup failed in a way the caller must not paper over."""


@dataclass(frozen=True)
class ClosureFile:
    """One file of a bot's source closure, by both hash schemes."""

    path: str  # repo-relative, POSIX
    sha256: str  # what the content hash is built from
    blob: str | None  # git blob SHA, None when git is unavailable

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "blob": self.blob}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClosureFile:
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            blob=None if data.get("blob") is None else str(data["blob"]),
        )


@dataclass(frozen=True)
class BotVersion:
    """One rated program: a content hash plus the evidence behind it."""

    content_hash: str
    first_seen_at: str
    git_commit: str | None
    git_dirty: bool
    closure_ref: str | None
    closure_tree: str | None
    files: tuple[ClosureFile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "first_seen_at": self.first_seen_at,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "closure_ref": self.closure_ref,
            "closure_tree": self.closure_tree,
            "files": [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BotVersion:
        return cls(
            content_hash=str(data["content_hash"]),
            first_seen_at=str(data["first_seen_at"]),
            git_commit=None if data.get("git_commit") is None else str(data["git_commit"]),
            git_dirty=bool(data.get("git_dirty", False)),
            closure_ref=None if data.get("closure_ref") is None else str(data["closure_ref"]),
            closure_tree=None if data.get("closure_tree") is None else str(data["closure_tree"]),
            files=tuple(ClosureFile.from_dict(f) for f in data.get("files", [])),
        )

    def paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files)


@dataclass(frozen=True)
class Step:
    """One position in a bot's lineage. May repeat a `content_hash`."""

    seq: int
    content_hash: str
    at: str
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "seq": self.seq,
            "content_hash": self.content_hash,
            "at": self.at,
        }
        if self.note is not None:
            data["note"] = self.note
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        note = data.get("note")
        return cls(
            seq=int(data["seq"]),
            content_hash=str(data["content_hash"]),
            at=str(data["at"]),
            note=None if note is None else str(note),
        )


@dataclass
class BotRegistryFile:
    """The contents of one `data/bot_versions/<bot_id>.json`."""

    bot_id: str
    versions: list[BotVersion]
    steps: list[Step]
    version: int = FILE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "bot_id": self.bot_id,
            "versions": [v.to_dict() for v in self.versions],
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BotRegistryFile:
        return cls(
            bot_id=str(data["bot_id"]),
            versions=[BotVersion.from_dict(v) for v in data.get("versions", [])],
            steps=[Step.from_dict(s) for s in data.get("steps", [])],
            version=int(data.get("version", FILE_VERSION)),
        )

    def find(self, content_hash: str) -> BotVersion | None:
        for entry in self.versions:
            if entry.content_hash == content_hash:
                return entry
        return None


# --- git plumbing -----------------------------------------------------------


def _git(
    args: list[str],
    *,
    repo_root: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
    stdin: str | None = None,
) -> str:
    cmd = ["git", "-C", str(repo_root), *args]
    full_env = None
    if env:
        full_env = {**os.environ, **env}
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=full_env, input=stdin, check=False
    )
    if check and result.returncode != 0:
        raise RegistryError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_available(repo_root: Path) -> bool:
    try:
        return bool(_git(["rev-parse", "--git-dir"], repo_root=repo_root, check=False))
    except OSError:
        return False


def _head_commit(repo_root: Path) -> str | None:
    try:
        return _git(["rev-parse", "HEAD"], repo_root=repo_root, check=False) or None
    except OSError:
        return None


def _closure_is_dirty(repo_root: Path, paths: Iterable[str]) -> bool:
    """
    True when any closure file differs from HEAD (or is untracked).

    Scoped to the closure rather than the whole tree: an unrelated edit
    elsewhere in the repo does not make *this* bot's `git_commit` a lie.
    """
    rel = list(paths)
    if not rel:
        return False
    try:
        status = _git(
            ["status", "--porcelain", "--", *rel], repo_root=repo_root, check=False
        )
    except OSError:
        return False
    return bool(status.strip())


def _blob_mode(path: Path) -> str:
    return "100755" if os.access(path, os.X_OK) else "100644"


def _anchor_closure(
    repo_root: Path,
    content_hash: str,
    entries: list[tuple[str, Path]],
) -> tuple[str | None, str | None, dict[str, str]]:
    """
    Write the closure into git's object database under `refs/bot-versions/<hash>`.

    Returns `(ref, tree_sha, {path: blob_sha})`. The ref keeps the objects
    reachable across `git gc` and survives the rebase-onto-main workflow, which
    rewrites the commits recorded in `git_commit` but leaves these independent
    commit objects alone.
    """
    # Batched through stdin: one subprocess per stage rather than per file, so
    # the cost does not grow with closure size (morpheus pulls in 20+ files).
    written = _git(
        ["hash-object", "-w", "--stdin-paths"],
        repo_root=repo_root,
        stdin="".join(f"{path}\n" for _, path in entries),
    ).splitlines()
    if len(written) != len(entries):
        raise RegistryError("git hash-object returned the wrong number of blobs")
    blobs = {label: sha for (label, _), sha in zip(entries, written)}

    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git(
            ["update-index", "--add", "--index-info"],
            repo_root=repo_root,
            env=env,
            stdin="".join(
                f"{_blob_mode(path)} {blobs[label]}\t{label}\n" for label, path in entries
            ),
        )
        tree = _git(["write-tree"], repo_root=repo_root, env=env)

    commit = _git(
        ["commit-tree", tree, "-m", f"bot-version {content_hash}"],
        repo_root=repo_root,
        env=_ANCHOR_ENV,
    )
    ref = f"{REF_PREFIX}{content_hash}"
    _git(["update-ref", ref, commit], repo_root=repo_root)
    return ref, tree, blobs


# --- registry ---------------------------------------------------------------


class Registry:
    """Read/write access to `data/bot_versions/`, one JSON file per bot."""

    def __init__(self, directory: Path | None = None, *, repo_root: Path | None = None) -> None:
        self.directory = directory or REGISTRY_DIR
        self._repo_root = repo_root
        self._cache: dict[str, BotRegistryFile | None] = {}

    @property
    def repo_root(self) -> Path:
        # Resolved late so tests that monkeypatch `fingerprint.REPO_ROOT` onto a
        # sandbox repo get the sandbox, not the real one.
        return self._repo_root or fingerprint.REPO_ROOT

    def path_for(self, bot_id: str) -> Path:
        return self.directory / f"{bot_id}.json"

    def bot_ids(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json"))

    def load(self, bot_id: str) -> BotRegistryFile | None:
        if bot_id not in self._cache:
            path = self.path_for(bot_id)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self._cache[bot_id] = BotRegistryFile.from_dict(data)
            else:
                self._cache[bot_id] = None
        return self._cache[bot_id]

    def reload(self) -> None:
        self._cache.clear()

    def version(self, bot_id: str, content_hash: str) -> BotVersion | None:
        entry = self.load(bot_id)
        return None if entry is None else entry.find(content_hash)

    def is_registered(self, bot_id: str, content_hash: str) -> bool:
        return self.version(bot_id, content_hash) is not None

    def require_registered(self, bot_id: str, content_hash: str) -> BotVersion:
        """Assert-only lookup for pool workers, which must never write."""
        found = self.version(bot_id, content_hash)
        if found is None:
            raise RegistryError(
                f"{bot_id}@{content_hash} is not registered in {self.directory}; "
                f"register it in the parent process before running matches "
                f"(python -m arena.records.registry --register {bot_id})"
            )
        return found

    def find_hash(self, content_hash: str) -> tuple[str, BotVersion] | None:
        """Locate a hash across every bot, for CLI lookups that omit the bot id."""
        for bot_id in self.bot_ids():
            found = self.version(bot_id, content_hash)
            if found is not None:
                return bot_id, found
        return None

    def save(self, entry: BotRegistryFile) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(entry.bot_id)
        path.write_text(
            json.dumps(entry.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._cache[entry.bot_id] = entry
        return path

    # -- registration --

    def register(
        self,
        bot_dir: Path,
        *,
        bot_id: str | None = None,
        strict: bool = False,
        note: str | None = None,
        now: str | None = None,
    ) -> tuple[BotVersion, bool]:
        """
        Register the current content hash of `bot_dir`. Idempotent.

        Returns `(version, is_new_hash)`. An existing hash is never mutated; a
        step is appended only when the hash differs from the current lineage
        head, so re-running the same roster does not grow `steps`.

        `strict=True` refuses a closure that differs from HEAD — the right
        default for a published round, where `git_commit` must actually name
        the bytes that ran.
        """
        bot_dir = bot_dir.resolve()
        if not bot_dir.is_dir():
            raise RegistryError(f"not a bot directory: {bot_dir}")
        name = bot_id or bot_dir.name
        content_hash = fingerprint.content_hash_for_dir(bot_dir)
        stamp = now or utc_now_iso()

        entry = self.load(name) or BotRegistryFile(bot_id=name, versions=[], steps=[])
        existing = entry.find(content_hash)

        if existing is None:
            closure = fingerprint.bot_source_closure(bot_dir)
            labelled = [(fingerprint.relative_label(p), p) for p in closure]
            dirty = False
            head = None
            ref = tree = None
            blobs: dict[str, str] = {}
            if _git_available(self.repo_root):
                head = _head_commit(self.repo_root)
                dirty = _closure_is_dirty(self.repo_root, (label for label, _ in labelled))
                if strict and dirty:
                    raise RegistryError(
                        f"{name}@{content_hash}: closure differs from HEAD and "
                        f"--strict is set; commit the bot first"
                    )
                ref, tree, blobs = _anchor_closure(self.repo_root, content_hash, labelled)
            elif strict:
                raise RegistryError(
                    f"{name}@{content_hash}: --strict needs git, which is unavailable"
                )
            if dirty:
                print(
                    f"[registry] {name}@{content_hash}: registered from a dirty tree; "
                    f"git_commit={head} does not describe these bytes "
                    f"(closure anchored at {ref})"
                )
            existing = BotVersion(
                content_hash=content_hash,
                first_seen_at=stamp,
                git_commit=head,
                git_dirty=dirty,
                closure_ref=ref,
                closure_tree=tree,
                files=tuple(
                    ClosureFile(
                        path=label,
                        sha256=_sha256_file(path),
                        blob=blobs.get(label),
                    )
                    for label, path in labelled
                ),
            )
            entry.versions.append(existing)
            is_new = True
        else:
            is_new = False

        if not entry.steps or entry.steps[-1].content_hash != content_hash:
            seq = len(entry.steps) + 1
            step_note = note
            if step_note is None and not is_new:
                prior = next(
                    (s.seq for s in entry.steps if s.content_hash == content_hash), None
                )
                if prior is not None:
                    step_note = f"revert to seq {prior}"
            entry.steps.append(
                Step(seq=seq, content_hash=content_hash, at=stamp, note=step_note)
            )

        self.save(entry)
        return existing, is_new

    def register_run_scripts(
        self, run_scripts: Iterable[Path], *, strict: bool = False
    ) -> dict[str, str]:
        """
        Register a whole roster once, returning `{bot_id: content_hash}`.

        Called by the tournament runner in the parent process, before the pool
        starts, so the roster is hashed and registered exactly once per round.
        """
        hashes: dict[str, str] = {}
        for run_sh in run_scripts:
            bot_dir = Path(run_sh).resolve().parent
            if not is_registerable(bot_dir):
                raise RegistryError(
                    f"{bot_dir} is not under {fingerprint.BOTS_DIR}, so its source "
                    f"closure cannot be resolved and it has no rating identity; "
                    f"drop it from the roster (see the plan's q1)"
                )
            entry, _ = self.register(bot_dir, strict=strict)
            hashes[bot_dir.name] = entry.content_hash
        return hashes

    # -- verification --

    def verify(self) -> list[dict[str, str]]:
        """
        Report problems without ever touching a rating.

        `git-unresolvable` is informational: a hash whose diff cannot be
        resolved is still a perfectly valid rated entity, because its identity
        comes from content, not from git.
        """
        issues: list[dict[str, str]] = []
        have_git = _git_available(self.repo_root)
        for bot_id in self.bot_ids():
            entry = self.load(bot_id)
            if entry is None:
                continue
            if entry.bot_id != bot_id:
                issues.append(
                    {
                        "kind": "bot-id-mismatch",
                        "bot_id": bot_id,
                        "detail": f"file declares bot_id={entry.bot_id!r}",
                    }
                )
            seen: set[str] = set()
            for version in entry.versions:
                if version.content_hash in seen:
                    issues.append(
                        {
                            "kind": "duplicate-version",
                            "bot_id": bot_id,
                            "detail": version.content_hash,
                        }
                    )
                seen.add(version.content_hash)
                if not have_git:
                    continue
                if version.closure_ref is None or not _ref_exists(
                    self.repo_root, version.closure_ref
                ):
                    issues.append(
                        {
                            "kind": "git-unresolvable",
                            "bot_id": bot_id,
                            "detail": f"{version.content_hash}: closure_ref missing",
                        }
                    )
                if version.git_commit and not _commit_exists(
                    self.repo_root, version.git_commit
                ):
                    issues.append(
                        {
                            "kind": "git-unresolvable",
                            "bot_id": bot_id,
                            "detail": f"{version.content_hash}: git_commit "
                            f"{version.git_commit[:12]} unreachable",
                        }
                    )
            for i, step in enumerate(entry.steps, start=1):
                if step.seq != i:
                    issues.append(
                        {
                            "kind": "step-out-of-order",
                            "bot_id": bot_id,
                            "detail": f"position {i} carries seq {step.seq}",
                        }
                    )
                if step.content_hash not in seen:
                    issues.append(
                        {
                            "kind": "step-unknown-hash",
                            "bot_id": bot_id,
                            "detail": f"seq {step.seq}: {step.content_hash}",
                        }
                    )
        return issues


def is_registerable(bot_dir: Path) -> bool:
    """
    True for bots under `bots/`, whose closure `fingerprint` can resolve.

    `competition-module`'s own `expander_python` lives outside `bots/`, so its
    imports do not resolve and its hash names nothing useful. It stays runnable
    ad hoc; it just never becomes a rated entity — `bots/cm_expander/` is the
    in-repo wrapper over the same upstream agent, and it is the anchor.
    """
    try:
        bot_dir.resolve().relative_to(fingerprint.BOTS_DIR.resolve())
    except (ValueError, OSError):
        return False
    return True


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref_exists(repo_root: Path, ref: str) -> bool:
    return bool(_git(["rev-parse", "--verify", "--quiet", ref], repo_root=repo_root, check=False))


def _commit_exists(repo_root: Path, commit: str) -> bool:
    return bool(
        _git(
            ["rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
            repo_root=repo_root,
            check=False,
        )
    )


# Structural problems are a real defect; a hash git cannot resolve is not.
_INFORMATIONAL_ISSUES = {"git-unresolvable"}


def closure_delta(a: BotVersion, b: BotVersion) -> dict[str, list[str]]:
    """Which closure files were added, removed, or changed between two hashes."""
    left = {f.path: f.sha256 for f in a.files}
    right = {f.path: f.sha256 for f in b.files}
    return {
        "added": sorted(set(right) - set(left)),
        "removed": sorted(set(left) - set(right)),
        "changed": sorted(p for p in set(left) & set(right) if left[p] != right[p]),
    }


# --- CLI --------------------------------------------------------------------


def _resolve(registry: Registry, content_hash: str) -> tuple[str, BotVersion]:
    found = registry.find_hash(content_hash)
    if found is None:
        raise SystemExit(f"[registry] unknown content hash: {content_hash}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register, verify, and diff bot versions in data/bot_versions/."
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=REGISTRY_DIR,
        help=f"registry directory (default: {REGISTRY_DIR})",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--register",
        nargs="*",
        metavar="BOT",
        help="register bot ids (default: every bots/*/run.sh)",
    )
    group.add_argument(
        "--verify", action="store_true", help="report structural and git problems"
    )
    group.add_argument(
        "--diff", nargs=2, metavar=("HASH_A", "HASH_B"), help="closure + git diff"
    )
    group.add_argument(
        "--files", nargs="+", metavar="HASH", help="print the closure paths of hashes"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="refuse to register a closure that differs from HEAD",
    )
    args = parser.parse_args(argv)

    registry = Registry(args.registry_dir)

    if args.register is not None:
        names = args.register or sorted(
            p.parent.name for p in fingerprint.BOTS_DIR.glob("*/run.sh")
        )
        for name in names:
            bot_dir = fingerprint.BOTS_DIR / name
            try:
                version, is_new = registry.register(bot_dir, strict=args.strict)
            except RegistryError as exc:
                print(f"[registry] {name}: {exc}")
                return 1
            state = "registered" if is_new else "already registered"
            print(f"[registry] {name:<18} {version.content_hash}  {state}")
        return 0

    if args.verify:
        issues = registry.verify()
        if not issues:
            print(f"[registry] {len(registry.bot_ids())} bot file(s), no issues")
            return 0
        for issue in issues:
            print(f"[registry] {issue['kind']:<20} {issue['bot_id']:<18} {issue['detail']}")
        blocking = [i for i in issues if i["kind"] not in _INFORMATIONAL_ISSUES]
        return 1 if blocking else 0

    if args.diff:
        hash_a, hash_b = args.diff
        bot_a, version_a = _resolve(registry, hash_a)
        bot_b, version_b = _resolve(registry, hash_b)
        delta = closure_delta(version_a, version_b)
        print(f"[registry] {bot_a}@{hash_a} -> {bot_b}@{hash_b}")
        for kind in ("added", "removed", "changed"):
            for path in delta[kind]:
                print(f"    {kind:<8} {path}")
        if version_a.closure_ref and version_b.closure_ref:
            print()
            print(
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(registry.repo_root),
                        "diff",
                        version_a.closure_ref,
                        version_b.closure_ref,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout,
                end="",
            )
        else:
            print("    (no closure refs; use the recorded git_commit + --files instead)")
        return 0

    paths: list[str] = []
    for content_hash in args.files:
        _, version = _resolve(registry, content_hash)
        paths.extend(version.paths())
    for path in sorted(set(paths)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
