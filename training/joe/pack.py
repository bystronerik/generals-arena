"""Pack the Joe training checkout into a tarball for R2 code delivery.

``git archive`` skips submodules, so this walks the working tree of
``competition-module/`` plus ``training/joe`` (the same set Modal mounts
with ``add_local_dir``). ``.git`` and ``training/joe/tests`` stay out.
Stdlib only — cheap-suite tests pack a fake tree, not this repo.
"""

import os
import subprocess
import tarfile
from pathlib import Path

# Relative to the repo root. Matches scripts/joe_modal_train.py mounts:
# competition-module (whole working tree), training/__init__.py, training/joe.
PACK_ROOTS = (
    "competition-module",
    "training/__init__.py",
    "training/joe",
)

SKIP_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
SKIP_FILE_NAMES = {".DS_Store", ".gitkeep"}


def _is_skipped_dir(name):
    if name in SKIP_DIR_NAMES:
        return True
    return name.endswith(".egg-info")


def excluded(rel):
    """True if this repo-relative path must not enter the tarball."""
    parts = Path(rel).parts
    if not parts:
        return True
    if any(p in SKIP_DIR_NAMES or p.endswith(".egg-info") for p in parts):
        return True
    if parts[-1] in SKIP_FILE_NAMES:
        return True
    if len(parts) >= 3 and parts[:3] == ("training", "joe", "tests"):
        return True
    return False


def iter_pack_files(repo_root):
    """Yield ``(absolute_path, repo-relative posix path)`` pairs."""
    repo_root = Path(repo_root)
    for rel in PACK_ROOTS:
        path = repo_root / rel
        if not path.exists():
            raise FileNotFoundError(f"pack root missing: {path}")
        if path.is_file():
            if not excluded(rel):
                yield path, Path(rel).as_posix()
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if not _is_skipped_dir(d)]
            rel_dir = os.path.relpath(dirpath, repo_root)
            if rel_dir == ".":
                rel_dir = ""
            # Drop training/joe/tests even if a walk started at training/joe.
            if Path(rel_dir).parts[:3] == ("training", "joe") and \
                    "tests" in dirnames:
                dirnames.remove("tests")
            for name in filenames:
                rel_file = (Path(rel_dir) / name).as_posix() if rel_dir \
                    else name
                if excluded(rel_file):
                    continue
                yield Path(dirpath) / name, rel_file


def repo_git_state(repo_root):
    """``(sha, dirty)`` for the outer repo. dirty includes untracked files."""
    repo_root = str(repo_root)
    sha = subprocess.check_output(
        ["git", "-C", repo_root, "rev-parse", "HEAD"], text=True).strip()
    porcelain = subprocess.check_output(
        ["git", "-C", repo_root, "status", "--porcelain"], text=True)
    return sha, bool(porcelain.strip())


def engine_sha(repo_root):
    """competition-module HEAD — pinned into the run the same way Modal does."""
    path = Path(repo_root) / "competition-module"
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _safe_members(tf, dest):
    dest = os.path.abspath(dest)
    for member in tf.getmembers():
        target = os.path.abspath(os.path.join(dest, member.name))
        if target != dest and not target.startswith(dest + os.sep):
            raise RuntimeError(f"unsafe tar member {member.name!r}")
        yield member


def pack_checkout(repo_root, dest_tar_gz, git_sha=None):
    """Write a gzip tarball of the training checkout. Returns a meta dict."""
    repo_root = Path(repo_root)
    dest_tar_gz = Path(dest_tar_gz)
    dest_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    files = list(iter_pack_files(repo_root))
    if not files:
        raise RuntimeError(f"nothing to pack under {repo_root}")
    with tarfile.open(dest_tar_gz, "w:gz") as tf:
        for abs_path, rel in files:
            tf.add(abs_path, arcname=rel, recursive=False)
    size = dest_tar_gz.stat().st_size
    sha = _sha256_path(dest_tar_gz)
    dirty = False
    if git_sha is None:
        git_sha, dirty = repo_git_state(repo_root)
    return {"path": str(dest_tar_gz), "git_sha": git_sha, "git_dirty": dirty,
            "size": size, "sha256": sha, "n_files": len(files)}


def extract_checkout(tar_gz, dest):
    """Unpack a tarball produced by ``pack_checkout`` into ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz, "r:gz") as tf:
        members = list(_safe_members(tf, dest))
        try:
            tf.extractall(dest, members=members, filter="data")
        except TypeError:
            tf.extractall(dest, members=members)
    return dest


def _sha256_path(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
