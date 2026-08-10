#!/usr/bin/env python3
"""Package morpheus-rs for generals.bot, and prove the archive before it ships.

    python bots/morpheus-rs/tools/package_submission.py --force
    python bots/morpheus-rs/tools/package_submission.py --force --gate
    python bots/morpheus-rs/tools/package_submission.py --no-minify

Started at milestone M0.5 of docs/bots/morpheus-rs/rewrite-plan.md. One archive
comes out — sources, `Cargo.lock`, vendored crates, and a `build.sh` running
`cargo build --release --offline` — and the judge compiles it at intake on its
own toolchain.

It is named `morpheus-rs-<content_hash>.zip` through
`arena.bundle.default_output_path`, the same call the Python bundles use, so an
archive says on its face which rated program it holds and two builds of
different code cannot land on one path.

**The static-binary variant is gone** (M8). It existed as R4's fallback for a
sandbox that cannot build the vendored tree, and §9 asked for it to be built
every run "so the fallback stays tested rather than theoretical". Two things
overtook that: it was never latency-qualified — M8 measured it 1.75x to 2.2x
slower per decision under musl, so falling back to it meant shipping an
unmeasured bot — and keeping a second archive alive cost a cross-linker path,
a second `build.sh`, and a smoke test that could not run on the packaging host
anyway. If R4 ever fires, the fallback has to be rebuilt from git history
(`git log -- bots/morpheus-rs/tools/package_submission.py`) and qualified
before it plays, which is the honest description of where it now stands.

The archive is audited against the judge's limits through
`arena.bundle.check_limits` — the same code path that guards the Python bots,
so the two cannot drift — and then smoke-tested by extracting it, running its
`build.sh`, and driving a scripted frame through its `run.sh`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arena.bundle import (  # noqa: E402
    MAX_FILES,
    MAX_UNPACKED_BYTES,
    MAX_ZIP_BYTES,
    BundleError,
    check_limits,
    default_output_path,
)
from arena.records.fingerprint import bot_content_hash  # noqa: E402

BOT_ID = "morpheus-rs"
BINARY = "morpheus-rs"

# The dev-time minifier (tools/minify). Built on the packaging host, where a
# network exists; never shipped, never linked into the bot.
MINIFY_DIR = BOT_DIR / "tools" / "minify"
MINIFY_BIN = MINIFY_DIR / "target" / "release" / "morpheus-rs-minify"

# Fixed timestamp, matching arena.bundle: identical sources produce a
# byte-identical zip, so "did the submission change" is a checksum question.
_ZIP_DATE = (2020, 1, 1, 0, 0, 0)

# Source files the archive ships. `rust-toolchain.toml` is
# deliberately **not** here: it pins an exact patch release, and a sandbox
# whose stable toolchain is any other patch would try to download the pinned
# one — over a network that does not exist at intake. The pin is a
# development-side guarantee that nothing newer than the sandbox's 1.97 gets
# used; shipping it would convert that guarantee into a build failure.
SOURCE_MEMBERS = ("Cargo.toml", "Cargo.lock")
SOURCE_TREES = ("crates",)

# Required, because it is the only thing that says which bot this is. Without
# it the binary falls back to the Part 07 placeholders — four times the
# particles, twice the search depth, a different deadline — and plays a
# configuration nobody measured. Landed at M5, when the runtime started
# reading it.
CONFIG_MEMBERS = ("deployment.json",)

# The safetensors weights and their manifest — about 1 MB against a 50 MB
# limit — and required as of M8.
#
# It used to be optional, on the reasoning that a zip without weights "fails
# loudly at load". It does not. `Seat::new` returning `Err` degrades the seat
# to passing every turn, because the judge forfeits a game on an early exit and
# charges one fault out of fifty for a bad reply — so a bot with no weights
# answers every frame with a well-formed skip. That is the right trade during a
# game and it is why nothing downstream can see the difference; the packager is
# where the difference still exists.
ARTIFACT_TREES = ("artifact",)

RUN_SH_VENDORED = """#!/usr/bin/env bash
# Submission launcher. No build here: build.sh already ran at intake.
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/target/release/morpheus-rs"
"""

# The `.cargo/config.toml` that goes in the vendored zip. The source
# replacement is what makes `--offline` work: cargo resolves every dependency
# out of `vendor/` instead of reaching for crates.io.
CARGO_CONFIG_VENDORED = """# Generated by tools/package_submission.py — do not edit in the zip.
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=x86-64-v3"]
"""

# One handshake and two frames on the smallest board the parser will accept.
# Enough to prove the binary speaks the protocol; not a game.
SMOKE_INPUT = "0 2 2\n" + ("1 1 1 1 1\n1 1\n1 1\n1 0\n0 0\n1 0\n0 0\n" * 2)
SMOKE_EXPECTED_LINES = 2


# Ships in the zip. The judge never reads it; a human comparing a rated
# result on generals.bot against a row in data/bot_versions/morpheus-rs.json
# does, and without it "which program is playing up there" is answerable only
# by rebuilding and hoping. Deliberately carries no timestamp: the zip is
# byte-reproducible from the sources, and a clock in it would end that.
PROVENANCE_NAME = "SUBMISSION.json"


class PackageError(RuntimeError):
    """The submission cannot be built, or failed its own smoke test."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(REPO), capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _provenance(minify_stats: dict | None = None) -> bytes:
    manifest = json.loads((BOT_DIR / "artifact" / "manifest.json").read_text())
    payload = {
        "bot_id": BOT_ID,
        "minified": bool((minify_stats or {}).get("minified")),
        "content_hash": bot_content_hash(BOT_DIR / "run.sh"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "bots/morpheus-rs")),
        "weights_sha256": manifest["weights_sha256"],
        "checkpoint": manifest.get("training_run", {}).get("checkpoint_id", ""),
        "note": (
            "Rated identity of the program in this zip. The content hash is the "
            "repo bot's, computed over sources + Cargo.lock + run.sh + artifact; "
            "the zip's own launchers are generated and are not part of it. "
            "`minified` means the .rs members were stripped of comments on the "
            "way in — the program is the same, the line numbers are not, so a "
            "judge traceback locates a fault in the stripped file. Re-package "
            "with --no-minify when you need to read one."
        ),
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


@dataclass
class Package:
    content_hash: str
    zip_path: Path
    zip_sha256: str
    zip_bytes: int
    unpacked_bytes: int
    file_count: int
    smoked: bool
    smoke_note: str
    selfcheck: dict[str, str]


def _cargo_env() -> dict[str, str]:
    """Environment with cargo on PATH, wherever rustup put it."""
    env = os.environ.copy()
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        env["PATH"] = f"{cargo_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def _run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command, cwd=str(cwd), env=env or _cargo_env(), capture_output=True, text=True
    )
    if result.returncode != 0:
        raise PackageError(
            f"{' '.join(command)} failed in {cwd}:\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def _ensure_minifier() -> Path:
    """Build `tools/minify` if it is not already built, and return the binary."""
    if not MINIFY_BIN.is_file():
        _run(["cargo", "build", "--release"], cwd=MINIFY_DIR)
    if not MINIFY_BIN.is_file():
        raise PackageError(f"no minifier at {MINIFY_BIN}")
    return MINIFY_BIN


def strip_line_comments(text: str) -> str:
    """
    Drop whole-line `#` comments from a shell script or a TOML file.

    The launchers and `build.sh` carry as much reasoning as the Rust does — why
    `cd` before `cargo build`, what a missing FMA costs, why intake fails loudly
    — and none of it is needed to run them. This is the shell half of what
    `tools/minify` does to `crates/**`, and it runs under the same rule: on the
    way into the zip only.

    **Whole-line only, and the shebang stays.** A trailing `# …` after code is
    left alone, because a line-based rule cannot tell a comment from a `#`
    inside a string or from `${var#prefix}`, and a launcher that stops working
    is a forfeit. Blank runs left behind by removed blocks are collapsed so the
    result reads like a script rather than a sieve.
    """
    kept: list[str] = []
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if index == 0 and stripped.startswith("#!"):
            kept.append(line)
            continue
        if stripped.startswith("#"):
            continue
        if not stripped and (not kept or not kept[-1].strip()):
            continue
        kept.append(line)
    return "\n".join(kept).rstrip("\n") + "\n"


def _shell(text: str, *, minify: bool) -> bytes:
    return (strip_line_comments(text) if minify else text).encode()


def _minify_rust(source: bytes, arcname: str) -> bytes:
    """
    One `.rs` file, stripped of everything the judge does not need.

    Same discipline as `arena/bundle.py`: the rewrite happens **on the way into
    the zip only** and the repo file is never touched. A failure is fatal
    rather than a fallback to the original text — shipping the un-minified file
    while reporting a minified bundle is the one outcome worse than not
    minifying at all.
    """
    result = subprocess.run(
        [str(_ensure_minifier())], input=source, capture_output=True
    )
    if result.returncode != 0:
        raise PackageError(
            f"minifying {arcname}: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _artifact_members() -> list[tuple[Path, str]]:
    """
    (path, arcname) for the weights, checked against their own manifest.

    The manifest's `weights_sha256` is verified here as well as by the loader,
    for the reason the tree is no longer optional: a loader that rejects the
    weights produces a bot that passes every turn, and the packager is the last
    place a wrong artifact is still a visible failure.
    """
    root = BOT_DIR / "artifact"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise PackageError(
            "missing artifact/manifest.json; run tools/convert_artifact.py. "
            "A zip without weights does not fail loudly — it plays a seat that "
            "skips every turn."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = root / manifest["artifact_file"]
    if not weights.is_file():
        raise PackageError(f"manifest names {manifest['artifact_file']}, which is missing")
    import hashlib

    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != manifest["weights_sha256"]:
        raise PackageError(
            f"{weights.name} hashes to {digest}, manifest says "
            f"{manifest['weights_sha256']}"
        )
    return [
        (path, str(path.relative_to(BOT_DIR)))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _iter_source_files(*, minify: bool) -> tuple[list[tuple[str, bytes]], dict]:
    """
    (arcname, bytes) for everything the archive compiles from.

    Returns contents rather than paths because `.rs` members are rewritten on
    the way in — comments and doc comments stripped — and the repo file is
    never touched. The second return value is what the rewrite cost, so
    "was this bundle minified" is a number rather than a flag.
    """
    members: list[tuple[str, bytes]] = []
    for name in SOURCE_MEMBERS:
        path = BOT_DIR / name
        if not path.is_file():
            raise PackageError(f"missing {name}; run cargo build once first")
        members.append((name, path.read_bytes()))
    for name in CONFIG_MEMBERS:
        path = BOT_DIR / name
        if not path.is_file():
            raise PackageError(f"missing {name}; the bot would play placeholder knobs")
        members.append((name, path.read_bytes()))

    before = after = 0
    for tree in SOURCE_TREES:
        root = BOT_DIR / tree
        if not root.is_dir():
            raise PackageError(f"missing {tree}/")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "target" in path.parts:
                continue
            arcname = str(path.relative_to(BOT_DIR))
            data = path.read_bytes()
            if minify and path.suffix == ".rs":
                before += len(data)
                data = _minify_rust(data, arcname)
                after += len(data)
            members.append((arcname, data))

    for path, arcname in _artifact_members():
        members.append((arcname, path.read_bytes()))
    return members, {
        "minified": bool(minify and before),
        "rust_bytes_before": before,
        "rust_bytes_after": after,
    }


def _write_entry(zf: zipfile.ZipFile, arcname: str, data: bytes, *, mode: int) -> None:
    info = zipfile.ZipInfo(arcname, date_time=_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    # The judge executes run.sh and build.sh; a zip that loses the executable
    # bit produces a "permission denied" at intake that looks like a bot bug.
    info.external_attr = (mode & 0o7777) << 16
    zf.writestr(info, data)


def _vendor(workdir: Path) -> tuple[Path, int]:
    """
    Vendor the dependency graph. Returns (dir, file count) — often (dir, 0).

    Zero is the good answer and the reason the crate has no dependencies: the
    binding sandbox limit is 10,000 unpacked files, and `cargo vendor` over a
    fat graph clears it easily (rewrite-plan §1, R6).
    """
    vendor_dir = workdir / "vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    _run(["cargo", "vendor", "--versioned-dirs", str(vendor_dir)], cwd=BOT_DIR)
    return vendor_dir, sum(1 for p in vendor_dir.rglob("*") if p.is_file())


def build_vendored(
    out_path: Path, workdir: Path, *, minify: bool
) -> tuple[Path, int, dict]:
    vendor_dir, vendor_files = _vendor(workdir)
    build_sh = (BOT_DIR / "tools" / "submission" / "build.sh").read_text()
    members, minify_stats = _iter_source_files(minify=minify)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_entry(zf, "run.sh", _shell(RUN_SH_VENDORED, minify=minify), mode=0o755)
        _write_entry(zf, "build.sh", _shell(build_sh, minify=minify), mode=0o755)
        _write_entry(
            zf,
            ".cargo/config.toml",
            _shell(CARGO_CONFIG_VENDORED, minify=minify),
            mode=0o644,
        )
        for arcname, data in members:
            _write_entry(zf, arcname, data, mode=0o644)
        _write_entry(zf, PROVENANCE_NAME, _provenance(minify_stats), mode=0o644)
        for path in sorted(vendor_dir.rglob("*")):
            if path.is_file():
                arcname = f"vendor/{path.relative_to(vendor_dir)}"
                _write_entry(zf, arcname, path.read_bytes(), mode=0o644)
    return out_path, vendor_files, minify_stats


def _selfcheck_facts(stdout: str) -> dict[str, str]:
    """The `key value` lines `morpheus-rs selfcheck` prints, as a dict."""
    facts: dict[str, str] = {}
    for line in stdout.splitlines():
        head, _, tail = line.partition(" ")
        if head in {
            "hardware_fma",
            "weights_sha256",
            "checkpoint_id",
            "load_ms",
            "warmup_ms",
            "init_ms",
            "decision",
            "decide_ms",
            "selfcheck",
        }:
            facts[head] = tail.strip()
    return facts


def smoke(zip_path: Path) -> tuple[bool, str, dict[str, str]]:
    """
    Extract the archive, run its build.sh, and speak the protocol to run.sh.

    This is the part that makes packaging a check rather than a hope: it runs
    the submitted bytes, from a directory with no repo around them, exactly as
    the judge will.

    **What it could not see until M8.** A well-formed reply is not evidence of
    a working bot. A seat that cannot load its weights or its knobs passes
    every turn rather than exiting — the deliberate trade in `main.rs`, since
    the judge forfeits on an early exit — so this test scored a bundle with
    `artifact/` deleted as two well-formed actions, byte-identical to a healthy
    one. The `selfcheck` that `build.sh` now runs is what closes it, and its
    output is parsed here rather than merely allowed to pass.
    """
    with tempfile.TemporaryDirectory(prefix="morpheus-rs-smoke-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        for name in ("run.sh", "build.sh"):
            (root / name).chmod(0o755)

        build = subprocess.run(
            ["bash", str(root / "build.sh")],
            cwd=str(root),
            env=_cargo_env(),
            capture_output=True,
            text=True,
        )
        facts = _selfcheck_facts(build.stdout)
        if build.returncode != 0:
            return (
                False,
                f"build.sh failed: {(build.stderr or build.stdout)[-600:]}",
                facts,
            )
        if facts.get("selfcheck") != "ok":
            return False, "build.sh did not run selfcheck", facts

        play = subprocess.run(
            ["bash", str(root / "run.sh")],
            cwd=str(root),
            input=SMOKE_INPUT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        lines = [line for line in play.stdout.strip().splitlines() if line.strip()]
        if len(lines) != SMOKE_EXPECTED_LINES:
            return (
                False,
                (
                    f"run.sh replied {len(lines)} line(s), expected "
                    f"{SMOKE_EXPECTED_LINES}: {play.stdout!r} {play.stderr[-300:]}"
                ),
                facts,
            )
        for line in lines:
            parts = line.split()
            if len(parts) != 5 or not all(p.lstrip("-").isdigit() for p in parts):
                return False, f"malformed reply {line!r}", facts
        return (
            True,
            (
                f"selfcheck ok (fma={facts.get('hardware_fma')}, "
                f"decided `{facts.get('decision')}` in "
                f"{facts.get('decide_ms')} ms), replied {len(lines)} "
                f"well-formed actions"
            ),
            facts,
        )


def package(
    *, out_dir: Path, force: bool, run_smoke: bool, minify: bool = True
) -> Package:
    out_dir.mkdir(parents=True, exist_ok=True)
    content_hash = bot_content_hash(BOT_DIR / "run.sh")
    # `arena.bundle`'s own naming, so a Rust bundle and a Python one are the
    # same kind of object on disk: `<bot_id>-<content_hash>.zip`. The hash is
    # the rated identity, so an archive names the program it holds and two
    # builds of different code cannot collide on one path.
    out_path = default_output_path(BOT_ID, content_hash)
    if out_dir != out_path.parent:
        out_path = out_dir / out_path.name
    if out_path.exists() and not force:
        raise PackageError(f"{out_path} exists (pass --force to overwrite)")

    vendor_files = 0
    minify_stats: dict = {}
    with tempfile.TemporaryDirectory(prefix="morpheus-rs-pkg-") as tmp:
        # Built beside the destination and moved into place only on success.
        # Writing straight to `out_path` leaves a **partial archive** when a
        # member check fails mid-zip, and nothing downstream can tell one from
        # a finished bundle: an aborted static build left a four-file zip with
        # no weights in `data/bundles/`, which the Modal smoke then dutifully
        # shipped to an x86 container. It was caught there only because
        # `build.sh` now runs `selfcheck`.
        staged = Path(tmp) / out_path.name
        _, vendor_files, minify_stats = build_vendored(
            staged, Path(tmp), minify=minify
        )
        out_path.unlink(missing_ok=True)
        shutil.move(str(staged), str(out_path))

    zip_bytes, unpacked_bytes, file_count = check_limits(out_path)
    smoked, note, facts = (False, "skipped", {})
    if run_smoke:
        smoked, note, facts = smoke(out_path)
    note = f"{note}; {vendor_files} vendored file(s)"
    if minify_stats.get("minified"):
        note = (
            f"{note}; sources minified "
            f"{minify_stats['rust_bytes_before']} -> "
            f"{minify_stats['rust_bytes_after']} B"
        )
    import hashlib

    return Package(
        content_hash=content_hash,
        zip_path=out_path,
        # The zip is byte-reproducible from the sources (fixed member dates,
        # sorted members, no clock in SUBMISSION.json), so this digest answers
        # "are the bytes I submitted the bytes I still have" without a rebuild.
        zip_sha256=hashlib.sha256(out_path.read_bytes()).hexdigest(),
        zip_bytes=zip_bytes,
        unpacked_bytes=unpacked_bytes,
        file_count=file_count,
        smoked=smoked,
        smoke_note=note,
        selfcheck=facts,
    )


def gate(zip_path: Path, *, opponent: str, seed: int, keep: bool) -> dict:
    """
    Play the AGENTS.md verification gate with the *submitted* bytes.

    Every gate this project has run so far drove `bots/morpheus-rs/run.sh` —
    the repo launcher, which builds from the working tree and finds the
    artifact by walking up from `target/`. The zip has a different launcher, a
    different directory shape and a build step, and none of that was ever put
    in front of a competition match. A frame script cannot substitute: it never
    reaches a belief update, an admission decision, a castle build or a
    deathtouch turn.

    **`build.sh` is run and then deleted**, which is a deviation worth stating
    rather than hiding. The judge's sequence is exactly that — build once at
    intake, then spawn `run.sh` for every game and never look at `build.sh`
    again (RULES.md §08) — so removing it models the end of intake. It is also
    the only way this gate can run: `matchup.py::build_agent` executes any
    `build.sh` beside a `run.sh` and formats its log line with
    `build.relative_to(REPO_ROOT)`, unguarded, where `REPO_ROOT` is the
    *submodule's* root. Every path outside `competition-module/` raises, so a
    submission-shaped directory crashes the gate before the first move no
    matter where it is extracted. What goes untested is `build_agent`, which
    the judge does not have.
    """
    root = REPO / "data" / "bundles" / f"extracted-{zip_path.stem}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    for name in ("run.sh", "build.sh"):
        (root / name).chmod(0o755)

    build = subprocess.run(
        ["bash", str(root / "build.sh")],
        cwd=str(root),
        env=_cargo_env(),
        capture_output=True,
        text=True,
        timeout=60 * 30,
    )
    if build.returncode != 0:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
        return {
            "opponent": opponent,
            "seed": seed,
            "returncode": build.returncode,
            "tail": (build.stdout + build.stderr).strip().splitlines()[-12:],
        }
    (root / "build.sh").unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "competition-module" / "competition" / "matchup.py"),
            str(root / "run.sh"),
            str(REPO / "bots" / opponent / "run.sh"),
            "--mode",
            "competition",
            "--seed",
            str(seed),
        ],
        cwd=str(REPO),
        # The Python side of the pairing needs an interpreter with the repo's
        # dependencies; `matchup.py` spawns `run.sh`, which honours PYTHON.
        env={**_cargo_env(), "PYTHON": str(REPO / ".venv" / "bin" / "python")},
        capture_output=True,
        text=True,
        timeout=60 * 30,
    )
    tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    return {
        "opponent": opponent,
        "seed": seed,
        "returncode": result.returncode,
        "tail": tail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=default_output_path(BOT_ID).parent)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument(
        "--no-minify",
        action="store_true",
        help="ship the .rs sources verbatim, comments and all. Use when a judge "
        "traceback has to be read against real line numbers.",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="play the AGENTS.md competition gate with the extracted bundle "
        "(minutes, not seconds)",
    )
    parser.add_argument("--gate-opponent", default="cm_expander")
    parser.add_argument("--gate-seed", type=int, default=0)
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="leave data/bundles/extracted-* in place after the gate",
    )
    args = parser.parse_args(argv)

    failures: list[dict] = []
    try:
        built = package(
            out_dir=args.out_dir,
            force=args.force,
            run_smoke=not args.no_smoke,
            minify=not args.no_minify,
        )
    except (PackageError, BundleError) as exc:
        print(json.dumps({"package": None, "failures": [{"error": str(exc)}]}, indent=2))
        return 1

    row: dict = {
        "content_hash": built.content_hash,
        "zip": str(built.zip_path),
        "zip_sha256": built.zip_sha256,
        "zip_bytes": built.zip_bytes,
        "zip_limit": MAX_ZIP_BYTES,
        "unpacked_bytes": built.unpacked_bytes,
        "unpacked_limit": MAX_UNPACKED_BYTES,
        "files": built.file_count,
        "files_limit": MAX_FILES,
        "smoked": built.smoked,
        "smoke": built.smoke_note,
        "selfcheck": built.selfcheck,
    }
    # With one variant left, a smoke that ran and did not pass is a failed
    # packaging run rather than a row in a table nobody reads. `--no-smoke` is
    # the caller saying they know.
    if not args.no_smoke and not built.smoked:
        failures.append({"error": f"smoke test did not pass: {built.smoke_note}"})
    if args.gate and built.smoked:
        try:
            row["gate"] = gate(
                built.zip_path,
                opponent=args.gate_opponent,
                seed=args.gate_seed,
                keep=args.keep_extracted,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            row["gate"] = {"error": str(exc)}
        if row["gate"].get("returncode") != 0:
            failures.append({"error": "competition gate failed"})

    print(json.dumps({"package": row, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
