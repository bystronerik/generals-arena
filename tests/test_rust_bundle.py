"""
The shared Rust packager: the shell minifier, and the per-bot spec facts.

`arena/rust_bundle.py` strips whole-line `#` comments from the zip's `run.sh`,
`build.sh` and `.cargo/config.toml`, for the same reason the `.rs` members go
through `tools/rust-minify`: a submission carries logic, not the reasoning
behind it. The risk is entirely on one side — a rule that removes too much
ships a launcher that does not launch, and the judge forfeits a game on a bot
that fails to start. So the cases below are mostly about what must survive.

The rest pins the spec fields where morpheus-rs and joe-rs genuinely differ and
where getting one wrong is silent: which manifest key names the file that
ships, which digest goes in `SUBMISSION.json`, whether the toolchain pin is a
member, and where `build.sh` sits.

Everything here is string-, dict- or path-level. No cargo, no zip writing, no
subprocess: packaging a real bundle costs a `cargo vendor` plus a release build
of 93 crates, which belongs to the J5.3 command and not to a 15 s suite.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from arena.rust_bundle import (
    PackageError,
    _artifact_members,
    _dotted,
    default_output_path,
    strip_line_comments,
)
from arena.records.fingerprint import bot_content_hash
from arena.records.store import REPO_ROOT


def _load_spec(bot_id: str):
    """The bot's SPEC, loaded by path.

    `bots/<bot>/tools/` is not a package — it is excluded from the bot's
    content hash, so it has no `__init__.py` and never will. The sys.modules
    registration is not optional: `@dataclass` resolves its own module by name
    while a class body executes, and an unregistered module makes that lookup
    return None.
    """
    tool = REPO_ROOT / "bots" / bot_id / "tools" / "package_submission.py"
    name = f"{bot_id.replace('-', '_')}_package_submission"
    spec = importlib.util.spec_from_file_location(name, tool)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.SPEC


MORPHEUS = _load_spec("morpheus-rs")
JOE = _load_spec("joe-rs")
BOTH = pytest.mark.parametrize("spec", [MORPHEUS, JOE], ids=["morpheus-rs", "joe-rs"])


# --- the shell minifier ---------------------------------------------------


def test_the_shebang_survives():
    """Without it the judge runs the script under whatever it feels like."""
    out = strip_line_comments("#!/usr/bin/env bash\n# a note\nexec ./bot\n")
    assert out.splitlines()[0] == "#!/usr/bin/env bash"
    assert "a note" not in out


def test_whole_line_comments_go_indented_or_not():
    out = strip_line_comments("#!/bin/sh\n# top\n    # indented\ncd \"$DIR\"\n")
    assert out == '#!/bin/sh\ncd "$DIR"\n'


def test_a_hash_that_is_not_a_comment_is_left_alone():
    """
    The reason this is line-based. `${var#prefix}` is a parameter expansion and
    `"#"` is a string; a rule that hunted trailing comments would eat both, and
    the failure would be a bot that does not start.
    """
    source = '#!/bin/sh\nX="${DIR#/}"   # trailing note\necho "#1"\n'
    out = strip_line_comments(source)
    assert "${DIR#/}" in out
    assert 'echo "#1"' in out


def test_blank_runs_left_by_removed_blocks_collapse_to_one():
    """Removing a comment block should not leave the hole it came out of."""
    out = strip_line_comments("#!/bin/sh\n\n# one\n# two\n\n\nexec ./bot\n")
    assert out == "#!/bin/sh\n\nexec ./bot\n"


# --- the generated launchers ----------------------------------------------

# One per math backend, plus rayon. Every bot is single-threaded by
# construction and every latency number in the project assumes it.
THREAD_PINS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


@BOTH
def test_the_shipped_launcher_still_does_what_it_did(spec):
    """
    Not a syntax check — the packaging smoke runs this for real. This is the
    cheap half: every line that makes the launcher a launcher is still there
    after the comments come out.
    """
    out = strip_line_comments(spec.run_sh)
    assert out.startswith("#!/usr/bin/env bash\n")
    # joe-rs relaxed `-e` on purpose: its launcher inspects a failing command
    # (`selfcheck`) and falls back instead of dying, and `set -e` would kill
    # the shell before the fallback could run.
    assert ("set -euo pipefail" in out) or ("set -uo pipefail" in out)
    for pin in THREAD_PINS:
        assert f"export {pin}=1" in out
    # Where the binary lives differs: morpheus-rs compiles at intake, joe-rs
    # ships one. Both must still hand the process over with `exec`.
    assert f'exec "$BIN"' in out or f'exec "$DIR/target/release/{spec.binary}"' in out
    assert "#" not in out.split("\n", 1)[1]


def test_the_joe_launcher_has_no_prebuilt_fallback_seat():
    """
    The prebuilt-binary era shipped a launcher that self-tested and dropped to
    a bash seat passing every turn, because a foreign-built binary could fail
    to run on the judge's glibc. The 2026-08-15 return to a source build
    (packaging.md §11) deletes that risk — the binary is built by the judge's
    own toolchain, and `build.sh` runs `selfcheck` under `set -e` at intake —
    so the launcher is a plain exec again, like morpheus-rs's. A fallback seat
    reappearing here would mean the prebuilt path crept back without its
    sidecar checks.
    """
    out = strip_line_comments(JOE.run_sh)
    assert "printf '1 0 0 0 0" not in out, "the fallback bash seat is gone"
    assert '"$DIR/bin/joe-rs"' not in out, "no prebuilt binary path"


def test_the_joe_launcher_exports_the_artifact_dir():
    """
    joe-rs's exe-relative fallback would resolve in the submission layout, but
    its last resort is a bare relative `PathBuf::from("artifact")` that depends
    on the judge's working directory, and `current_exe()` can fail. The export
    removes both, and it has to survive the minifier to do that.
    """
    out = strip_line_comments(JOE.run_sh)
    assert 'export JOE_RS_ARTIFACT="$DIR/artifact"' in out
    assert 'exec "$DIR/target/release/joe-rs"' in out
    # The six thread pins plus the artifact directory.
    assert out.count("export ") == 7


# --- what must never ship -------------------------------------------------


@BOTH
def test_the_toolchain_pin_never_ships(spec):
    """
    `rust-toolchain.toml` pins an exact patch release. A sandbox whose stable
    toolchain is any other patch would try to download the pinned one over a
    network that does not exist at intake, so the pin is a development-side
    guarantee and shipping it would convert that guarantee into a build
    failure. Its absence from every spec field is the whole mechanism, which
    makes it worth an assertion rather than a comment.

    Scope: this is about the *bot's own* pin. In the 93-crate era joe-rs's
    zip also contained four `vendor/*/rust-toolchain.toml` files copied in by
    `cargo vendor`; they were inert (rustup walks *up* from the working
    directory, and `build.sh` runs from the bundle root, above them), and
    J5.4 proved it empirically. Both bots now vendor zero crates, so only the
    bot's own pin is left to keep out.
    """
    pin = "rust-toolchain.toml"
    assert (spec.bot_dir / pin).is_file(), "the pin should exist, just not ship"
    assert pin not in spec.source_members
    assert pin not in spec.config_members
    for tree in spec.source_trees:
        assert not (spec.bot_dir / tree / pin).exists()


@BOTH
def test_build_sh_never_sits_beside_run_sh(spec):
    """
    `matchup.py::build_agent` executes any `build.sh` it finds next to a
    `run.sh`, then crashes formatting `build.relative_to(REPO_ROOT)` for a path
    outside the submodule. A `build.sh` in a bot directory therefore breaks the
    repo's own verification gate before the first move.
    """
    assert spec.build_sh_path.is_file()
    assert spec.build_sh_path.parent != spec.bot_dir


# --- the two manifest schemas ---------------------------------------------


def _fake_artifact(root: Path, payload: dict, filename: str, body: bytes) -> Path:
    artifact = root / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / filename).write_bytes(body)
    (artifact / "manifest.json").write_text(json.dumps(payload))
    return artifact


@pytest.mark.parametrize(
    "spec, filename, payload_for",
    [
        pytest.param(
            MORPHEUS,
            "weights.safetensors",
            lambda digest: {"artifact_file": "weights.safetensors", "weights_sha256": digest},
            id="morpheus-rs",
        ),
        pytest.param(
            JOE,
            "model.packed",
            # Two decoy digests sit right beside the shipped one: the eqx
            # digest under morpheus's key, and the safetensors digest — the
            # file build.sh reconstructs at intake, not a zip member. Reading
            # either wrong key is the failure this test exists for.
            lambda digest: {
                "packed": "model.packed",
                "packed_sha256": digest,
                "safetensors": "model.safetensors",
                "safetensors_sha256": "1" * 64,
                "weights": "ema.eqx",
                "weights_sha256": "0" * 64,
            },
            id="joe-rs",
        ),
    ],
)
def test_manifest_keys_are_read_per_spec(tmp_path, spec, filename, payload_for):
    import hashlib

    body = b"not really weights"
    digest = hashlib.sha256(body).hexdigest()
    _fake_artifact(tmp_path, payload_for(digest), filename, body)
    local = replace(spec, bot_dir=tmp_path)

    # Sorted, because member order is what makes the zip byte-reproducible.
    arcnames = [arcname for _, arcname in _artifact_members(local)]
    assert arcnames == sorted(["artifact/manifest.json", f"artifact/{filename}"])

    # A digest mismatch is fatal, and stays fatal: for joe it is the only
    # automatic detector of a stale artifact, which is gitignored and so
    # invisible to `git_dirty`.
    _fake_artifact(tmp_path, payload_for("f" * 64), filename, body)
    with pytest.raises(PackageError, match="hashes to"):
        _artifact_members(local)


def test_joe_provenance_does_not_call_the_eqx_digest_weights():
    """
    joe's `weights_sha256` is the `.eqx` checkpoint, which is **not in the
    zip**. Publishing it under that name in `SUBMISSION.json` would invite a
    human to compare it against `model.safetensors` and conclude the bundle is
    corrupt.
    """
    assert "weights_sha256" not in JOE.provenance_fields
    assert JOE.provenance_fields["source_eqx_sha256"] == "weights_sha256"
    assert JOE.provenance_fields["safetensors_sha256"] == "safetensors_sha256"
    # The shipped file is the joe-net-v2 container; its digest must be the
    # one _artifact_members verifies, and the safetensors must stay out of
    # the zip (build.sh reconstructs it at intake).
    assert JOE.provenance_fields["packed_sha256"] == "packed_sha256"
    assert JOE.artifact_file_key == "packed"
    assert JOE.artifact_exclude == ("model.safetensors",)
    # morpheus keeps its key, because its manifest means the shipped file by it.
    assert MORPHEUS.provenance_fields["weights_sha256"] == "weights_sha256"


def test_dotted_manifest_paths_resolve_or_blank():
    """Provenance is for a human; a blank field beats an unpackageable bot."""
    manifest = {"checkpoint": {"run_name": "joe-M", "global_step": 6000}}
    assert _dotted(manifest, "checkpoint.run_name") == "joe-M"
    assert _dotted(manifest, "checkpoint.global_step") == 6000
    assert _dotted(manifest, "checkpoint.missing") == ""
    assert _dotted(manifest, "training_run.checkpoint_id") == ""
    assert _dotted(manifest, "checkpoint.run_name.deeper") == ""


@BOTH
def test_archives_are_named_for_the_program_they_hold(spec):
    """
    `<bot_id>-<content_hash>.zip`, through `arena.bundle`'s own call — so a
    Rust bundle and a Python one are the same kind of object on disk, and two
    builds of different code cannot land on one path.
    """
    path = default_output_path(spec.bot_id)
    assert path.parent.name == "bundles"
    assert path.name.startswith(f"{spec.bot_id}-") and path.suffix == ".zip"
    assert path.stem.rsplit("-", 1)[1] == bot_content_hash(spec.bot_dir / "run.sh")
