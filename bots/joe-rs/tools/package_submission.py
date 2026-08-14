#!/usr/bin/env python3
"""Package joe-rs for generals.bot, and prove the archive before it ships.

    python bots/joe-rs/tools/package_submission.py --force
    python bots/joe-rs/tools/package_submission.py --force --gate

Milestone J5 of docs/bots/joe-rs/packaging.md. The work is all in
`arena/rust_bundle.py`, which morpheus-rs shares; this file is the spec that
says what joe-rs is, and nothing else.

One thing differs from morpheus in a way worth knowing before reading the spec:
the dependency graph is real — 93 crates, ~3,946 vendored files — and the
**zip cap is the binding limit at about 80 %**, three quarters of it the
float32 `model.safetensors`; the file count is not close.

The other difference is gone. joe-rs used to propagate a load failure out of
`main` and exit 1, so a bundle that could not find its artifact answered
nothing and the smoke's line count caught it. That is a forfeit under RULES.md
§08, so the seat now degrades to passing every turn the way morpheus always
has — and a broken bundle now answers two well-formed skips, indistinguishable
from a healthy one by line count alone. `selfcheck` is what replaces the lost
detector, and `smoke_reject_all_pass` still covers the seat that runs without
deciding.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arena.rust_bundle import RustBotSpec, main  # noqa: E402

# The shipped binary and the record of what it was built from. Under `tools/`
# deliberately: the content-hash walk skips that directory, so a 2 MB binary
# does not enter the rated closure — and, unlike `target/`, it survives a
# `cargo clean`. See `_check_prebuilt_is_current` for the staleness guard this
# placement makes necessary.
PREBUILT = BOT_DIR / "tools" / "submission" / "joe-rs-x86_64-linux-gnu"
SIDECAR = PREBUILT.with_suffix(".json")

# The zip's launcher **sets `JOE_RS_ARTIFACT` explicitly**. The exe-relative
# fallback in `main.rs::artifact_dir` would also resolve in the submission
# layout — `target/release/joe-rs` walks up three parents to the bundle root,
# where `artifact/` sits — but its last resort is a bare relative
# `PathBuf::from("artifact")`, which depends on the judge's working directory,
# and `current_exe()` can fail. One `export` line costs nothing and removes
# both. Both paths are proved, not argued: J5.3 runs the extracted bundle
# twice, once with this launcher and once with the variable unset.
RUN_SH_VENDORED = """#!/usr/bin/env bash
# Submission launcher. No build here: build.sh already ran at intake.
set -uo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1
DIR="$(cd "$(dirname "$0")" && pwd)"
export JOE_RS_ARTIFACT="$DIR/artifact"
BIN="$DIR/bin/joe-rs"

# Self-test before committing the seat to the binary. `selfcheck` loads the
# artifact and decides one frame, so it fails for every reason the binary
# would fail to play: intake never produced it, the build lost its target
# features, the artifact is missing. It costs about 150 ms of the 10 s
# first-move grace.
if [ -x "$BIN" ] && "$BIN" selfcheck >/dev/null 2>&1; then
  exec "$BIN"
fi

# The binary cannot play. RULES.md 08 forfeits the match for a crash or an
# early exit, and charges one fault out of fifty for a bad reply — so a seat
# that passes every turn loses games but stays in the tournament, and a
# launcher that dies here loses everything. It also makes the failure legible:
# "lost every game" and "crashed in every game" point at different causes.
echo "[joe-rs] binary unusable; playing the fallback seat" >&2
read -r _player H _W || exit 0
while read -r _turn _a _b _c _d; do
  rows=$((3 * H))
  while [ "$rows" -gt 0 ]; do
    read -r _ || exit 0
    rows=$((rows - 1))
  done
  printf '1 0 0 0 0\\n'
done
"""

# Wire cell types and owners (competition-module/competition/protocol.py,
# mirrored in src/wire.rs).
_FOG, _PLAIN, _GENERAL = 0, 1, 4
_OWNER_NONE, _OWNER_ME = 0, 1


def _smoke_frame(h: int, w: int, turn: int, army: int) -> str:
    """One frame where passing is the wrong answer.

    Our general sits mid-board on plentiful army with four empty plains around
    it and the rest of the board in fog. A seat that is deciding moves out; a
    seat that is merely running replies `1 0 0 0 0`, which is what
    `smoke_reject_all_pass` exists to fail.

    It has to be built rather than sliced from the committed parity fixture:
    the first 22 turns of a real recorded game are *all* passes, because the
    general has not accumulated army yet, so a fixture prefix would fail a
    healthy bundle. It also has to be 21x21 or smaller — the net pads to 21
    and `Seat::new` refuses anything larger.
    """
    gr, gc = h // 2, w // 2
    types = [[_FOG] * w for _ in range(h)]
    owners = [[_OWNER_NONE] * w for _ in range(h)]
    armies = [[0] * w for _ in range(h)]
    types[gr][gc] = _GENERAL
    owners[gr][gc] = _OWNER_ME
    armies[gr][gc] = army
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        types[gr + dr][gc + dc] = _PLAIN
    grids = "".join(
        "\n".join(" ".join(str(v) for v in row) for row in grid) + "\n"
        for grid in (types, owners, armies)
    )
    return f"{turn} 1 {army} 1 1\n{grids}"


_SMOKE_H = _SMOKE_W = 21
SMOKE_INPUT = f"0 {_SMOKE_H} {_SMOKE_W}\n" + "".join(
    _smoke_frame(_SMOKE_H, _SMOKE_W, turn, army)
    for turn, army in ((50, 40), (51, 41))
)

SPEC = RustBotSpec(
    bot_id="joe-rs",
    binary="joe-rs",
    bot_dir=BOT_DIR,
    source_trees=("src",),
    # The file that ships is the safetensors conversion, named and digested by
    # its own keys. `weights_sha256` in this manifest is the **`.eqx`
    # checkpoint**, which is not in the zip — reading it here would put a
    # digest in SUBMISSION.json that a human compares against a file that
    # is not there.
    artifact_file_key="safetensors",
    artifact_sha_key="safetensors_sha256",
    run_sh=RUN_SH_VENDORED,
    provenance_fields={
        "safetensors_sha256": "safetensors_sha256",  # the file in this zip
        "source_eqx_sha256": "weights_sha256",  # NOT in this zip
        "checkpoint": "checkpoint.run_name",
        "checkpoint_step": "checkpoint.global_step",
        "engine_sha": "checkpoint.engine_sha",
        "tensor_schema": "tensor_schema",
    },
    provenance_note=(
        "Rated identity of the program in this zip. The content hash is the "
        "repo bot's, computed over sources + Cargo.lock + run.sh + artifact; "
        "the zip's own launchers are generated and are not part of it. "
        "`safetensors_sha256` digests model.safetensors, the weights this zip "
        "actually carries; `source_eqx_sha256` is the .eqx checkpoint the "
        "conversion read, which is provenance of that conversion and not of "
        "anything shipped here. `minified` means the .rs members were stripped "
        "of comments on the way in — the program is the same, the line numbers "
        "are not, so a judge traceback locates a fault in the stripped file. "
        "Re-package with --no-minify when you need to read one."
    ),
    # No cargo at intake. The judge used to compile 93 vendored crates, and
    # every failure hypothesis that survived a day of measurement lived in that
    # step — its disk (342 MB), its memory (1.6 GB), its wall time (71-131 s
    # against morpheus-rs's 17 s), and its dependence on whatever toolchain the
    # sandbox has. A 1.9 MB prebuilt binary removes the step entirely.
    #
    # Linked against **glibc 2.31** (Debian bullseye) rather than static musl,
    # and that is a measurement, not a preference: musl's allocator costs this
    # network 4x per forward pass — p99 79.5 ms against 19.4 ms, on a 150 ms
    # budget — because the forward pass allocates hard. An old-glibc build is
    # the ordinary way to ship a portable Linux binary and runs on any glibc
    # from 2021 onward. If the judge's runtime is older than that, `run.sh`'s
    # self-test catches it and plays the fallback seat instead of crashing.
    vendor=False,
    extra_files=(("tools/submission/joe-rs-x86_64-linux-gnu", "bin/joe-rs", 0o755),),
    # An x86_64 Linux binary does not run on the arm64 macOS host that packages
    # it, so the packager's own smoke cannot execute this bundle. Verified
    # remotely instead — selfcheck plus a 1,602-turn game in a toolchain-free
    # x86 container — and skipped loudly here rather than passing vacuously.
    local_smoke=False,
    local_smoke_skip_reason=(
        "bundle ships a prebuilt x86_64 Linux binary, which cannot run on this host; "
        "verify with scripts/joe_rs_modal_static_build.py"
    ),
    smoke_input=SMOKE_INPUT,
    # Still on, and now carrying more weight rather than less. The seat no
    # longer exits when it cannot load — it passes every turn, the way morpheus
    # always has — so the line count alone no longer catches a broken artifact.
    # This is the half of that detector which lives in the packager;
    # `selfcheck` below is the other half.
    smoke_expected_lines=2,
    smoke_reject_all_pass=True,
    selfcheck_argv=("selfcheck",),
    selfcheck_keys=frozenset(
        {
            "hardware_fma",
            "hardware_avx2",
            "artifact_dir",
            "safetensors_sha256",
            "tensor_schema",
            "checkpoint",
            "checkpoint_step",
            "startup_ms",
            "decide_ms",
            "decision",
            "selfcheck",
        }
    ),
    gate_opponent_default="joe",
)


def _check_prebuilt_is_current() -> None:
    """Refuse to ship a binary built from sources that have since changed.

    The binary lives under `tools/`, which the content-hash walk skips, so it
    is **derived state outside the rated identity** — edit `src/` and the hash
    moves while the binary silently does not. That is the same shape as the
    artifact fan-out that has bitten this repo before, and the judge would
    never notice: a stale binary plays perfectly well, just as a different
    program than the one the registry names.

    The sidecar records the content hash the binary was built from. Disagreeing
    with the tree is fatal, not a warning.
    """
    if not PREBUILT.is_file():
        raise SystemExit(
            f"missing {PREBUILT.relative_to(BOT_DIR)}\n"
            f"Build it:  .venv/bin/modal run scripts/joe_rs_static_build.py\n"
            f"Fetch it:  .venv/bin/modal volume get joe-rs-static joe-rs "
            f"{PREBUILT} --force"
        )
    if not SIDECAR.is_file():
        raise SystemExit(f"missing {SIDECAR.name}; rebuild the binary to regenerate it")

    from arena.records.fingerprint import bot_content_hash

    recorded = json.loads(SIDECAR.read_text())
    current = bot_content_hash(BOT_DIR / "run.sh")
    if recorded.get("content_hash") != current:
        raise SystemExit(
            f"the prebuilt binary is stale: built from {recorded.get('content_hash')}, "
            f"the tree is {current}. Rebuild it before packaging."
        )
    digest = hashlib.sha256(PREBUILT.read_bytes()).hexdigest()
    if recorded.get("sha256") != digest:
        raise SystemExit(
            f"{PREBUILT.name} hashes to {digest}, the sidecar says {recorded.get('sha256')}"
        )


if __name__ == "__main__":
    _check_prebuilt_is_current()
    raise SystemExit(main(SPEC))
