#!/usr/bin/env python3
"""Package unclejoe for generals.bot, and prove the archive before it ships.

    python bots/unclejoe/tools/package_submission.py --force
    python bots/unclejoe/tools/package_submission.py --force --gate

The work is all in `arena/rust_bundle.py`, which joe-rs and morpheus-rs share;
this file is the spec that says what unclejoe is, and nothing else. It is the
joe-rs spec with the identity fields changed, because the bundle shape is the
same one: **an empty [dependencies]**, so intake compiles one crate from
source, `vendor/` ships empty, and `--offline` proves it. The 93-crate
vendored build is what qualification rejected, and this fork inherits the
answer rather than re-deriving it.

The seat still degrades to passing every turn when it cannot load (RULES.md
§08 forfeits a crash, charges a bad reply as one fault of fifty), so
`selfcheck` remains the intake-time detector: `build.sh` runs it and aborts a
submission whose binary cannot load the artifact and decide a frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arena.rust_bundle import RustBotSpec, main  # noqa: E402

# The zip's launcher **sets `UNCLEJOE_ARTIFACT` explicitly**. The exe-relative
# fallback in `main.rs::artifact_dir` would also resolve in the submission
# layout — `target/release/unclejoe` walks up three parents to the bundle root,
# where `artifact/` sits — but its last resort is a bare relative
# `PathBuf::from("artifact")`, which depends on the judge's working directory,
# and `current_exe()` can fail. One `export` line costs nothing and removes
# both.
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
export UNCLEJOE_ARTIFACT="$DIR/artifact"
exec "$DIR/target/release/unclejoe"
"""

# Wire cell types and owners (competition-module/competition/protocol.py,
# mirrored in src/io/wire.rs).
_FOG, _PLAIN, _GENERAL = 0, 1, 4
_OWNER_NONE, _OWNER_ME = 0, 1


def _smoke_frame(h: int, w: int, turn: int, army: int) -> str:
    """One frame where passing is the wrong answer.

    Our general sits mid-board on plentiful army with four empty plains around
    it and the rest of the board in fog. A seat that is deciding moves out; a
    seat that is merely running replies `1 0 0 0 0`, which is what
    `smoke_reject_all_pass` exists to fail.

    It has to be built rather than sliced from a recorded game: the first 22
    turns of a real game are *all* passes, because the general has not
    accumulated army yet, so a recorded prefix would fail a healthy bundle. It
    also has to be 21x21 or smaller — the net pads to 21 and `Seat::new`
    refuses anything larger.

    The position is deliberately one no tactic answers: the enemy general has
    never been seen, so the kill trigger cannot fire, and nothing enemy-owned
    is visible near ours. The expected reply class is therefore the network's,
    the same as joe-rs's, and this smoke keeps testing what it always tested
    once the tactics layer lands.
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
    bot_id="unclejoe",
    binary="unclejoe",
    bot_dir=BOT_DIR,
    source_trees=("src",),
    # The file that ships is joe-rs's safetensors conversion, byte-copied here
    # by tools/sync_artifact.py and named and digested by its own keys.
    # `weights_sha256` in this manifest is the **`.eqx` checkpoint**, which is
    # not in the zip — reading it here would put a digest in SUBMISSION.json
    # that a human compares against a file that is not there.
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
        "unclejoe is a fork of joe-rs and carries the same weights by byte "
        "copy, so `safetensors_sha256` equals joe-rs's for the same "
        "checkpoint; the programs differ, and their content hashes do too. "
        "`safetensors_sha256` digests model.safetensors, the weights this zip "
        "actually carries; `source_eqx_sha256` is the .eqx checkpoint the "
        "conversion read, which is provenance of that conversion and not of "
        "anything shipped here. `minified` means the .rs members were stripped "
        "of comments on the way in — the program is the same, the line numbers "
        "are not, so a judge traceback locates a fault in the stripped file. "
        "Re-package with --no-minify when you need to read one."
    ),
    smoke_input=SMOKE_INPUT,
    # Two frames, two well-formed replies, at least one of them a move: with
    # a seat that degrades to passing rather than exiting, the line count
    # alone cannot catch a broken artifact, so both halves of that detector
    # stay on — this one in the packager, `selfcheck` at intake.
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
    # joe-rs, not joe: the fork's baseline is the bot it forked, and a gate
    # against it is also the cheapest read on whether the two still agree.
    gate_opponent_default="joe-rs",
)


if __name__ == "__main__":
    raise SystemExit(main(SPEC))
