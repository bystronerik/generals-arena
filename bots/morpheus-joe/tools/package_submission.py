#!/usr/bin/env python3
"""Package morpheus-joe for generals.bot, and prove the archive before it ships.

    python bots/morpheus-joe/tools/package_submission.py --force
    python bots/morpheus-joe/tools/package_submission.py --force --gate
    python bots/morpheus-joe/tools/package_submission.py --no-minify

Started at milestone M0.5 of docs/bots/morpheus-rs/rewrite-plan.md. The work is
all in `arena/rust_bundle.py`, which every Rust bot here shares; this file is
the spec that says what morpheus-joe is, and nothing else.

**N5 is the phase that qualifies this**, not N1. The spec below is correct as
far as it is exercised — the source trees, the artifact keys and the selfcheck
keys all changed with the port — but no bundle has been built from it yet and
none of the three sandbox caps has been quoted against a real archive. The
projection is ~32 MB in ~85 files, i.e. 64% of the binding 50 MB zip cap and
under 1% of the 10,000-file cap; 34.2 MB of fp32 weights compress to ~31.5 MB,
as joe-rs's own bundle demonstrates (joe-net-plan §8.8).

Three source trees, not one. `crates/joenet` holds byte-identical copies of
joe's files and has to ship for the binary to build; the minifier that strips
comments on the way in does **not** break that byte-identity, because the
digest test compares the files in the repo and not the ones in the zip.

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
"""
from __future__ import annotations

import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arena.rust_bundle import RustBotSpec, main  # noqa: E402

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
exec "$DIR/target/release/morpheus-joe"
"""

# One handshake and two frames on the smallest board the parser will accept.
# Enough to prove the binary speaks the protocol; not a game. A pass here is
# not a wrong answer on a 2x2, which is why `smoke_reject_all_pass` is off and
# `selfcheck` carries the weight instead.
SMOKE_INPUT = "0 2 2\n" + ("1 1 1 1 1\n1 1\n1 1\n1 0\n0 0\n1 0\n0 0\n" * 2)

SPEC = RustBotSpec(
    bot_id="morpheus-joe",
    binary="morpheus-joe",
    bot_dir=BOT_DIR,
    source_trees=("crates",),
    # Required, because it is the only thing that says which bot this is.
    # Without it the binary falls back to the Part 07 placeholders — four times
    # the particles, twice the search depth, a different deadline — and plays a
    # configuration nobody measured.
    config_members=("deployment.json",),
    # joe's manifest names its own keys: `safetensors` for the file and
    # `safetensors_sha256` for its digest. morpheus's `artifact_file` /
    # `weights_sha256` described a converter this bot does not have.
    artifact_file_key="safetensors",
    artifact_sha_key="safetensors_sha256",
    run_sh=RUN_SH_VENDORED,
    provenance_fields={
        "safetensors_sha256": "safetensors_sha256",
        "checkpoint": "checkpoint.run_name",
        "checkpoint_step": "checkpoint.global_step",
    },
    provenance_note=(
        "Rated identity of the program in this zip. The content hash is the "
        "repo bot's, computed over sources + Cargo.lock + run.sh + artifact; "
        "the zip's own launchers are generated and are not part of it. "
        "`minified` means the .rs members were stripped of comments on the "
        "way in — the program is the same, the line numbers are not, so a "
        "judge traceback locates a fault in the stripped file. Re-package "
        "with --no-minify when you need to read one."
    ),
    smoke_input=SMOKE_INPUT,
    smoke_expected_lines=2,
    smoke_reject_all_pass=False,
    selfcheck_argv=("selfcheck",),
    selfcheck_keys=frozenset(
        {
            "hardware_fma",
            "safetensors_sha256",
            "tensor_schema",
            "checkpoint",
            "checkpoint_step",
            "load_ms",
            "warmup_ms",
            "init_ms",
            "decision",
            "decide_ms",
            "selfcheck",
        }
    ),
    gate_opponent_default="cm_expander",
)


if __name__ == "__main__":
    raise SystemExit(main(SPEC))
