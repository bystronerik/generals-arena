# joe-rs build and packaging

**Status: built, 2026-08-14. J5.0 through J5.5 have landed; J5.6 is still
deferred by choice (§5.3).** This is milestone J5 of the
[port plan](port-plan.md). Sections below are description now, not plan; where
a milestone taught something the plan had wrong, the correction is in place and
labelled.

The work was not "write joe-rs a packager". It was "extract the morpheus-rs
packager into shared logic and give both bots a spec", because a second
hand-written copy of that file would drift from the first within one change to
the judge's limits.

Every number below is labelled **measured** or **assumed**. The measured ones
were taken on 2026-08-14, macOS arm64, `cargo 1.97.1`, against the working tree
at `joe-rs@6d59eaa57e5f`, plus the Linux x86 figures from J5.4.

What exists:

```
arena/rust_bundle.py                          shared: the whole packager
bots/morpheus-rs/tools/package_submission.py  spec + delegate
bots/joe-rs/tools/package_submission.py       spec + delegate
bots/joe-rs/tools/submission/build.sh         new
tools/rust-minify/                            moved out of morpheus-rs
tests/test_rust_bundle.py                     moved out of morpheus-rs, extended
scripts/joe_rs_modal_submission_smoke.py      new
```

---

## 1. Where the brief and the code disagree

Three corrections, found by reading the code. The code wins.

**joe-rs does not degrade silently at startup.** The brief's central premise —
"a seat that cannot load its weights passes every turn instead of dying, so a
well-formed reply is not evidence" — is true of morpheus-rs and **false of
joe-rs**. [`main.rs:177`](../../../bots/joe-rs/src/main.rs:177) constructs the
seat with `Seat::new(...)?`, so a load failure propagates out of `wire_main`,
`main` prints `[joe-rs] fatal:` and calls `exit(1)`. A joe-rs bundle with
`artifact/` deleted answers **zero** frames, not two well-formed skips. The
packager's protocol smoke therefore already detects the failure morpheus needed
`selfcheck` to catch. What stays silent in joe-rs is the *per-turn* path: an
error or panic inside `act` becomes `PASS`
([`main.rs:189-203`](../../../bots/joe-rs/src/main.rs:189)). This changes the
selfcheck decision in §5.3 — it is no longer forced.

**`joe-rs` has three subcommands, not one.** The brief says `parity <surface>`
only; [`main.rs:257`](../../../bots/joe-rs/src/main.rs:257) dispatches
`bench` and `parity`, plus the no-argument wire seat. `bench` is what
[latency.md](latency.md) measures with.

**A shell reference bypasses `_SKIP_DIRS` entirely.** The brief warns that a
bot-relative path in a shell *comment* pulls that file into the rated closure.
It is worse than that:
[`fingerprint.py:212-216`](../../../arena/records/fingerprint.py:212) appends
the shell-referenced candidate without passing it through `_is_source`, so a
comment naming `joe-rs/tools/package_submission.py` in any `.sh` under `bots/`
would hash a `tools/` file that the directory walk deliberately skips. The rule
is not "avoid sibling bots" but "no `bots/`-relative `.sh` or `.py` path in any
shell source under `bots/`, comment or code".

Confirmed rather than assumed: neither bot's closure contains a `tools/` or
`tests/` path today, so relocating tooling is free.

```bash
.venv/bin/python -c "
from pathlib import Path
from arena.records.fingerprint import bot_source_closure, relative_label
for b in ('joe-rs','morpheus-rs'):
    c=[relative_label(p) for p in bot_source_closure(Path('bots')/b)]
    print(b, [x for x in c if '/tools/' in x or '/tests/' in x])"
```

---

## 2. The budget, measured before anything is built

`cargo vendor` over joe-rs's real graph, then a zip assembled in the exact
submission shape (generated `run.sh` + `build.sh` + `.cargo/config.toml` +
`SUBMISSION.json`, `Cargo.toml`, `Cargo.lock`, `src/*.rs`, `artifact/`,
`vendor/`), fixed member dates, `ZIP_DEFLATED` per member — the same writer the
packager uses.

| limit | predicted | **built (J5.3)** | cap | used |
| --- | ---: | ---: | ---: | ---: |
| **zip bytes** | 42,191,512 B | **42,183,023 B (40.23 MiB)** | 50 MiB | **80.5 %** |
| files | 3,961 | **3,961** | 10,000 | 39.6 % |
| unpacked bytes | 86,323,580 B | **86,293,389 B (82.30 MiB)** | 512 MiB | 16.1 % |

The pre-build estimate came in 8,489 bytes (0.02 %) heavy and the file count
exactly right, so the rest of this section stands as written.

**The zip cap binds, and the file count does not.** That inverts R4 as written
in the port plan and inverts the morpheus-era note that "the file count is the
binding one for a Rust bot". `vendor/` is 93 crates and 3,946 files but only
**9.91 MiB deflated**; `artifact/model.safetensors` is one file that deflates
32.65 MiB → **30.29 MiB (ratio 0.928)**, because float32 weights are close to
incompressible. Three quarters of the archive is the artifact.

Headroom: **9.76 MiB**. Under the stricter reading of "50 MB" as 50·10⁶ bytes
(47.68 MiB) rather than the 50·2²⁰ that `arena.bundle.MAX_ZIP_BYTES` enforces,
headroom is 7.44 MiB. The bundle passes either reading; the ambiguity only
matters if the artifact grows.

Heaviest vendored crates (files / unpacked / deflated):

| crate | files | unpacked | deflated |
| --- | ---: | ---: | ---: |
| `windows-sys-0.61.2` | 258 | 17.37 MiB | 2.56 MiB |
| `libc-0.2.189` | 404 | 4.33 MiB | 1.02 MiB |
| `zerocopy-0.8.56` | 316 | 1.63 MiB | — |
| `candle-core-0.9.2` | 105 | 1.51 MiB | — |

`windows-sys` is 35 % of the unpacked bytes and compiles on no target we ship
to, which makes pruning it look attractive. **It is not an option.** Under
source replacement cargo requires every package in the resolve graph to exist
under `vendor/`, and each vendored crate carries a `.cargo-checksum.json` over
its own files — so deleting a crate breaks `--locked` resolution and deleting
files inside one breaks checksum verification. It would buy 2.56 MiB of the
9.76 MiB headroom in exchange for an offline build that cannot resolve. Recorded
as measured-and-rejected, not overlooked.

Reproduce:

```bash
cargo vendor --versioned-dirs --manifest-path bots/joe-rs/Cargo.toml /tmp/joe-vendor
```

The morpheus packager's `_vendor` comment — "Zero is the good answer and the
reason the crate has no dependencies" — describes morpheus's situation, not a
rule. It must not survive into the shared module.

---

## 3. The shared / per-bot split

[`arena/rust_bundle.py`](../../../arena/rust_bundle.py) sits beside
[`arena/bundle.py`](../../../arena/bundle.py), which already owns the judge's
limits, the `<bot_id>-<content_hash>.zip` naming and `_ZIP_DATE`, and it is
outside every bot's content hash by construction — the closure walk never leaves
`bots/`. `bots/_common/` is the one place it must *not* go: that directory is
inside the Python bots' closures.

`bots/morpheus-rs/tools/submission/build.sh` is unchanged, byte for byte. Each
`package_submission.py` is its spec and `raise SystemExit(main(SPEC))`.

### `RustBotSpec` (frozen dataclass)

| field | morpheus-rs | joe-rs |
| --- | --- | --- |
| `bot_id`, `binary` | `morpheus-rs` | `joe-rs` |
| `source_members` | `("Cargo.toml", "Cargo.lock")` | same |
| `source_trees` | `("crates",)` | `("src",)` |
| `config_members` | `("deployment.json",)` | `()` |
| `artifact_dir` | `"artifact"` | `"artifact"` |
| `artifact_file_key` | `"artifact_file"` | `"safetensors"` |
| `artifact_sha_key` | `"weights_sha256"` | `"safetensors_sha256"` |
| `run_sh` | current `RUN_SH_VENDORED` | §5.4 |
| `cargo_config` | current `CARGO_CONFIG_VENDORED` | same text (a shared default) |
| `build_sh` | `tools/submission/build.sh` | `tools/submission/build.sh` |
| `selfcheck_argv` | `("selfcheck",)` | `None` (§5.3) |
| `selfcheck_keys` | current nine-key set | — |
| `smoke_input`, `smoke_expected_lines` | current | §5.4 |
| `smoke_reject_all_pass` | `False` | `True` |
| `provenance_fields` | `{"weights_sha256": "weights_sha256", "checkpoint": "training_run.checkpoint_id"}` | §5.4 |
| `provenance_note` | current text verbatim | new text |
| `minify_bin` | `tools/rust-minify` | same |
| `gate_opponent_default` | `cm_expander` | `joe` |

`rust-toolchain.toml` appears in no field on either side. That is the whole
mechanism keeping it out of the zip, and §7 makes it a test rather than a
comment.

One qualification found by inspecting the built zip: joe-rs's archive **does**
contain four `vendor/*/rust-toolchain.toml` files, copied out of `proc-macro2`,
`quote` and both `thiserror` majors by `cargo vendor`. They ask for
`components = ["rust-src"]` and name no channel, and they are inert — rustup
resolves a toolchain file by walking *up* from the working directory, and
`build.sh` runs from the bundle root, above all of them. J5.4 settles it
empirically: the offline container has only 1.97.1 and no network, so a honoured
`rust-src` request would have failed the build. The rule is unchanged, but it is
about the bot's own pin, not about the string appearing in the archive.

Four fields the plan did not list turned out to be needed. `bot_dir` anchors
every other path and is what lets a test point a spec at a `tmp_path`.
`artifact_dir` names the tree (`"artifact"` on both sides, so it is a default).
`minify_dir` sits beside `minify_bin`, because `_ensure_minifier` has to build
the crate as well as run it. And `build_sh` is stored bot-relative with
`build_sh_path` derived from it, so a spec *can* express the mistake that §7's
test catches — a field that could only ever hold one value would make that test
tautological.

### Functions moved to `arena/rust_bundle.py`

Verbatim, only gaining a `spec` parameter: `strip_line_comments`, `_shell`,
`_ensure_minifier`, `_minify_rust`, `_artifact_members`, `_iter_source_files`,
`_write_entry`, `_vendor`, `build_vendored`, `_provenance`, `_selfcheck_facts`,
`smoke`, `package`, `gate`, `main`, plus `PackageError`, `PROVENANCE_NAME` and
`_ZIP_DATE` (re-exported from `arena.bundle`, not redefined).

Two gain behaviour:

- `_artifact_members(spec)` reads the manifest through
  `spec.artifact_file_key` / `spec.artifact_sha_key` instead of the hardcoded
  morpheus names, and keeps the same failure: a digest mismatch raises.
- `_provenance(spec, …)` builds the common core (`bot_id`, `minified`,
  `content_hash`, `git_commit`, `git_dirty`, `note`) and then merges
  `spec.provenance_fields`, each value a dotted path into the manifest through
  `_dotted`, which returns `""` for any missing hop. That reproduces morpheus's
  old `manifest.get("training_run", {}).get("checkpoint_id", "")` exactly, and
  the strictness it gives up is already covered: `_artifact_members` runs first
  and raises when the manifest lacks the file or digest key. The morpheus key
  set and note text are preserved exactly, because `SUBMISSION.json` is a zip
  member and §6's byte-identity check runs through it.

`gate()` keeps its absolute `PYTHON=<repo>/.venv/bin/python`
([`rust_bundle.py:643`](../../../arena/rust_bundle.py:643)), which is
load-bearing and already correct — a relative value silently BrokenPipes the
Python seat.

`smoke()` gained the `smoke_reject_all_pass` branch and a `selfcheck_argv is
None` skip, and reports a different note in each case: morpheus's quotes the
selfcheck facts, joe's counts how many replies were moves.

### What each bot's `package_submission.py` keeps

The spec, and `raise SystemExit(main(SPEC))`. Nothing else. Both files came out
around 100 lines, most of it the provenance note and the reasoning for the two
or three fields where a wrong value fails silently.

---

## 4. Homes and names

| thing | home | why, in one line |
| --- | --- | --- |
| shared module | `arena/rust_bundle.py` | `arena/` already owns the limits and the naming, and is outside every bot's hash |
| `build.sh` | per-bot `bots/<bot>/tools/submission/build.sh` | keeps morpheus's shipped bytes provably unchanged, and the two scripts genuinely differ (binary name, selfcheck line, bot-specific reasoning in the comments) |
| minifier crate | `tools/rust-minify/`, binary `rust-minify` | a shared packager must not reach into one bot's private tooling; joe-rs depending on `bots/morpheus-rs/tools/` would make morpheus undeletable |

AGENTS.md gained two rows: `Shared Rust dev tooling | tools/` and
`Rust submission packager (shared) | arena/rust_bundle.py + per-bot
bots/<name>/tools/package_submission.py`. `minify_dir` / `minify_bin` are
module-level defaults on the spec rather than per-bot values, so both bots point
at `tools/rust-minify` without saying so.

---

## 5. The five divergence decisions

### 5.1 Vendoring — no shared assumption, a measured spec field

Nothing in the shared module may assume a graph size. `_vendor` returns the file
count and the packager reports it; `arena.bundle.check_limits` is the only
enforcement, unchanged, so the Rust and Python bundles cannot drift. §2 is the
measurement; §8's P1 is the tripwire.

### 5.2 Offline build proof — Linux x86, not the packaging host

`vendor_probe.py` exists because morpheus ships an empty `vendor/`, where
`--offline` cannot fail. joe-rs has 93 crates and real build scripts, so the
probe's whole reason is absent: **joe-rs does not get a vendor probe**, and
`vendor_probe.py` stays morpheus-only.

What replaces it is `scripts/joe_rs_modal_submission_smoke.py`, modelled on
[`morpheus_rs_modal_submission_smoke.py`](../../../scripts/morpheus_rs_modal_submission_smoke.py)
— one variant, `cpu=1, memory=2048, block_network=True`, plus the same
curl-to-crates.io probe proving the block is real. Four things it proves that a
macOS run cannot:

1. **Source replacement against a real graph.** 93 crates resolved from
   `vendor/` with no network, and the build scripts in `libc`, `candle-core`,
   `gemm-*` and `half` compiling and *running* at intake.
2. **The compile flag actually applies.** `.cargo/config.toml` scopes
   `target-cpu=x86-64-v3` to `x86_64-unknown-linux-gnu`, a target a macOS host
   never selects — so a macOS build exercises none of it, and the wrong-cwd
   failure mode that cost morpheus a 49× pessimisation is invisible there.
3. **The 2 GB cap at build time.** `lto = "fat"` with `codegen-units = 1` over
   this graph on one core has an unmeasured peak RSS and an unmeasured wall
   time. RULES.md §08 sets 2 GB per bot and documents no intake timeout; this is
   the only place either can be observed. See P2.
4. **A different toolchain installation than the one that vendored.**

All four were proved at J5.4, and the third came with a caveat the plan did not
anticipate: Modal's `cpu=1` reserves a core rather than capping one, so the run
carries a second `-j1` build to bound the wall time honestly.

### 5.3 selfcheck — deferred, and the packager compensates

Because of §1, a joe-rs bundle that cannot load its artifact fails the existing
smoke by producing no output. The residual gaps are narrower than morpheus's:
a per-turn error becomes a pass, and a lost `x86-64-v3` produces a correct but
slower bot.

**Decision: no `joe-rs selfcheck` in the first landing.** Instead the spec sets
`smoke_reject_all_pass = True` and ships a smoke frame on a real 21×21 board
where a pass is the wrong answer, so the packager fails a bundle whose every
reply is `1 0 0 0 0`. The sha256 of `model.safetensors` is verified against the
committed manifest at pack time (§3), which covers the artifact-substitution
half.

**What that leaves uncovered, stated plainly:** no intake-time refusal. A
bundle broken in a way that only bites on the judge's machine — a build that
lost the target flag, a per-turn error path — is rejected by nothing, and
`build.sh` is a build step rather than a gate. Morpheus's own history says this
matters: an aborted build once shipped a four-file zip that only `selfcheck`
caught.

**The cost of adding it, which is why it is deferred:** `selfcheck` is a new
arm in `main.rs`, which is in the rated closure — so it moves joe-rs's content
hash, mints a new `data/bot_versions/joe-rs.json` version, and invalidates the
J4 gate and A/B verdict for the packaged program. The shared module therefore
treats selfcheck as pure configuration (`selfcheck_argv=None` skips the step and
the parse), so turning it on later is a spec edit plus a Rust subcommand, not a
packager change. Milestone J5.6 does exactly that, and it should be batched with
whatever other source change precedes a real submission so the hash moves once.

### 5.4 Per-bot shape — the joe-rs values

**Generated `run.sh`.** The zip's launcher **sets `JOE_RS_ARTIFACT`
explicitly**:

```bash
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1
DIR="$(cd "$(dirname "$0")" && pwd)"
export JOE_RS_ARTIFACT="$DIR/artifact"
exec "$DIR/target/release/joe-rs"
```

The exe-relative fallback in
[`artifact_dir()`](../../../bots/joe-rs/src/main.rs:43) *would* also resolve in
the submission layout — `target/release/joe-rs` walks up three parents to the
bundle root, where `artifact/` sits — but its last resort is a bare relative
`PathBuf::from("artifact")`, which depends on the judge's cwd, and
`current_exe()` can fail. One `export` line costs nothing and removes both.
**Both paths get proved**, not argued: J5.3's done-when runs the extracted
bundle twice, once with the launcher and once with `JOE_RS_ARTIFACT` unset.

**Smoke.** The morpheus script is a 2×2 board; joe-rs's net pads to 21 and
`Seat::new` refuses anything larger, so a 21×21 handshake with a general on
plentiful army and empty neighbours is both legal and the frame where
`smoke_reject_all_pass` has meaning. Two frames, two well-formed replies, at
least one not a pass. Built by `_smoke_frame` in the spec rather than written
out: two 21×21 boards are about 5 KB of literal, and the reason they cannot come
from the committed parity fixture instead is in J5.3. Measured replies:
`0 10 10 0 0` and `0 10 10 2 0`, the same on macOS arm64 and Linux x86.

**Provenance.** The two manifests are different schemas, and getting this wrong
puts a digest in `SUBMISSION.json` that a human will compare against the wrong
file. joe-rs's `weights_sha256` is the **`.eqx` checkpoint, which is not in the
zip**; the shipped file is named by `safetensors` and digested by
`safetensors_sha256`; provenance lives under `checkpoint`, not `training_run`.
So:

```python
provenance_fields = {
    "safetensors_sha256":  "safetensors_sha256",   # the file in this zip
    "source_eqx_sha256":   "weights_sha256",       # NOT in this zip
    "checkpoint":          "checkpoint.run_name",
    "checkpoint_step":     "checkpoint.global_step",
    "engine_sha":          "checkpoint.engine_sha",
    "tensor_schema":       "tensor_schema",
}
```

No key called `weights_sha256`. The note text says in one sentence that the
`.eqx` digest is the provenance of the conversion and not of anything shipped.

**Artifact verification.** [`net.rs:142`](../../../bots/joe-rs/src/net.rs:142)
schema-checks the tensor names, shapes, dtypes and the manifest's network block,
but never hashes the file — so the digest check belongs to the packager, which
already has one in `_artifact_members`. See P3: `artifact/*.safetensors` is
gitignored, so `git_dirty` cannot see a stale artifact, and this digest is the
only thing that can.

### 5.5 `build.sh`

`bots/joe-rs/tools/submission/build.sh`, deliberately **not** beside `run.sh` —
`matchup.py::build_agent` executes any `build.sh` it finds next to a `run.sh`
and then crashes formatting `build.relative_to(REPO_ROOT)` for a path outside
the submodule. Same body as morpheus's minus the selfcheck line: `cd "$DIR"`
before `cargo build --release --offline --locked` (cargo discovers
`.cargo/config.toml` from the working directory, not the manifest), then the
`ls -l` line. The gate deletes it after intake, as morpheus's does.

---

## 6. Milestones — what happened

Ordered so that the refactor is proved byte-for-byte before joe-rs is added, and
so nothing that moves a content hash happens before the measurement steps.

### The correction that runs through J5.0–J5.2: a raw digest cannot be the test

The plan's criterion was "the zip's sha256 does not move". **It cannot be met
by any sequencing, and this is a flaw in the plan rather than in the tooling.**
`SUBMISSION.json` is a zip member and carries `git_commit` and `git_dirty`;
refactoring dirties `bots/morpheus-rs`, and committing first moves the commit
instead. Either way the raw digest changes for a reason with nothing to do with
the refactor. P6's list of three hiding places — the key set, the note text,
member ordering — missed this fourth.

What replaced it keeps every bit of P6's intent: a **normalized digest** over
the ordered `(arcname, mode, sha256(member))` triples with those two fields
blanked. It still pins member ordering, member modes, every source and vendored
byte, the minifier's output, and the provenance key set and note text. The raw
digest is recorded beside it.

### J5.0 — Baseline the byte-identity claim

```bash
.venv/bin/python bots/morpheus-rs/tools/package_submission.py --force --no-smoke \
  && shasum -a 256 data/bundles/morpheus-rs-3f06212b5532.zip
```

*Result:* the raw digest of a fresh package is
`a15e09b5bf75d04ae01ae6a0657f335aab2f81deb5c5e00e14e239296e3bf91c`, **not** the
`7a504229…` the committed zip carries. Diffing the two archives member by member
showed the entire difference to be `SUBMISSION.json`: the committed zip was made
at commit `63b34e5` with a dirty tree, the fresh one at `9bce318` clean, and the
one-byte size change is `false` → `true`. Every other member is byte-identical.

So nothing had drifted, and the baseline is:

| | |
| --- | --- |
| normalized | `283cdeb18d4b3765064d54918378908db3c387d31a6121e6f4b2c34d331e6e97` |
| raw, at clean `9bce318` | `a15e09b5bf75d04ae01ae6a0657f335aab2f81deb5c5e00e14e239296e3bf91c` |

Reproducible: two packaging runs at the same git state produced the same raw
digest.

### J5.1 — Extract `arena/rust_bundle.py`

Move the functions of §3; morpheus's `package_submission.py` becomes spec plus
delegate. No behaviour change, no new fields exercised. The spec kept a
temporary `minify_dir` / `minify_bin` override pointing at the old location, so
that the extraction and the rename each got their own check and a moved digest
would say which one moved it.

*Result:* normalized digest `283cdeb1…`, unchanged. Content hash still
`3f06212b5532`. The raw digest moved to `e53c2f58…` — one byte smaller, which is
`git_dirty` flipping to `true`, exactly the mechanism above.

### J5.2 — Move the minifier (separable)

`bots/morpheus-rs/tools/minify/` → `tools/rust-minify/`, binary renamed
`rust-minify`, the override dropped, `.gitignore` retargeted, AGENTS.md given
its row.

*Result:* normalized digest `283cdeb1…` and raw `e53c2f58…`, both unchanged from
J5.1 — the rename changed the tool's address and not its output. Content hash
still `3f06212b5532`, and neither bot's closure contains a `tools/` or `tests/`
path.

One thing was deliberately **not** fixed at the time: a comment at
`bots/morpheus-rs/Cargo.toml:15` still named `tools/minify`. `Cargo.toml` is
inside morpheus's rated closure, so correcting even a comment there moves the
content hash — which P4 forbids for J5.1–J5.5.

**It was corrected afterwards, as a decision taken with the cost known** (see
§10). J5.1–J5.5 all ran and were verified at `3f06212b5532`, so nothing in this
document's results depends on the correction; every digest and hash quoted for a
milestone is the one that milestone actually produced.

### J5.3 — joe-rs spec, `build.sh`, and the first real bundle

```bash
.venv/bin/python bots/joe-rs/tools/package_submission.py --force
```

*Result:* `joe-rs-6d59eaa57e5f.zip`, 42,183,023 B in 3,961 files, all three
limits clear, smoke green — "replied 2 well-formed actions, 2 of them a move",
which is `smoke_reject_all_pass` doing its job rather than passing vacuously.
Two packaging runs produced the same `zip_sha256` (`b1efb600…`), so joe's bundle
is byte-reproducible like morpheus's.

**The smoke frames had to be built, not sliced.** The obvious source for a real
21×21 position is the committed parity fixture, and it is the wrong one: the
first 22 turns of `smoke-aegis-seed0.out.log` are *all* `1 0 0 0 0`, because the
general has not accumulated army yet. A fixture prefix would have failed a
healthy bundle. The synthetic frame in §5.4 draws `0 10 10 0 0` and
`0 10 10 2 0` from the real binary.

**The plan's second command proves nothing, on two counts.** `env -u
JOE_RS_ARTIFACT ./run.sh` cannot reach the fallback, because the generated
`run.sh` exports the variable itself; and `< /dev/null` means `read_handshake`
hits EOF and `wire_main` returns *before* `Seat::new`, so the artifact is never
resolved at all — the run prints nothing and exits 0 whether the artifact is
there or not. What actually exercises the branch is the binary directly, with a
handshake on stdin:

```bash
B=data/bundles/extracted-joe-rs-6d59eaa57e5f
cd "$B" && bash build.sh
printf '0 21 21\n' | ./run.sh                                    # launcher path
cd / && printf '0 21 21\n' | env -u JOE_RS_ARTIFACT "$PWD/$B/target/release/joe-rs"
cp "$B/target/release/joe-rs" /tmp/joe-rs-orphan                 # control
cd / && printf '0 21 21\n' | env -u JOE_RS_ARTIFACT /tmp/joe-rs-orphan
```

*Result:* both paths start the seat (`[joe-rs] load 43.4 ms, warmup 15.8 ms`,
exit 0, no `fatal:`), from `cwd=/` as well as from the bundle root. The control
matters: the same binary copied away from its bundle, with the variable unset,
dies with `[joe-rs] fatal: read manifest.json: No such file or directory` and
exit 1 — so the two successes came from the exe-relative branch and not from an
accident of the working directory.

### J5.4 — Offline build on Linux x86

[`scripts/joe_rs_modal_submission_smoke.py`](../../../scripts/joe_rs_modal_submission_smoke.py),
one variant, `cpu=1`, `memory=2048`, `block_network=True`.

```bash
.venv/bin/modal run scripts/joe_rs_modal_submission_smoke.py
```

Per AGENTS.md, check the job's own output a few minutes in and confirm it got
past import — a Modal job that dies at import looks exactly like one that works.
The `modal.is_local()` guard is what that failure mode would come from here: the
module is re-imported *inside* the container, where `arena` does not exist.

*Result:* green. `rustc 1.97.1`, x86_64, kernel `4.19.0-gvisor`, curl to
crates.io exits 6 (`Could not resolve host`), so the offline claim is proved
rather than asserted. 93 crates resolved from `vendor/` with no network, build
scripts running at intake, and the binary answering the same two frames it
answers on macOS — `0 10 10 0 0` and `0 10 10 2 0`, bit-identical decisions
across architecture and toolchain installation.

| cargo jobs | exit | wall s | peak RSS |
| --- | ---: | ---: | ---: |
| default | 0 | 74.97 | 1,288 MiB |
| `-j1` | 0 | 129.21 | 1,616 MiB |

**Two builds, because `cpu=1` is not a cap.** The container reports
`cpu.max` = **17.00 cpu** and `os.cpu_count()` = 17: Modal's `cpu=1` reserves a
core, it does not confine to one. A wall time measured under default
parallelism is therefore not the one-core number RULES.md §08 describes, and
reporting it as one would have understated the judge's build. `CARGO_BUILD_JOBS=1`
puts a bound under it. Even that is a bound on *cargo's* process parallelism and
not a true single core — rustc keeps its own threads — so 129 s is conservative
in one direction and optimistic in the other, and it is 8.6× inside the 15-minute
tripwire either way.

The RSS column is `ru_maxrss` over `RUSAGE_CHILDREN`, which is a running maximum
across every child the container has reaped — so the `-j1` figure is the
high-water mark of the whole run, not of that build alone. The cgroup counters
(`memory.peak`, `memory.max_usage_in_bytes`) do not exist under gVisor, which is
why this is the number available. With `lto = "fat"` and `codegen-units = 1` the
peak is the final single-threaded link, so the largest child is close to the
whole.

### J5.5 — The competition gate with the submitted bytes

```bash
.venv/bin/python bots/joe-rs/tools/package_submission.py --force --gate --gate-opponent joe
```

*Result:* `gate.returncode` 0, the match running to turn 139 and ending
normally. The gate's `PYTHON` is the absolute `<repo>/.venv/bin/python` the
shared `gate()` already sets; the Python `joe` seat dies silently without it.
The value telemetry in the tail (`+0.55` at turn 131 down to `-0.96` at 139) is
joe-rs losing that particular game, which the gate does not care about — a
normal end is the whole criterion.

### J5.6 — `joe-rs selfcheck` — **not done, still deferred**

Add the subcommand mirroring morpheus's shape — resolve the artifact
explicitly, verify `safetensors_sha256`, construct the real `Seat`, decide one
hand-built frame where a pass is wrong, print `key value` lines, exit non-zero
on any failure — set `selfcheck_argv=("selfcheck",)` in the spec, and add the
line to joe-rs's `build.sh`.

*Done when:* the packager's smoke reports `selfcheck ok`; **and** joe-rs is
re-registered at its new content hash with the §7-of-the-port-plan gate re-run,
because this milestone re-identifies the bot.

---

## 7. Test plan

**Measured constraint: the warm default suite was at 15.24 s** (715 passed, 429
deselected), i.e. already at the AGENTS.md ceiling of 15 s. Fixing that was out
of scope here; the binding requirement was that this work add no measurable
time. Every test below is string-, dict- or path-level: no cargo, no zip
writing, no subprocess.

*Result:* **726 passed, 429 deselected in 13.09 s** warm. The new file is 17
tests and 0.70 s of that, most of it import cost it shares with the rest of the
suite.

**Moved.** `bots/morpheus-rs/tests/test_packaging.py` →
`tests/test_rust_bundle.py`, in full. Its five `strip_line_comments` cases now
guard shared code, and a test that guards shared code from inside one bot's
directory is how drift starts. `test_the_shipped_launcher_still_does_what_it_did`
becomes parametrized over both specs. Deleting the old file moves no hash —
`tests` is in `_SKIP_DIRS`, confirmed in §1.

**Added**, all in `tests/test_rust_bundle.py`:

| test | what it pins |
| --- | --- |
| `test_the_toolchain_pin_never_ships` | `rust-toolchain.toml` in no spec's `source_members`/`source_trees`, both bots |
| `test_build_sh_never_sits_beside_run_sh` | `spec.build_sh_path.parent != spec.bot_dir` — the `matchup.py::build_agent` trap as an assertion instead of a comment |
| `test_the_joe_launcher_exports_the_artifact_dir` | generated `run.sh` exports `JOE_RS_ARTIFACT`, execs `target/release/joe-rs`, 7 exports, survives `strip_line_comments` |
| `test_manifest_keys_are_read_per_spec` | fake manifests in `tmp_path` for both schemas; the right file/digest keys are used, and a digest mismatch raises `PackageError` |
| `test_joe_provenance_does_not_call_the_eqx_digest_weights` | no `weights_sha256` key in joe-rs's `provenance_fields`; `source_eqx_sha256` is present |
| `test_dotted_manifest_paths_resolve_or_blank` | `_dotted` walks nested dicts and returns `""` for any missing hop, including a leaf that is not a dict |
| `test_archives_are_named_for_the_program_they_hold` | parametrized over both specs (was morpheus-only) |

`test_manifest_keys_are_read_per_spec` earns its place by construction: joe's
fake manifest carries a *decoy* `weights_sha256` of sixty-four zeros beside the
real `safetensors_sha256`, under the very name morpheus uses for the file it
ships. Reading the wrong key raises, so the test fails loudly rather than
silently agreeing with itself.

**Not added to pytest.** Packaging a real bundle costs a `cargo vendor` plus a
release build of 93 crates. That belongs to the J5.3 command, not the suite.
Nothing here carries the `joe` or `morpheus` marker, because nothing here needs
a binary.

*Result:* `.venv/bin/python -m pytest -q` — 726 passed, 429 deselected, 13.09 s.

---

## 8. Risks and tripwires

**P1 — the zip cap, at 80.5 %.** Measured on the built bundle, and the binding
limit: 42,183,023 B of 52,428,800. 30.29 MiB of the 40.23 MiB is incompressible
float32.
*Tripwire:* any packaging run reporting `zip_bytes` > 47,185,920 (45 MiB), or
any artifact re-export that grows `safetensors_size`.
*What is left:* not vendor pruning (§2 — it breaks offline resolution for
2.56 MiB). The real levers are a smaller dtype or a smaller net, both of which
change the bot and belong to a port-plan milestone, not to packaging. Assumed:
that the net stays at 8,556,250 float32 parameters.

**P2 — intake build under one core and 2 GB. Measured at J5.4, and memory is
the half that stayed tight.** Wall time is comfortable: 129 s with
`CARGO_BUILD_JOBS=1`, against a 15-minute tripwire. Peak RSS is **1,616 MiB
against the 2,048 MiB cap — 79 %**, which is a much smaller margin than the
wall time and is the number to watch.
*Tripwire, revised:* a J5.4 run whose peak RSS exceeds 1,850 MiB, or whose
`-j1` build exceeds 15 minutes. The old tripwire ("the container OOMs") is too
late to be useful — an OOM is the failure, not the warning.
*What moves it:* `lto = "fat"` links the whole graph in one rustc process, so
peak RSS scales with the graph rather than with the net. A dependency addition
is the likeliest thing to spend that 431 MiB.
*Fallback:* `lto = "thin"` — which changes the shipped binary, so the
[latency](latency.md) figure must be re-measured. There is room: measured p99 is
25.4 ms against a 50 ms target and a 150 ms limit.

**P3 — the artifact is gitignored and already stale.** Measured: the registered
version `joe-rs@0995d03a59c8` in `data/bot_versions/joe-rs.json` differs from
the working tree at `bots/joe-rs/artifact/manifest.json` and
`model.safetensors`; the tree now hashes to `6d59eaa57e5f` (the step-6000
export), while [the A/B sanity check](../../research/measurements/joe-rs-ab-sanity.md)
was run against `joe-rs@0995d03a59c8` (step 5000). `.gitignore` excludes
`bots/joe-rs/artifact/*.safetensors`, so `git_dirty` in `SUBMISSION.json` cannot
see this at all.
*Tripwire:* the packager's `safetensors_sha256` check failing, which is the only
automatic detector — keep it fatal.
*Consequence for J5, stated rather than fixed:* the program that would be
packaged today is not the one the J4 verdict covers. Registering and re-running
the gate is a submission precondition, not a packaging step.

**P4 — `selfcheck` moves the content hash.** §5.3. *Tripwire:* any J5 step other
than J5.6 changing `bot_content_hash(bots/joe-rs/run.sh)` away from
`6d59eaa57e5f`. Held throughout: J5.1–J5.5 all left it untouched, and morpheus's
stayed at `3f06212b5532` for the whole of J5. joe-rs is still at `6d59eaa57e5f`.
morpheus moved afterwards and on purpose — §10.

**P5 — a shell reference widening a closure.** Sharper than documented: §1 shows
shell-referenced paths skip `_is_source` entirely, so a comment could hash a
`tools/` file. The new `bots/joe-rs/tools/submission/build.sh` and both
`package_submission.py` files are exactly the kind of path a comment wants to
name. *Tripwire:* the §1 closure command printing any `/tools/` or `/tests/`
path — run at J5.1, J5.2 and J5.3, empty each time.

**P6 — morpheus byte-drift.** The refactor's one unacceptable outcome, and it
did not happen: the normalized digest is `283cdeb1…` at J5.0, J5.1 and J5.2.
*Tripwire:* the **normalized** digest moving, not the raw one — see §6's
correction. Four places it could hide, not three: `_provenance`'s key set, its
note text, member ordering, and `git_commit` / `git_dirty`, which move on their
own and are what the normalization blanks.

**P7 — the graph moving under the measurement.** §2 is one `Cargo.lock`. A
`cargo update` can change the crate count, and `windows-sys` alone is 2.56 MiB
deflated. *Tripwire:* a packaging run whose `files` or `zip_bytes` moves more
than 5 % from §2 without an intended dependency change.

---

## 9. Measured vs assumed

**Measured** (2026-08-14, macOS arm64, cargo 1.97.1, plus Linux x86 for J5.4):
vendor graph 93 crates / 3,946 files / 49.6 MiB unpacked / 9.91 MiB deflated;
artifact 32.65 → 30.29 MiB deflated; the per-crate table in §2; morpheus's
bundle digests and both bots' content hashes; neither closure containing
`tools/` or `tests/`; joe-rs's registry drift in P3; joe-rs exiting rather than
passing when the seat cannot start.

Added by the milestones:

- the built bundle at 3,961 files / 42,183,023 B zipped / 86,293,389 B unpacked,
  reproducible across two packaging runs (J5.3);
- morpheus repackaging to a byte-identical program at J5.0, J5.1 and J5.2, once
  the live git fields are held constant (§6);
- the offline Linux x86 build: 74.97 s default, 129.21 s at `-j1`, peak RSS
  1,616 MiB against 2,048 MiB, network provably blocked (J5.4);
- identical decisions on macOS arm64 and Linux x86 for the same frames (J5.4);
- both artifact-resolution paths, with a control proving the exe-relative branch
  was the one that fired (J5.3);
- the extracted bundle finishing a competition match against `joe` (J5.5);
- the default suite at 13.09 s / 726 passed (§7).

**Still assumed:** that generals.bot's own sandbox agrees with Modal — the same
proxy caveat morpheus carries, and the only thing a real submission settles.
That the `-j1` bound is conservative enough for a genuinely single-core intake
build (§6, J5.4). And P3's consequence stands untouched: **the program that
would be packaged today is not the one the J4 verdict covers.** Registering
joe-rs at `6d59eaa57e5f` and re-running the port-plan §7 gate is a submission
precondition, not a packaging step.

---

## 10. After J5 — morpheus-rs at `5634e454adbd`, ungated

The stale comment from J5.2 was corrected on 2026-08-14, deliberately and on
its own, rather than batched with a later source change as J5.2 recommended.
Recorded here because it spends something the rest of this document was careful
not to spend.

**What moved.** `bots/morpheus-rs/Cargo.toml` only, and only its comment above
`strip = "symbols"`. The setting itself is untouched, so the compiled program is
identical. The comment was stale twice over: it named `tools/minify` after J5.2
moved that crate, and it justified `strip` as concealment against the
static-binary variant that M8 removed. Since the zip ships sources, and
`tools/rust-minify` strips comments but not identifiers, the concealment
argument was dead independently of the path.

**What it cost.**

| | before | after |
| --- | --- | --- |
| content hash | `3f06212b5532` | **`5634e454adbd`** |
| registry step | seq 4 | seq 5, `git_dirty: false` |
| bundle name | `morpheus-rs-3f06212b5532.zip` | `morpheus-rs-5634e454adbd.zip` |

`data/bundles/morpheus-rs-3f06212b5532.zip` now names a hash that is no longer
the working tree's. It is not stale — it is the bundle of the program J5.0–J5.2
verified — but it is no longer what a fresh package produces.

**⚠ seq 5 has never passed the competition gate.** The gate re-run was
deliberately deferred. `5634e454adbd` is a registered rating identity that no
`--mode competition` match has ever exercised, and AGENTS.md's verification gate
is therefore unsatisfied for it. Before morpheus-rs plays a rated round under
this hash, run:

```bash
.venv/bin/python bots/morpheus-rs/tools/package_submission.py --force --gate
```

The risk is small and is not zero: the change is a comment, but `Cargo.toml` is
a real build input, and "a comment cannot break a build" is exactly the class of
claim the gate exists to stop anyone from having to take on trust. A packaging
run with `--no-smoke` did succeed at the new hash, which proves the manifest
still parses and the archive still assembles — not that the bot still plays.
