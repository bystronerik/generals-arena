# Morpheus-rs build and packaging

How the Rust bot builds in the repo, how it is packaged for generals.bot, and
which of its facts are load-bearing. Established at milestone M0.5 of the
[rewrite plan](rewrite-plan.md), whose whole purpose is to find out whether the
submission path works *before* any of the port is written.

## Layout

```
bots/morpheus-rs/
  Cargo.toml            workspace: crates/core (lib) + crates/bot (bin)
  Cargo.lock            pins the graph; the reason vendor/ need not be hashed
  rust-toolchain.toml   dev-side pin, deliberately NOT shipped (below)
  .cargo/config.toml    target-cpu for the Linux targets only
  crates/core/src/      wire protocol today; the port lands here
  crates/bot/src/       main: read frames, decide, reply
  run.sh                repo launcher — builds when stale, then execs
  tools/                dev tooling; outside the content hash
    submission/build.sh the build.sh copied into the vendored zip
    package_submission.py
    vendor_probe.py     proves the offline build with real vendored crates
    dependency_budget.py what a crate would cost in the zip
    capture_morpheus.py the M0 capture module
  tests/                pytest parity slice; outside the content hash
  target/  vendor/      build output and vendored crates; gitignored, unhashed
```

## Toolchain

`rustc`/`cargo` **1.97.1**, matching the sandbox's 1.97 stable. Installed with
rustup; `rust-toolchain.toml` pins the channel and carries the
`x86_64-unknown-linux-musl` target for the fallback variant.

## Content hash

`bots/morpheus-rs/` is rated on its sources: the `.rs` files, `Cargo.toml`,
`Cargo.lock`, `rust-toolchain.toml`, `.cargo/config.toml`, and `run.sh`.
`fingerprint._SKIP_DIRS` gained `target`, `vendor`, and `tools` for this — the
first two because build output is what sources compile *to* and vendored
crates are a copy of crates.io the lock file already pins, the third for the
same reason `probe.py` is excluded: arena-owned tooling that never plays must
not fork a rating identity when it is edited.

**A shell comment can widen the closure.** `_SHELL_REF_RE` scans shell sources
for bot-relative paths and cannot tell a comment from a `source` line. An early
`run.sh` mentioned the Python bot's launcher in prose, which put
`bots/morpheus/run.sh` inside morpheus-rs's hash — so editing Python morpheus
would have silently re-identified the Rust bot. Refer to sibling bots by
description, not by path, in any `.sh` under `bots/`.

## Two variants, both built every time

```bash
python bots/morpheus-rs/tools/package_submission.py --force
```

**vendored** — sources, `Cargo.lock`, `vendor/`, and a `build.sh` running
`cargo build --release --offline --locked`. The intended submission: the judge
compiles it at intake. Note `vendor/` is currently **empty**, because the crate
has no dependencies — which is why the probe below exists.

**static** — a prebuilt `x86_64-unknown-linux-musl` binary with `build.sh`
reduced to a no-op. This is R4's fallback (§9) for a sandbox that cannot build
the vendored tree. It is built and smoked on every packaging run because a
fallback nobody exercises is a fallback nobody can rely on.

Both are audited through `arena.bundle.check_limits` — the same code that
guards the Python bundles, so the two cannot drift — then extracted to a
scratch directory with no repo around them, built, and driven with a scripted
frame.

## The offline build is proven by a probe, not by the shipping zips

The two shipping variants have no dependencies. That makes their
`cargo build --offline` cheap to pass and nearly meaningless: with nothing to
resolve, `--offline` cannot fail. It never exercises `.cargo/config.toml`
source replacement, a transitive graph, or a build script running at intake —
and M3's inference crate brings ninety crates of exactly that.

`tools/vendor_probe.py` closes the gap without putting a dependency in the
shipped binary. It builds a scratch crate that depends on `sha2` — chosen for
its shape: a transitive chain (`digest` → `block-buffer` → `generic-array` →
`typenum`) and a build script via the `version_check` build-dependency, which
is the failure mode most likely to be missed. It vendors, packages in exactly
the submission shape, builds offline locally, and rides the Modal smoke as a
third variant.

```bash
python bots/morpheus-rs/tools/vendor_probe.py
modal run scripts/morpheus_rs_modal_submission_smoke.py
```

Result: 571 files, 6.3 MB unpacked, builds in 2.5 s inside a `block_network`
container on the sandbox toolchain, and answers the protocol.

## What a dependency costs

```bash
python bots/morpheus-rs/tools/dependency_budget.py --crate tract-onnx=0.21
```

Measured, because the plan's §1 originally asserted that `cargo vendor` "blows
past 10k files easily" and that assertion had already been used to justify
hand-writing SHA-256. It is not true — `sha2` is 5.6% of the cap. Current
figures live in
[morpheus-rs-dependency-budget.md](../../research/measurements/morpheus-rs-dependency-budget.md);
the budget binds at M3's inference crate, not before.

## Things that would have broken the submission

**`rust-toolchain.toml` must not ship.** It pins an exact patch release. A
sandbox whose stable toolchain is any other patch would see the pin and try to
*download* the pinned one, over a network that does not exist at intake. The
pin is a development-side guarantee that nothing newer than 1.97 gets used;
shipping it converts that guarantee into a build failure.

**`build.sh` must not sit beside `run.sh` in the repo.** `matchup.py`'s
`build_agent` runs any `build.sh` it finds next to a `run.sh`, then crashes
formatting its log line — `build.relative_to(REPO_ROOT)` against the
submodule's root, unguarded, where the identical call in `spawn_agent` twenty
lines below *is* guarded. So a `build.sh` in the bot directory breaks the
repo's own verification gate. It lives in `tools/submission/` and is copied
into the zip. (A one-line submodule fix would work too; AGENTS.md prefers
wrapping, and here the plan's own layout already agrees.)

**Cross-linking musl from macOS needs `rust-lld`.** Apple's `ld` cannot emit
ELF, and the failure is a wall of "unknown options". The packager points rustc
at the `rust-lld` that ships in the toolchain, discovering the host triple from
`rustc -vV` — `platform.machine()` says `arm64` where Rust's triple says
`aarch64`, close enough to look right and wrong enough to miss the file. On an
x86 Linux host no override is used at all.

## Compile target

`x86-64-v3`, from the [M0 CPU probe](../../research/measurements/morpheus-rs-cpu-probe.md):
the floor across sampled one-core Linux containers, which included an AMD
generation without AVX-512. Set per-target in `.cargo/config.toml` so it
applies to the Linux builds and not to arm64 macOS development. Any
hand-written SIMD path must still detect features at runtime — the sandbox
cannot be probed, so the target is an informed guess about a machine nobody
has measured.

## Panic policy

`panic = "unwind"`, and the decision runs inside `catch_unwind`. The judge
forfeits a game on a crash or an early exit but charges one fault out of fifty
for a bad reply (RULES.md §08), so a panicking turn should cost a fault, not
the match. `panic = "abort"` would be the reflexive release-profile choice and
is the wrong trade here.

## Verification

```bash
cargo test --release --manifest-path bots/morpheus-rs/Cargo.toml
python bots/morpheus-rs/tools/package_submission.py --force
modal run scripts/morpheus_rs_modal_submission_smoke.py
```

The last one extracts both zips inside a one-core Linux x86 container with
`block_network=True` and the sandbox's own toolchain, builds them, and speaks
the protocol to each. Results:
[morpheus-rs-sandbox-smoke.md](../../research/measurements/morpheus-rs-sandbox-smoke.md).
It is a proxy, not the judge — necessary, not sufficient, which is the whole
reason the fallback variant exists.
