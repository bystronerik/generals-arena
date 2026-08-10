# Morpheus-rs — submission rehearsal on Linux x86, offline

Milestone M0.5 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md)
proved the offline build path, the file-count budget, and binary
compatibility before any real porting started. Re-run at **M8** against the
finished bot, where the interesting column is no longer *replies* but
*selfcheck*: a seat that cannot load its weights, its knobs, or a hardware
FMA still answers every frame with a well-formed skip, so a green
replies column proved nothing about this bot at all.

- Container: `@app.function(cpu=1, block_network=True)`, x86_64, kernel 4.19.0-gvisor, 17 cpus visible
- Toolchain: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Network probe (curl to crates.io): exit 6 — blocked, as intended

Modal stands in for the generals.bot sandbox, which cannot be probed.
A green result here is necessary, not sufficient — which is why R4's
static-binary fallback is built and smoked on every packaging run.

| variant | verdict | build s | files | unpacked | zip | replies |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| vendored | ok | 17.21 | 43 | 1689449 B | 1132136 B | `1 0 0 0 0` / `1 0 0 0 0` |
| static | ok | 0.2 | 7 | 2494592 B | 1587500 B | `1 0 0 0 0` / `1 0 0 0 0` |
| vendor-probe | ok | 2.44 | 571 | 6258671 B | 1360046 B | `1 0 0 0 0` / `1 0 0 0 0` |

## What the binary says about itself, on x86

`build.sh` runs `morpheus-rs selfcheck` at intake and aborts on a
non-zero exit. It loads the weights against their manifest digest,
resolves `deployment.json` rather than falling back to the placeholder
knobs, constructs the playing seat with its warmup and thread-count
invariant, and decides one hand-built frame — a general on thirteen
army with four empty neighbours, where a skip is the only wrong answer.

`warmup_ms` is 26 forwards, so it doubles as a per-forward probe.
Divide the column by 26: a build that lost its FMA would read fifty
times what it does, which is the failure this whole check exists for.

| variant | fma | load ms | warmup ms | decision | decide ms | verdict |
| --- | --- | ---: | ---: | --- | ---: | --- |
| vendored | true | 9.517 | 107.779 | `0 10 10 0 0` | 4.732 | ok |
| static | true | 17.207 | 109.315 | `0 10 10 0 0` | 9.927 | ok |

Five runs each, because one is a check and not a measurement —
the two variants differ by their libc, and the fallback inheriting
M7's latency qualification is an assumption until it is measured.

| variant | warmup ms (5 runs) | decide ms (5 runs) |
| --- | --- | --- |
| vendored | 107.8, 107.7, 108.1, 107.9, 107.7 | 4.7, 4.5, 4.7, 4.8, 4.7 |
| static | 109.3, 109.9, 111.4, 109.0, 108.8 | 9.9, 11.7, 10.0, 10.9, 10.3 |

Every variant builds with no network and answers the protocol.

`vendor-probe` is not a shipping variant. The two that are have no
dependencies, so their `--offline` build proves less than it appears
to — with nothing to resolve, `--offline` cannot fail. The probe adds
a real transitive graph (`sha2` → `digest` → `block-buffer` →
`generic-array` → `typenum`) and a build script, so registry
replacement and intake-time build scripts are exercised before M3
depends on ninety crates.

The judge's limits are 50 MB zipped, 512 MB unpacked, 10,000 files. The
file count is the binding one for a Rust bot, though not as binding as
this line used to claim: M2 measured `sha2` at 565 files and candle-core
at 3,888, against a cap of 10,000. A zero-dependency crate vendors to
nothing, which is why the shipped variants need the probe above to
exercise offline resolution at all.

