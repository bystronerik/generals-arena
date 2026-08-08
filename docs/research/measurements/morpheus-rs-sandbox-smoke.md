# Morpheus-rs M0.5 — submission rehearsal on Linux x86, offline

Milestone M0.5 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md):
prove the offline build path, the file-count budget, and binary
compatibility before any real porting starts.

- Container: `@app.function(cpu=1, block_network=True)`, x86_64, kernel 4.19.0-gvisor, 17 cpus visible
- Toolchain: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Network probe (curl to crates.io): exit 6 — blocked, as intended

Modal stands in for the generals.bot sandbox, which cannot be probed.
A green result here is necessary, not sufficient — which is why R4's
static-binary fallback is built and smoked on every packaging run.

| variant | build | build s | files | unpacked | zip | replies |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| vendored | ok | 4.62 | 20 | 139059 B | 46813 B | `1 0 0 0 0` / `1 0 0 0 0` |
| static | ok | 0.03 | 3 | 590646 B | 268243 B | `1 0 0 0 0` / `1 0 0 0 0` |
| vendor-probe | ok | 2.52 | 571 | 6258671 B | 1360046 B | `1 0 0 0 0` / `1 0 0 0 0` |

Every variant builds with no network and answers the protocol.

`vendor-probe` is not a shipping variant. The two that are have no
dependencies, so their `--offline` build proves less than it appears
to — with nothing to resolve, `--offline` cannot fail. The probe adds
a real transitive graph (`sha2` → `digest` → `block-buffer` →
`generic-array` → `typenum`) and a build script, so registry
replacement and intake-time build scripts are exercised before M3
depends on ninety crates.

The judge's limits are 50 MB zipped, 512 MB unpacked, 10,000 files. The
file count is the binding one for a Rust bot (`cargo vendor` over a fat
graph clears it easily), and a zero-dependency crate vendors to nothing —
which is why the dependency budget is reviewed at every `cargo add`.

