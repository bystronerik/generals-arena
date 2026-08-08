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
| vendored | ok | 3.31 | 10 | 18182 B | 8705 B | `1 0 0 0 0` / `1 0 0 0 0` |
| static | ok | 0.03 | 3 | 501558 B | 225874 B | `1 0 0 0 0` / `1 0 0 0 0` |

Both variants build with no network and answer the protocol.

The judge's limits are 50 MB zipped, 512 MB unpacked, 10,000 files. The
file count is the binding one for a Rust bot (`cargo vendor` over a fat
graph clears it easily), and a zero-dependency crate vendors to nothing —
which is why the dependency budget is reviewed at every `cargo add`.

