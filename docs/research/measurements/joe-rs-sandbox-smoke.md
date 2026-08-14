# joe-rs — submission rehearsal on Linux x86, offline

Milestone J5.4 of [the packaging plan](../../bots/joe-rs/packaging.md).
joe-rs ships 93 vendored crates, so unlike the morpheus rehearsal there is
no separate `vendor-probe` variant: the submitted archive is itself the
proof that offline source replacement resolves a real graph.

- Container: `@app.function(cpu=1, memory=2048, block_network=True)`, x86_64, kernel 4.19.0-gvisor
- CPU: quota `17.00 cpu`, but `os.cpu_count()` reports 17 — the host's count leaking through gVisor, which is why the `-j1` row below exists
- Toolchain: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Network probe (curl to crates.io): exit 6 — blocked, as intended

| verdict | files | vendored | unpacked | zip |
| --- | ---: | ---: | ---: | ---: |
| ok | 3961 | 3946 | 86293389 B | 42183023 B |

## P2 — the intake build under one core and 2 GB

`lto = "fat"` with `codegen-units = 1` links the whole graph in one rustc
invocation, so the memory peak lands at the end of the build and an OOM kill
would appear as a failed build with a truncated link step. The tripwire in
packaging.md is an OOM kill or a build over 15 minutes.

Two builds, because cargo's default parallelism here is the host's and not
the judge's. The `-j1` row is the conservative bound and the one the
tripwire should be read against.

| cargo jobs | exit | wall s | peak RSS (largest child) | cgroup peak |
| --- | ---: | ---: | ---: | ---: |
| default | 0 | 74.97 | 1288 MiB | — |
| 1 | 0 | 129.21 | 1616 MiB | — |

`cgroup peak` is blank when the sandbox does not expose the counter, which
gVisor generally does not. The RSS column is the largest single child rather
than the cgroup total — for a fat-LTO build those nearly coincide, because
the last link is one process and the biggest one.

## What the bundle replied

Two 21x21 frames where our general sits on plentiful army with four empty
plains around it. A pass is the wrong answer there, which is what makes
the replies evidence rather than a formality: a joe-rs seat that cannot
load its artifact exits 1 and answers nothing, and one that runs without
deciding answers `1 0 0 0 0`.

| path | exit | replies |
| --- | ---: | --- |
| generated `run.sh` (exports `JOE_RS_ARTIFACT`) | 0 | `0 10 10 0 0` / `0 10 10 2 0` |
| binary directly, variable unset, cwd `/` | 0 | `0 10 10 0 0` / `0 10 10 2 0` |

The second row is the exe-relative branch of `main.rs::artifact_dir`, which
the launcher normally hides. Both paths resolving is why the `export` line
is insurance rather than a dependency.

