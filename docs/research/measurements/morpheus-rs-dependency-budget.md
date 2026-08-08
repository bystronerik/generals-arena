# Morpheus-rs dependency budget

What a crate would cost in the submission zip, measured rather than
assumed. Limits: 50 MB zipped, 512 MB unpacked, **10,000 files**
(RULES.md §08); the file count is the binding one.

- Measured: 2026-08-08T16:16:11+00:00
- Toolchain: `cargo 1.97.1 (c980f4866 2026-06-30)`

Each crate is vendored **in isolation**, with default features off, so
these are marginal-if-first figures. Real graphs share crates, so two
of them together cost less than the sum of their rows.

| crate | crates in graph | files | % of 10k cap | unpacked |
| --- | ---: | ---: | ---: | ---: |
| `sha2` 0.10 | 10 | 565 | 5.65% | 6.3 MB |
| `candle-core` 0.9 | 91 | 3,888 | 38.88% | 51.5 MB |

## Why this exists

The plan's §1 asserted that `cargo vendor` "blows past 10k files
easily", and that claim had already been used to justify hand-writing
SHA-256 rather than depending on `sha2`. The measurement does not
support it: `sha2` is 5.6% of the cap. The hand-written implementation
is still the right call for other reasons — hashing costs 0.001 ms p99,
so the library's speed advantage is worthless, and the digests are
dictionary keys rather than a security boundary — but the budget was
not one of them.

The budget is real where it binds: an inference crate is two orders of
magnitude heavier than a hash, and M3 picks one.

## Heaviest crates per graph

**`sha2`**

- `libc-0.2.189` — 404 files, 4.5 MB
- `sha2-0.10.9` — 32 files, 0.1 MB
- `typenum-1.20.1` — 25 files, 1.2 MB
- `digest-0.10.7` — 22 files, 0.1 MB
- `generic-array-0.14.7` — 17 files, 0.1 MB

**`candle-core`**

- `libc-0.2.189` — 404 files, 4.5 MB
- `zerocopy-0.8.56` — 316 files, 1.7 MB
- `windows-sys-0.61.2` — 258 files, 18.2 MB
- `libm-0.2.16` — 156 files, 0.7 MB
- `zerocopy-derive-0.8.56` — 135 files, 1.0 MB

A single crate usually dominates. `sha2` pulls `cpufeatures` → `libc`
for runtime CPU detection, and `libc` alone is 404 of its 565 files —
worth knowing before assuming a small crate is small.

## Re-running

```bash
python bots/morpheus-rs/tools/dependency_budget.py --crate tract-onnx=0.21
```

Needs network. Nothing here touches the bot or its lock file; every
probe is vendored in a scratch directory and thrown away.

