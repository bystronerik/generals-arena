# Morpheus-rs M0 CPU-feature probe (Modal, one core)

Milestone M0 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md).
Chooses the Rust binary's `target-cpu`.

- Last sampled: 2026-08-08T11:21:51+00:00
- Containers: 24 × `@app.function(cpu=1)`, over 4 run(s)
- **Compile target: `x86-64-v3`** — the *floor*: hosts at a higher level were also seen

## Why this is a proxy

The generals.bot sandbox has no network and no visible logs, so it cannot be probed directly. Modal stands in for "a one-core x86 Linux server container" and nothing stronger. The compile target is therefore the floor of what was observed, and any hand-written SIMD path must do runtime feature detection rather than trust the flag (rewrite-plan §9, R4).

## Hosts seen

| cpu identity | level | containers | cache | avx512f |
| --- | --- | ---: | --- | --- |
| AuthenticAMD family 175 model 17 stepping unknown | `x86-64-v4` | 14 | 8192 KB | yes |
| AuthenticAMD family 175 model 1 stepping unknown | `x86-64-v3` | 6 | 8192 KB | no |
| GenuineIntel family 6 model 85 stepping unknown | `x86-64-v4` | 4 | 8192 KB | yes |

### Per run

| run | sampled | containers | levels |
| ---: | --- | ---: | --- |
| 1 | 2026-08-08T11:20:08+00:00 | 10 | `x86-64-v4` |
| 2 | 2026-08-08T11:21:17+00:00 | 6 | `x86-64-v3` |
| 3 | 2026-08-08T11:21:38+00:00 | 4 | `x86-64-v4` |
| 4 | 2026-08-08T11:21:51+00:00 | 4 | `x86-64-v4` |

**Distinct CPU identities: 3** across 24 containers. Containers launched together land on the same generation, so variety comes from running the probe again later, not from asking for more containers at once.

Containers report more visible CPUs than the `cpu=1` reservation, and the sandbox masks the model name and cache topology, so neither is evidence about the judge's host.

## Feature availability

| flag | present on every sampled host |
| --- | --- |
| `avx` | yes |
| `avx2` | yes |
| `avx512f` | no |
| `fma` | yes |
| `f16c` | yes |
| `avx_vnni` | no |
| `amx_tile` | no |

Flags present on some hosts and not others: `acpi`, `arch_capabilities`, `avx512_bitalg`, `avx512_vbmi2`, `avx512_vnni`, `avx512_vpopcntdq`, `avx512bw`, `avx512cd`, `avx512dq`, `avx512f`, `avx512vbmi`, `avx512vl`, `cmp_legacy`, `cqm`, `cr8_legacy`, `dca`, `ds_cpl`, `dtes64`, `dts`, `erms`, `est`, `flush_l1d`, `fsrm`, `fxsr_opt`, `gfni`, `hypervisor`, `md_clear`, `misalignsse`, `mmxext`, `monitor`, `mpx`, `ospke`, `osvw`, `pbe`, `pdcm`, `perfctr_core`, `pku`, `rdpid`, `rdt_a`, `sdbg`…. A binary may not assume any of these.

