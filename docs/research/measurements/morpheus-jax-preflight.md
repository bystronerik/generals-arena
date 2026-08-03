# Morpheus JAX preflight

> Verdict: **yes** — GPU self-play transitions are permitted.

Generated: 2026-08-03T18:17:55.369826+00:00

## Checks

| Check | Result |
| --- | --- |
| `a100_80gb` | yes |
| `compiled` | yes |
| `competition_modifiers` | yes |
| `cpu_gpu_parity` | yes |

## GPU device

- platform: `gpu`
- device: `cuda:0`
- nvidia: `{'nvidia_smi_error': "Command '['nvidia-smi', '--query-gpu=name,memory.total,driver_version,cuda_version', '--format=csv,noheader']' returned non-zero exit status 2.", 'nvcc': None}`
- versions: `{'jax': '0.11.0', 'jaxlib': '0.11.0', 'numpy': '2.4.6', 'xla_error': "module 'jax.lib' has no attribute 'xla_bridge'"}`

## Throughput

| Seat | Cold compile (s) | Warm steps/s | Peak device bytes | H2D (s) | D2H (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cpu` | 3.077 | 7,685 | None | 0.0032 | 0.0003 |
| `gpu` | 5.854 | 273,192 | 1280 | 0.0057 | 0.0019 |

Config: `{'seed': 0, 'num_envs': 64, 'scan_steps': 50, 'pool_size': 32, 'warm_reps': 3}`

## CPU/GPU parity

| Fixture | Match |
| --- | --- |
| `castle_build` | yes |
| `deathtouch` | yes |
| `deathtouch_chase_defense` | yes |
| `pass_noop` | yes |

## Accounting

- A100 hours (this run): `0.03235120986777778`
- Wall seconds: `159.7788955840515`
- Budget hours: `48`

## Competition modifiers (GPU seat)

```json
{
  "mode": "competition",
  "build_castles": true,
  "deathtouch_turn": 800,
  "truncation": 1200,
  "pad_to": 21,
  "min_grid_size": 18,
  "max_grid_size": 21,
  "perfect_info": false,
  "pool_size": 32
}
```
