# Morpheus army normalization

> Verdict: **keep_default** — Bootstrap max stack is 1545 (< 4096). Keep
> `log1p(x) / log1p(4096)` until a model manifest records a new scale.

Source: `data/trajectories/morpheus-bootstrap` (42 games, 4 302 060 owned-stack
samples).

## Overall quantiles

| p50 | p90 | p99 | p99.9 | max |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 12 | 46 | 356 | 1545 |

## Quantization error (256 levels)

| Slice | MAE | max abs | p99 relative |
| --- | ---: | ---: | ---: |
| all | 0.068 | 23.4 | 0.022 |
| ordinary (< 512) | 0.065 | 7.9 | 0.022 |
| extreme (≥ 512) | 6.31 | 23.4 | 0.016 |

## Command

```bash
python scripts/morpheus_measure.py army-normalization \
  --trajectories data/trajectories/morpheus-bootstrap \
  --output docs/research/measurements/morpheus-army-normalization.json
```
