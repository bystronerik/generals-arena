# Morpheus pilot class-1 learn

> Smoke ok: **yes** — thin Modal/local learning path on scraped class-1 items.

Generated: `2026-08-04T20:26:36.268751+00:00`

## Result

- loss step 0: `13.128320693969727`
- loss final: `0.4495953619480133`
- loss decreased: `True`
- checkpoint reloadable: `True`
- sample count: `8`
- source labels: `['ResBot_reconstructions']`
- wall_s: `5.99614370893687`
- a100_hours (wall): `0.0016655954747046861`
- accounting line: `learning_curve_pilot`

## Config

- `batch_size`: `2`
- `device`: `cpu`
- `learning_rate`: `0.001`
- `manifest`: `training/morpheus/manifests/pilot-class1-scraped.json`
- `max_items`: `8`
- `n_blocks`: `12`
- `n_particles`: `4`
- `objective`: `training/morpheus/configs/pilot-objective.json`
- `seed`: `0`
- `steps`: `50`

## Notes

- Thin pilot smoke only. Not Part 14. Not a Part 13 yes.
- A100 wall time charges to learning_curve_pilot accounting notes.
