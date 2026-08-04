# Morpheus online entry points (Modal Linux CPU)

Part 09a Phase 5 seat: `modal-cpu`

**Verdict:** `yes`

- seed: `0`
- n_blocks: `12`
- platform: `Linux-4.19.0-gvisor-x86_64-with-glibc2.36`
- torch: `2.13.0+cpu`
- supported_qengines: `['qnnpack', 'onednn', 'x86', 'fbgemm']`
- passed: `['fbgemm', 'x86']`
- failed: `[]`

## Engines

| Engine | Supported | Exported | Reloaded | Parity OK | Error |
| --- | --- | --- | --- | --- | --- |
| `fbgemm` | yes | yes | yes | yes |  |
| `x86` | yes | yes | yes | yes |  |

### `fbgemm`

- online_float_to_export_mae: `{'policy': {'policy_mae': 0.14108844101428986, 'pass_logit_mae': 0.11969733238220215}, 'policy_wdl': {'policy_mae': 0.1336280256509781, 'pass_logit_mae': 0.1189117431640625, 'wdl_mae': 0.7909323573112488}}`

### `x86`

- online_float_to_export_mae: `{'policy': {'policy_mae': 0.14108844101428986, 'pass_logit_mae': 0.11969733238220215}, 'policy_wdl': {'policy_mae': 0.1336280256509781, 'pass_logit_mae': 0.1189117431640625, 'wdl_mae': 0.7909323573112488}}`

