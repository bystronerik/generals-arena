# Morpheus sandbox export preflight

> Verdict: **yes** — At least one sandbox runtime exports, reloads, and executes a static 8-bit graph for the probe path.

Generated: 2026-08-03T18:41:37.494623+00:00

## Checks

| Check | Result |
| --- | --- |
| `sandbox_torch_pin_match` | yes |
| `any_static_8bit_viable` | yes |
| `batch_shapes_exercised` | yes |
| `fallbacks_reported` | yes |

Viable candidates: `['torch_fx_static_qnnpack', 'torch_fx_static_fbgemm', 'torch_fx_static_x86']`

## Host

```json
{
  "seat": "modal-cpu",
  "system": "Linux",
  "machine": "x86_64",
  "python": "3.12.10",
  "torch": "2.13.0+cpu",
  "quantized_engine": "x86",
  "supported_qengines": [
    "qnnpack",
    "onednn",
    "x86",
    "fbgemm"
  ],
  "modal_wall_s": 34.11661066696979
}
```

## Architecture probe

```json
{
  "in_channels": 49,
  "board": 21,
  "trunk_channels": 64,
  "expansion": 128,
  "policy_channels": 9,
  "n_blocks": 12,
  "group_norm_groups": 8,
  "dilation_cycle": [
    1,
    2,
    4
  ],
  "parameter_count": 247565,
  "outputs": {
    "policy": "N\u00d79\u00d721\u00d721",
    "pass_logit": "N\u00d71",
    "wdl_logits": "N\u00d73"
  }
}
```

## Batch shapes

- shapes: `[1, 4, 64]`
- roles: `{'1': 'root', '4': 'leaf', '64': 'enemy_proposal'}`

## Candidates

| Name | Viable | Static 8-bit | Exported | Reloaded | Executed | Serialized B | Peak RSS | Error |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `torch_fx_static_qnnpack` | yes | yes | yes | yes | yes | 614009 | 1334919168 | — |
| `torch_fx_static_fbgemm` | yes | yes | yes | yes | yes | 672751 | 1400848384 | — |
| `torch_fx_static_x86` | yes | yes | yes | yes | yes | 669849 | 1437773824 | — |
| `torch_jit_float32` | no | no | yes | yes | yes | 1051755 | 1444999168 | — |
| `safetensors_weight_only_int8` | no | no | yes | yes | yes | 283127 | 1445273600 | — |

## Candidate detail

### `torch_fx_static_qnnpack`

- runtime: `torch.ao.quantization.quantize_fx+qnnpack`
- quantization_format: `static_ptq_fx_qnnpack_int8`
- fallback_operators: `['GroupNorm:gn_stem', 'GroupNorm:blocks.0.gn_expand', 'GroupNorm:blocks.0.gn_dw', 'GroupNorm:blocks.0.gn_project', 'GroupNorm:blocks.1.gn_expand', 'GroupNorm:blocks.1.gn_dw', 'GroupNorm:blocks.1.gn_project', 'GroupNorm:blocks.2.gn_expand', 'GroupNorm:blocks.2.gn_dw', 'GroupNorm:blocks.2.gn_project', 'GroupNorm:blocks.3.gn_expand', 'GroupNorm:blocks.3.gn_dw', 'GroupNorm:blocks.3.gn_project', 'GroupNorm:blocks.4.gn_expand', 'GroupNorm:blocks.4.gn_dw', 'GroupNorm:blocks.4.gn_project', 'GroupNorm:blocks.5.gn_expand', 'GroupNorm:blocks.5.gn_dw', 'GroupNorm:blocks.5.gn_project', 'GroupNorm:blocks.6.gn_expand', 'GroupNorm:blocks.6.gn_dw', 'GroupNorm:blocks.6.gn_project', 'GroupNorm:blocks.7.gn_expand', 'GroupNorm:blocks.7.gn_dw', 'GroupNorm:blocks.7.gn_project', 'GroupNorm:blocks.8.gn_expand', 'GroupNorm:blocks.8.gn_dw', 'GroupNorm:blocks.8.gn_project', 'GroupNorm:blocks.9.gn_expand', 'GroupNorm:blocks.9.gn_dw', 'GroupNorm:blocks.9.gn_project', 'GroupNorm:blocks.10.gn_expand', 'GroupNorm:blocks.10.gn_dw', 'GroupNorm:blocks.10.gn_project', 'GroupNorm:blocks.11.gn_expand', 'GroupNorm:blocks.11.gn_dw', 'GroupNorm:blocks.11.gn_project']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 1.1868504285812378, 'pass_logit_mae': 0.3262636661529541, 'wdl_mae': 0.19943396747112274}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 239.2787219999999, "mean_ms": 235.2021926666665, "min_ms": 228.64707499999957, "p50_ms": 237.6807809999999, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3]]}, "leaf": {"batch": 4, "latency": {"max_ms": 26.413317999999464, "mean_ms": 26.075958666666565, "min_ms": 25.758936999999094, "p50_ms": 26.05562100000114, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3]]}, "root": {"batch": 1, "latency": {"max_ms": 21.746744999999734, "mean_ms": 21.184405666666944, "min_ms": 20.8096340000008, "p50_ms": 20.996838000000295, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3]]}}`
- notes: `['GroupNorm stays float between quantized Conv layers (dequant → GroupNorm → quant). Reported, not hidden.', 'Serialized via torch.jit.trace (script rejected FX annotations).']`

### `torch_fx_static_fbgemm`

- runtime: `torch.ao.quantization.quantize_fx+fbgemm`
- quantization_format: `static_ptq_fx_fbgemm_int8`
- fallback_operators: `['GroupNorm:gn_stem', 'GroupNorm:blocks.0.gn_expand', 'GroupNorm:blocks.0.gn_dw', 'GroupNorm:blocks.0.gn_project', 'GroupNorm:blocks.1.gn_expand', 'GroupNorm:blocks.1.gn_dw', 'GroupNorm:blocks.1.gn_project', 'GroupNorm:blocks.2.gn_expand', 'GroupNorm:blocks.2.gn_dw', 'GroupNorm:blocks.2.gn_project', 'GroupNorm:blocks.3.gn_expand', 'GroupNorm:blocks.3.gn_dw', 'GroupNorm:blocks.3.gn_project', 'GroupNorm:blocks.4.gn_expand', 'GroupNorm:blocks.4.gn_dw', 'GroupNorm:blocks.4.gn_project', 'GroupNorm:blocks.5.gn_expand', 'GroupNorm:blocks.5.gn_dw', 'GroupNorm:blocks.5.gn_project', 'GroupNorm:blocks.6.gn_expand', 'GroupNorm:blocks.6.gn_dw', 'GroupNorm:blocks.6.gn_project', 'GroupNorm:blocks.7.gn_expand', 'GroupNorm:blocks.7.gn_dw', 'GroupNorm:blocks.7.gn_project', 'GroupNorm:blocks.8.gn_expand', 'GroupNorm:blocks.8.gn_dw', 'GroupNorm:blocks.8.gn_project', 'GroupNorm:blocks.9.gn_expand', 'GroupNorm:blocks.9.gn_dw', 'GroupNorm:blocks.9.gn_project', 'GroupNorm:blocks.10.gn_expand', 'GroupNorm:blocks.10.gn_dw', 'GroupNorm:blocks.10.gn_project', 'GroupNorm:blocks.11.gn_expand', 'GroupNorm:blocks.11.gn_dw', 'GroupNorm:blocks.11.gn_project']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.13767758011817932, 'pass_logit_mae': 0.042981863021850586, 'wdl_mae': 0.08228454738855362}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 1151.9581279999968, "mean_ms": 1129.7102779999998, "min_ms": 1103.4624980000026, "p50_ms": 1133.7102080000002, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3]]}, "leaf": {"batch": 4, "latency": {"max_ms": 78.02986799999978, "mean_ms": 73.43656633333258, "min_ms": 71.12288099999908, "p50_ms": 71.15694999999889, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3]]}, "root": {"batch": 1, "latency": {"max_ms": 25.48076400000099, "mean_ms": 22.091770666667305, "min_ms": 17.16328900000086, "p50_ms": 23.63125900000007, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3]]}}`
- notes: `['GroupNorm stays float between quantized Conv layers (dequant → GroupNorm → quant). Reported, not hidden.', 'Serialized via torch.jit.trace (script rejected FX annotations).']`

### `torch_fx_static_x86`

- runtime: `torch.ao.quantization.quantize_fx+x86`
- quantization_format: `static_ptq_fx_x86_int8`
- fallback_operators: `['GroupNorm:gn_stem', 'GroupNorm:blocks.0.gn_expand', 'GroupNorm:blocks.0.gn_dw', 'GroupNorm:blocks.0.gn_project', 'GroupNorm:blocks.1.gn_expand', 'GroupNorm:blocks.1.gn_dw', 'GroupNorm:blocks.1.gn_project', 'GroupNorm:blocks.2.gn_expand', 'GroupNorm:blocks.2.gn_dw', 'GroupNorm:blocks.2.gn_project', 'GroupNorm:blocks.3.gn_expand', 'GroupNorm:blocks.3.gn_dw', 'GroupNorm:blocks.3.gn_project', 'GroupNorm:blocks.4.gn_expand', 'GroupNorm:blocks.4.gn_dw', 'GroupNorm:blocks.4.gn_project', 'GroupNorm:blocks.5.gn_expand', 'GroupNorm:blocks.5.gn_dw', 'GroupNorm:blocks.5.gn_project', 'GroupNorm:blocks.6.gn_expand', 'GroupNorm:blocks.6.gn_dw', 'GroupNorm:blocks.6.gn_project', 'GroupNorm:blocks.7.gn_expand', 'GroupNorm:blocks.7.gn_dw', 'GroupNorm:blocks.7.gn_project', 'GroupNorm:blocks.8.gn_expand', 'GroupNorm:blocks.8.gn_dw', 'GroupNorm:blocks.8.gn_project', 'GroupNorm:blocks.9.gn_expand', 'GroupNorm:blocks.9.gn_dw', 'GroupNorm:blocks.9.gn_project', 'GroupNorm:blocks.10.gn_expand', 'GroupNorm:blocks.10.gn_dw', 'GroupNorm:blocks.10.gn_project', 'GroupNorm:blocks.11.gn_expand', 'GroupNorm:blocks.11.gn_dw', 'GroupNorm:blocks.11.gn_project']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.13753321766853333, 'pass_logit_mae': 0.0181581974029541, 'wdl_mae': 0.06142665818333626}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 968.6457789999992, "mean_ms": 960.6205246666663, "min_ms": 954.2777369999982, "p50_ms": 958.9380580000011, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3]]}, "leaf": {"batch": 4, "latency": {"max_ms": 65.60816500000044, "mean_ms": 57.79703866666708, "min_ms": 53.326199000000685, "p50_ms": 54.45675200000011, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3]]}, "root": {"batch": 1, "latency": {"max_ms": 22.93982199999789, "mean_ms": 21.102399333333466, "min_ms": 20.006709000000455, "p50_ms": 20.360667000002053, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3]]}}`
- notes: `['GroupNorm stays float between quantized Conv layers (dequant → GroupNorm → quant). Reported, not hidden.', 'Serialized via torch.jit.trace (script rejected FX annotations).']`

### `torch_jit_float32`

- runtime: `torch.jit.script`
- quantization_format: `float32`
- fallback_operators: `[]`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.0, 'pass_logit_mae': 0.0, 'wdl_mae': 0.0}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 228.08131000000031, "mean_ms": 222.6642019999995, "min_ms": 216.39100499999842, "p50_ms": 223.52029099999982, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3]]}, "leaf": {"batch": 4, "latency": {"max_ms": 18.362076999999033, "mean_ms": 17.999902666665218, "min_ms": 17.514951999999084, "p50_ms": 18.122678999997532, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3]]}, "root": {"batch": 1, "latency": {"max_ms": 12.735484000000241, "mean_ms": 12.603159666666622, "min_ms": 12.45368600000063, "p50_ms": 12.620308999998997, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3]]}}`
- notes: `['Control path only. Float weights do not satisfy static 8-bit deployment.']`

### `safetensors_weight_only_int8`

- runtime: `safetensors+torch_float_dequant`
- quantization_format: `weight_only_int8_affine`
- fallback_operators: `['entire_forward:float32_after_dequant']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.015557243488729, 'pass_logit_mae': 0.0021889209747314453, 'wdl_mae': 0.006240785121917725}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 218.67481199999972, "mean_ms": 196.25757966666652, "min_ms": 184.23809999999818, "p50_ms": 185.8598270000016, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3]]}, "leaf": {"batch": 4, "latency": {"max_ms": 24.00634600000018, "mean_ms": 21.900844333333207, "min_ms": 20.357195999999078, "p50_ms": 21.338991000000362, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3]]}, "root": {"batch": 1, "latency": {"max_ms": 17.78944100000146, "mean_ms": 14.419601333333532, "min_ms": 11.82826499999834, "p50_ms": 13.641098000000795, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3]]}}`
- notes: `['Weight-only int8 with float runtime dequant. Not a static 8-bit graph.']`

## Pin audit

- judge matches sandbox: `True`
- installed vs sandbox: `{"jax": {"installed": "0.11.0", "match": true, "sandbox": "0.11.0"}, "numpy": {"installed": "2.4.6", "match": true, "sandbox": "2.4.6"}, "safetensors": {"installed": "0.8.0", "match": true, "sandbox": "0.8.0"}, "torch": {"installed": "2.13.0+cpu", "match": true, "sandbox": "2.13.0"}}`
- unavailable packages (not probed): `['onnxruntime', 'onnx', 'tensorrt', 'torchao', 'openvino']`
