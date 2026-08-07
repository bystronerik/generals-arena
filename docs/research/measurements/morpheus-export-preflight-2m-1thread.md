# Morpheus sandbox export preflight

> Verdict: **yes** — At least one sandbox runtime exports, reloads, and executes a static 8-bit graph for the probe path.

Generated: 2026-08-07T13:55:40.713244+00:00

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
  "torch_num_threads": 1,
  "modal_wall_s": 177.10467883304227
}
```

## Architecture probe

```json
{
  "in_channels": 49,
  "board": 21,
  "trunk_channels": 192,
  "expansion": 384,
  "policy_channels": 9,
  "n_blocks": 12,
  "n_army_bins": 16,
  "army_bin_edges": [
    0.0,
    1.681792830507429,
    2.82842712474619,
    4.756828460010883,
    7.999999999999998,
    13.454342644059432,
    22.627416997969508,
    38.05462768008706,
    63.99999999999998,
    107.63474115247539,
    181.01933598375612,
    304.4370214406964,
    511.9999999999995,
    861.0779292198033,
    1448.154687870048,
    2435.4961715255718,
    4096.0
  ],
  "group_norm_groups": 8,
  "dilation_cycle": [
    1,
    2,
    4
  ],
  "parameter_count": 1927524,
  "outputs": {
    "policy": "N\u00d79\u00d721\u00d721",
    "pass_logit": "N\u00d71",
    "wdl_logits": "N\u00d73",
    "hidden_owner": "N\u00d71\u00d721\u00d721",
    "enemy_army_bins": "N\u00d716\u00d721\u00d721",
    "enemy_general": "N\u00d71\u00d721\u00d721",
    "hidden_castle": "N\u00d71\u00d721\u00d721",
    "land_margin": "N\u00d71",
    "army_margin": "N\u00d71",
    "castle_margin": "N\u00d71",
    "turns_to_termination": "N\u00d71"
  }
}
```

## Batch shapes

- shapes: `[1, 4, 64]`
- roles: `{'1': 'root', '4': 'leaf', '64': 'enemy_proposal'}`

## Candidates

| Name | Viable | Static 8-bit | Exported | Reloaded | Executed | Serialized B | Peak RSS | Error |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `torch_fx_static_qnnpack` | yes | yes | yes | yes | yes | 2412291 | 1527013376 | — |
| `torch_fx_static_fbgemm` | yes | yes | yes | yes | yes | 2569661 | 1683394560 | — |
| `torch_fx_static_x86` | yes | yes | yes | yes | yes | 2566757 | 1740288000 | — |
| `torch_jit_float32` | no | no | yes | yes | yes | 7781945 | 1806368768 | — |
| `safetensors_weight_only_int8` | no | no | yes | yes | yes | 2011885 | 1806368768 | — |

## Candidate detail

### `torch_fx_static_qnnpack`

- runtime: `torch.ao.quantization.quantize_fx+qnnpack`
- quantization_format: `static_ptq_fx_qnnpack_int8`
- fallback_operators: `['GroupNorm:model.gn_stem', 'GroupNorm:model.blocks.0.gn_expand', 'GroupNorm:model.blocks.0.gn_dw', 'GroupNorm:model.blocks.0.gn_project', 'GroupNorm:model.blocks.1.gn_expand', 'GroupNorm:model.blocks.1.gn_dw', 'GroupNorm:model.blocks.1.gn_project', 'GroupNorm:model.blocks.2.gn_expand', 'GroupNorm:model.blocks.2.gn_dw', 'GroupNorm:model.blocks.2.gn_project', 'GroupNorm:model.blocks.3.gn_expand', 'GroupNorm:model.blocks.3.gn_dw', 'GroupNorm:model.blocks.3.gn_project', 'GroupNorm:model.blocks.4.gn_expand', 'GroupNorm:model.blocks.4.gn_dw', 'GroupNorm:model.blocks.4.gn_project', 'GroupNorm:model.blocks.5.gn_expand', 'GroupNorm:model.blocks.5.gn_dw', 'GroupNorm:model.blocks.5.gn_project', 'GroupNorm:model.blocks.6.gn_expand', 'GroupNorm:model.blocks.6.gn_dw', 'GroupNorm:model.blocks.6.gn_project', 'GroupNorm:model.blocks.7.gn_expand', 'GroupNorm:model.blocks.7.gn_dw', 'GroupNorm:model.blocks.7.gn_project', 'GroupNorm:model.blocks.8.gn_expand', 'GroupNorm:model.blocks.8.gn_dw', 'GroupNorm:model.blocks.8.gn_project', 'GroupNorm:model.blocks.9.gn_expand', 'GroupNorm:model.blocks.9.gn_dw', 'GroupNorm:model.blocks.9.gn_project', 'GroupNorm:model.blocks.10.gn_expand', 'GroupNorm:model.blocks.10.gn_dw', 'GroupNorm:model.blocks.10.gn_project', 'GroupNorm:model.blocks.11.gn_expand', 'GroupNorm:model.blocks.11.gn_dw', 'GroupNorm:model.blocks.11.gn_project']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 1.0277636051177979, 'pass_logit_mae': 0.02904456853866577, 'wdl_mae': 0.8217406272888184, 'hidden_owner_mae': 1.1622806787490845, 'enemy_army_bins_mae': 1.1602758169174194, 'enemy_general_mae': 1.015910267829895, 'hidden_castle_mae': 1.1460343599319458, 'land_margin_mae': 0.061682939529418945, 'army_margin_mae': 0.17394626140594482, 'castle_margin_mae': 0.31925058364868164, 'turns_to_termination_mae': 0.8055639266967773}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 3995.9434909999986, "mean_ms": 3903.0538870000014, "min_ms": 3833.2565400000008, "p50_ms": 3879.961630000004, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 180.75920700000125, "mean_ms": 180.6198290000012, "min_ms": 180.51446500000168, "p50_ms": 180.5858150000006, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 47.11831700000246, "mean_ms": 46.07019466666884, "min_ms": 45.478979000002084, "p50_ms": 45.61328800000197, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['GroupNorm stays float between quantized Conv layers (dequant → GroupNorm → quant). Reported, not hidden.', 'Serialized via torch.jit.trace (script rejected FX annotations).']`

### `torch_fx_static_fbgemm`

- runtime: `torch.ao.quantization.quantize_fx+fbgemm`
- quantization_format: `static_ptq_fx_fbgemm_int8`
- fallback_operators: `['GroupNorm:model.gn_stem', 'GroupNorm:model.blocks.0.gn_expand', 'GroupNorm:model.blocks.0.gn_dw', 'GroupNorm:model.blocks.0.gn_project', 'GroupNorm:model.blocks.1.gn_expand', 'GroupNorm:model.blocks.1.gn_dw', 'GroupNorm:model.blocks.1.gn_project', 'GroupNorm:model.blocks.2.gn_expand', 'GroupNorm:model.blocks.2.gn_dw', 'GroupNorm:model.blocks.2.gn_project', 'GroupNorm:model.blocks.3.gn_expand', 'GroupNorm:model.blocks.3.gn_dw', 'GroupNorm:model.blocks.3.gn_project', 'GroupNorm:model.blocks.4.gn_expand', 'GroupNorm:model.blocks.4.gn_dw', 'GroupNorm:model.blocks.4.gn_project', 'GroupNorm:model.blocks.5.gn_expand', 'GroupNorm:model.blocks.5.gn_dw', 'GroupNorm:model.blocks.5.gn_project', 'GroupNorm:model.blocks.6.gn_expand', 'GroupNorm:model.blocks.6.gn_dw', 'GroupNorm:model.blocks.6.gn_project', 'GroupNorm:model.blocks.7.gn_expand', 'GroupNorm:model.blocks.7.gn_dw', 'GroupNorm:model.blocks.7.gn_project', 'GroupNorm:model.blocks.8.gn_expand', 'GroupNorm:model.blocks.8.gn_dw', 'GroupNorm:model.blocks.8.gn_project', 'GroupNorm:model.blocks.9.gn_expand', 'GroupNorm:model.blocks.9.gn_dw', 'GroupNorm:model.blocks.9.gn_project', 'GroupNorm:model.blocks.10.gn_expand', 'GroupNorm:model.blocks.10.gn_dw', 'GroupNorm:model.blocks.10.gn_project', 'GroupNorm:model.blocks.11.gn_expand', 'GroupNorm:model.blocks.11.gn_dw', 'GroupNorm:model.blocks.11.gn_project']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.11222942918539047, 'pass_logit_mae': 0.6090337038040161, 'wdl_mae': 0.6599249839782715, 'hidden_owner_mae': 0.11067228764295578, 'enemy_army_bins_mae': 0.11550866812467575, 'enemy_general_mae': 0.1071619912981987, 'hidden_castle_mae': 0.11712217330932617, 'land_margin_mae': 0.5507736802101135, 'army_margin_mae': 0.08778166770935059, 'castle_margin_mae': 0.09078168869018555, 'turns_to_termination_mae': 0.8275470733642578}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 8391.175563000004, "mean_ms": 8094.163191333337, "min_ms": 7878.361608000006, "p50_ms": 8012.952403, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 451.7799239999931, "mean_ms": 451.2862539999981, "min_ms": 450.48923800000296, "p50_ms": 451.58959999999837, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 115.01101999999719, "mean_ms": 114.63765933333055, "min_ms": 113.9620679999993, "p50_ms": 114.93988999999516, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['GroupNorm stays float between quantized Conv layers (dequant → GroupNorm → quant). Reported, not hidden.', 'Serialized via torch.jit.trace (script rejected FX annotations).']`

### `torch_fx_static_x86`

- runtime: `torch.ao.quantization.quantize_fx+x86`
- quantization_format: `static_ptq_fx_x86_int8`
- fallback_operators: `['GroupNorm:model.gn_stem', 'GroupNorm:model.blocks.0.gn_expand', 'GroupNorm:model.blocks.0.gn_dw', 'GroupNorm:model.blocks.0.gn_project', 'GroupNorm:model.blocks.1.gn_expand', 'GroupNorm:model.blocks.1.gn_dw', 'GroupNorm:model.blocks.1.gn_project', 'GroupNorm:model.blocks.2.gn_expand', 'GroupNorm:model.blocks.2.gn_dw', 'GroupNorm:model.blocks.2.gn_project', 'GroupNorm:model.blocks.3.gn_expand', 'GroupNorm:model.blocks.3.gn_dw', 'GroupNorm:model.blocks.3.gn_project', 'GroupNorm:model.blocks.4.gn_expand', 'GroupNorm:model.blocks.4.gn_dw', 'GroupNorm:model.blocks.4.gn_project', 'GroupNorm:model.blocks.5.gn_expand', 'GroupNorm:model.blocks.5.gn_dw', 'GroupNorm:model.blocks.5.gn_project', 'GroupNorm:model.blocks.6.gn_expand', 'GroupNorm:model.blocks.6.gn_dw', 'GroupNorm:model.blocks.6.gn_project', 'GroupNorm:model.blocks.7.gn_expand', 'GroupNorm:model.blocks.7.gn_dw', 'GroupNorm:model.blocks.7.gn_project', 'GroupNorm:model.blocks.8.gn_expand', 'GroupNorm:model.blocks.8.gn_dw', 'GroupNorm:model.blocks.8.gn_project', 'GroupNorm:model.blocks.9.gn_expand', 'GroupNorm:model.blocks.9.gn_dw', 'GroupNorm:model.blocks.9.gn_project', 'GroupNorm:model.blocks.10.gn_expand', 'GroupNorm:model.blocks.10.gn_dw', 'GroupNorm:model.blocks.10.gn_project', 'GroupNorm:model.blocks.11.gn_expand', 'GroupNorm:model.blocks.11.gn_dw', 'GroupNorm:model.blocks.11.gn_project']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.11555773764848709, 'pass_logit_mae': 0.026761114597320557, 'wdl_mae': 0.6599249839782715, 'hidden_owner_mae': 0.10894424468278885, 'enemy_army_bins_mae': 0.11344719678163528, 'enemy_general_mae': 0.11315099895000458, 'hidden_castle_mae': 0.12160506844520569, 'land_margin_mae': 0.5507736802101135, 'army_margin_mae': 0.08778166770935059, 'castle_margin_mae': 0.09078168869018555, 'turns_to_termination_mae': 0.3179335594177246}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 8468.425744, "mean_ms": 8340.565418666662, "min_ms": 8184.394201999993, "p50_ms": 8368.87630999999, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 457.1532519999977, "mean_ms": 453.9215229999959, "min_ms": 449.5943639999922, "p50_ms": 455.01695299999767, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 113.86970200000235, "mean_ms": 113.80018866666812, "min_ms": 113.70783300000653, "p50_ms": 113.82303099999547, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['GroupNorm stays float between quantized Conv layers (dequant → GroupNorm → quant). Reported, not hidden.', 'Serialized via torch.jit.trace (script rejected FX annotations).']`

### `torch_jit_float32`

- runtime: `torch.jit.script`
- quantization_format: `float32`
- fallback_operators: `[]`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.0, 'pass_logit_mae': 0.0, 'wdl_mae': 0.0, 'hidden_owner_mae': 0.0, 'enemy_army_bins_mae': 0.0, 'enemy_general_mae': 0.0, 'hidden_castle_mae': 0.0, 'land_margin_mae': 0.0, 'army_margin_mae': 0.0, 'castle_margin_mae': 0.0, 'turns_to_termination_mae': 0.0}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 2495.312845000001, "mean_ms": 2444.8883976666784, "min_ms": 2354.531706000017, "p50_ms": 2484.820642000017, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 87.6642650000008, "mean_ms": 86.42192133333992, "min_ms": 85.50583200002393, "p50_ms": 86.09566699999505, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 23.228870999986384, "mean_ms": 23.110787999996774, "min_ms": 22.93625199999383, "p50_ms": 23.167241000010108, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['Control path only. Float weights do not satisfy static 8-bit deployment.']`

### `safetensors_weight_only_int8`

- runtime: `safetensors+torch_float_dequant`
- quantization_format: `weight_only_int8_affine`
- fallback_operators: `['entire_forward:float32_after_dequant']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.012247699312865734, 'pass_logit_mae': 0.011938989162445068, 'wdl_mae': 0.010526120662689209, 'hidden_owner_mae': 0.015358355827629566, 'enemy_army_bins_mae': 0.015196685679256916, 'enemy_general_mae': 0.011989284306764603, 'hidden_castle_mae': 0.013128547929227352, 'land_margin_mae': 0.013590097427368164, 'army_margin_mae': 0.004348278045654297, 'castle_margin_mae': 0.006590604782104492, 'turns_to_termination_mae': 0.0098419189453125}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 2742.1054630000017, "mean_ms": 2430.2955900000047, "min_ms": 2239.2628489999993, "p50_ms": 2309.5184580000137, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 87.8818540000168, "mean_ms": 87.41824433334955, "min_ms": 86.87365100001898, "p50_ms": 87.49922800001286, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 23.887447999982214, "mean_ms": 23.725089333320664, "min_ms": 23.55630099998507, "p50_ms": 23.731518999994705, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['Weight-only int8 with float runtime dequant. Not a static 8-bit graph.']`

## Pin audit

- judge matches sandbox: `True`
- installed vs sandbox: `{"jax": {"installed": "0.11.0", "match": true, "sandbox": "0.11.0"}, "numpy": {"installed": "2.4.6", "match": true, "sandbox": "2.4.6"}, "safetensors": {"installed": "0.8.0", "match": true, "sandbox": "0.8.0"}, "torch": {"installed": "2.13.0+cpu", "match": true, "sandbox": "2.13.0"}}`
- unavailable packages (not probed): `['onnxruntime', 'onnx', 'tensorrt', 'torchao', 'openvino']`
