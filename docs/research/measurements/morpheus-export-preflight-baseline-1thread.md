# Morpheus sandbox export preflight

> Verdict: **yes** — At least one sandbox runtime exports, reloads, and executes a static 8-bit graph for the probe path.

Generated: 2026-08-07T13:54:25.572613+00:00

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
  "modal_wall_s": 84.83975295897108
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
  "parameter_count": 249316,
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
| `torch_fx_static_qnnpack` | yes | yes | yes | yes | yes | 655747 | 1339592704 | — |
| `torch_fx_static_fbgemm` | yes | yes | yes | yes | yes | 719293 | 1339592704 | — |
| `torch_fx_static_x86` | yes | yes | yes | yes | yes | 716517 | 1365159936 | — |
| `torch_jit_float32` | no | no | yes | yes | yes | 1068921 | 1365159936 | — |
| `safetensors_weight_only_int8` | no | no | yes | yes | yes | 286653 | 1365159936 | — |

## Candidate detail

### `torch_fx_static_qnnpack`

- runtime: `torch.ao.quantization.quantize_fx+qnnpack`
- quantization_format: `static_ptq_fx_qnnpack_int8`
- fallback_operators: `['GroupNorm:model.gn_stem', 'GroupNorm:model.blocks.0.gn_expand', 'GroupNorm:model.blocks.0.gn_dw', 'GroupNorm:model.blocks.0.gn_project', 'GroupNorm:model.blocks.1.gn_expand', 'GroupNorm:model.blocks.1.gn_dw', 'GroupNorm:model.blocks.1.gn_project', 'GroupNorm:model.blocks.2.gn_expand', 'GroupNorm:model.blocks.2.gn_dw', 'GroupNorm:model.blocks.2.gn_project', 'GroupNorm:model.blocks.3.gn_expand', 'GroupNorm:model.blocks.3.gn_dw', 'GroupNorm:model.blocks.3.gn_project', 'GroupNorm:model.blocks.4.gn_expand', 'GroupNorm:model.blocks.4.gn_dw', 'GroupNorm:model.blocks.4.gn_project', 'GroupNorm:model.blocks.5.gn_expand', 'GroupNorm:model.blocks.5.gn_dw', 'GroupNorm:model.blocks.5.gn_project', 'GroupNorm:model.blocks.6.gn_expand', 'GroupNorm:model.blocks.6.gn_dw', 'GroupNorm:model.blocks.6.gn_project', 'GroupNorm:model.blocks.7.gn_expand', 'GroupNorm:model.blocks.7.gn_dw', 'GroupNorm:model.blocks.7.gn_project', 'GroupNorm:model.blocks.8.gn_expand', 'GroupNorm:model.blocks.8.gn_dw', 'GroupNorm:model.blocks.8.gn_project', 'GroupNorm:model.blocks.9.gn_expand', 'GroupNorm:model.blocks.9.gn_dw', 'GroupNorm:model.blocks.9.gn_project', 'GroupNorm:model.blocks.10.gn_expand', 'GroupNorm:model.blocks.10.gn_dw', 'GroupNorm:model.blocks.10.gn_project', 'GroupNorm:model.blocks.11.gn_expand', 'GroupNorm:model.blocks.11.gn_dw', 'GroupNorm:model.blocks.11.gn_project']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 1.1891231536865234, 'pass_logit_mae': 0.3998603820800781, 'wdl_mae': 0.1227579414844513, 'hidden_owner_mae': 1.1275769472122192, 'enemy_army_bins_mae': 1.4219917058944702, 'enemy_general_mae': 1.0481637716293335, 'hidden_castle_mae': 1.046311855316162, 'land_margin_mae': 0.20861853659152985, 'army_margin_mae': 0.33250248432159424, 'castle_margin_mae': 1.4072065353393555, 'turns_to_termination_mae': 1.3032153844833374}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 1004.1230220000017, "mean_ms": 989.7452560000002, "min_ms": 967.4742059999985, "p50_ms": 997.6385400000005, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 43.812661000000475, "mean_ms": 42.78390433333357, "min_ms": 41.87169900000143, "p50_ms": 42.6673529999988, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 12.413107000000423, "mean_ms": 11.503445333332252, "min_ms": 10.964811999997437, "p50_ms": 11.132416999998895, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['GroupNorm stays float between quantized Conv layers (dequant → GroupNorm → quant). Reported, not hidden.', 'Serialized via torch.jit.trace (script rejected FX annotations).']`

### `torch_fx_static_fbgemm`

- runtime: `torch.ao.quantization.quantize_fx+fbgemm`
- quantization_format: `static_ptq_fx_fbgemm_int8`
- fallback_operators: `['GroupNorm:model.gn_stem', 'GroupNorm:model.blocks.0.gn_expand', 'GroupNorm:model.blocks.0.gn_dw', 'GroupNorm:model.blocks.0.gn_project', 'GroupNorm:model.blocks.1.gn_expand', 'GroupNorm:model.blocks.1.gn_dw', 'GroupNorm:model.blocks.1.gn_project', 'GroupNorm:model.blocks.2.gn_expand', 'GroupNorm:model.blocks.2.gn_dw', 'GroupNorm:model.blocks.2.gn_project', 'GroupNorm:model.blocks.3.gn_expand', 'GroupNorm:model.blocks.3.gn_dw', 'GroupNorm:model.blocks.3.gn_project', 'GroupNorm:model.blocks.4.gn_expand', 'GroupNorm:model.blocks.4.gn_dw', 'GroupNorm:model.blocks.4.gn_project', 'GroupNorm:model.blocks.5.gn_expand', 'GroupNorm:model.blocks.5.gn_dw', 'GroupNorm:model.blocks.5.gn_project', 'GroupNorm:model.blocks.6.gn_expand', 'GroupNorm:model.blocks.6.gn_dw', 'GroupNorm:model.blocks.6.gn_project', 'GroupNorm:model.blocks.7.gn_expand', 'GroupNorm:model.blocks.7.gn_dw', 'GroupNorm:model.blocks.7.gn_project', 'GroupNorm:model.blocks.8.gn_expand', 'GroupNorm:model.blocks.8.gn_dw', 'GroupNorm:model.blocks.8.gn_project', 'GroupNorm:model.blocks.9.gn_expand', 'GroupNorm:model.blocks.9.gn_dw', 'GroupNorm:model.blocks.9.gn_project', 'GroupNorm:model.blocks.10.gn_expand', 'GroupNorm:model.blocks.10.gn_dw', 'GroupNorm:model.blocks.10.gn_project', 'GroupNorm:model.blocks.11.gn_expand', 'GroupNorm:model.blocks.11.gn_dw', 'GroupNorm:model.blocks.11.gn_project']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.13497185707092285, 'pass_logit_mae': 0.14584684371948242, 'wdl_mae': 0.3903588056564331, 'hidden_owner_mae': 0.14735786616802216, 'enemy_army_bins_mae': 0.14215202629566193, 'enemy_general_mae': 0.1349177211523056, 'hidden_castle_mae': 0.14168034493923187, 'land_margin_mae': 0.5696256756782532, 'army_margin_mae': 0.9873297214508057, 'castle_margin_mae': 0.11937856674194336, 'turns_to_termination_mae': 0.7259060144424438}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 3378.2862819999978, "mean_ms": 3325.212742, "min_ms": 3275.5583710000024, "p50_ms": 3321.793573000001, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 200.4293369999992, "mean_ms": 199.06638733333298, "min_ms": 197.20517700000073, "p50_ms": 199.56464799999907, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 51.25576600000059, "mean_ms": 50.47620666666693, "min_ms": 49.95915499999981, "p50_ms": 50.21369900000039, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['GroupNorm stays float between quantized Conv layers (dequant → GroupNorm → quant). Reported, not hidden.', 'Serialized via torch.jit.trace (script rejected FX annotations).']`

### `torch_fx_static_x86`

- runtime: `torch.ao.quantization.quantize_fx+x86`
- quantization_format: `static_ptq_fx_x86_int8`
- fallback_operators: `['GroupNorm:model.gn_stem', 'GroupNorm:model.blocks.0.gn_expand', 'GroupNorm:model.blocks.0.gn_dw', 'GroupNorm:model.blocks.0.gn_project', 'GroupNorm:model.blocks.1.gn_expand', 'GroupNorm:model.blocks.1.gn_dw', 'GroupNorm:model.blocks.1.gn_project', 'GroupNorm:model.blocks.2.gn_expand', 'GroupNorm:model.blocks.2.gn_dw', 'GroupNorm:model.blocks.2.gn_project', 'GroupNorm:model.blocks.3.gn_expand', 'GroupNorm:model.blocks.3.gn_dw', 'GroupNorm:model.blocks.3.gn_project', 'GroupNorm:model.blocks.4.gn_expand', 'GroupNorm:model.blocks.4.gn_dw', 'GroupNorm:model.blocks.4.gn_project', 'GroupNorm:model.blocks.5.gn_expand', 'GroupNorm:model.blocks.5.gn_dw', 'GroupNorm:model.blocks.5.gn_project', 'GroupNorm:model.blocks.6.gn_expand', 'GroupNorm:model.blocks.6.gn_dw', 'GroupNorm:model.blocks.6.gn_project', 'GroupNorm:model.blocks.7.gn_expand', 'GroupNorm:model.blocks.7.gn_dw', 'GroupNorm:model.blocks.7.gn_project', 'GroupNorm:model.blocks.8.gn_expand', 'GroupNorm:model.blocks.8.gn_dw', 'GroupNorm:model.blocks.8.gn_project', 'GroupNorm:model.blocks.9.gn_expand', 'GroupNorm:model.blocks.9.gn_dw', 'GroupNorm:model.blocks.9.gn_project', 'GroupNorm:model.blocks.10.gn_expand', 'GroupNorm:model.blocks.10.gn_dw', 'GroupNorm:model.blocks.10.gn_project', 'GroupNorm:model.blocks.11.gn_expand', 'GroupNorm:model.blocks.11.gn_dw', 'GroupNorm:model.blocks.11.gn_project']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.13497185707092285, 'pass_logit_mae': 0.14584684371948242, 'wdl_mae': 0.3903588056564331, 'hidden_owner_mae': 0.14735786616802216, 'enemy_army_bins_mae': 0.14215202629566193, 'enemy_general_mae': 0.1349177211523056, 'hidden_castle_mae': 0.14168034493923187, 'land_margin_mae': 0.5696256756782532, 'army_margin_mae': 0.9873297214508057, 'castle_margin_mae': 0.11937856674194336, 'turns_to_termination_mae': 0.7259060144424438}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 3331.8955440000054, "mean_ms": 3316.5148686666676, "min_ms": 3299.3275820000035, "p50_ms": 3318.3214799999946, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 200.3589799999972, "mean_ms": 199.7735669999988, "min_ms": 199.28791499999932, "p50_ms": 199.67380599999984, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 50.4541500000002, "mean_ms": 50.180599333333, "min_ms": 50.00245400000125, "p50_ms": 50.08519399999756, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['GroupNorm stays float between quantized Conv layers (dequant → GroupNorm → quant). Reported, not hidden.', 'Serialized via torch.jit.trace (script rejected FX annotations).']`

### `torch_jit_float32`

- runtime: `torch.jit.script`
- quantization_format: `float32`
- fallback_operators: `[]`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.0, 'pass_logit_mae': 0.0, 'wdl_mae': 0.0, 'hidden_owner_mae': 0.0, 'enemy_army_bins_mae': 0.0, 'enemy_general_mae': 0.0, 'hidden_castle_mae': 0.0, 'land_margin_mae': 0.0, 'army_margin_mae': 0.0, 'castle_margin_mae': 0.0, 'turns_to_termination_mae': 0.0}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 687.3216720000102, "mean_ms": 631.0160843333346, "min_ms": 593.1400789999941, "p50_ms": 612.5865019999992, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 18.50912600001209, "mean_ms": 18.468968333337443, "min_ms": 18.428569999997535, "p50_ms": 18.469209000002706, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 6.081500999997047, "mean_ms": 6.0518699999979235, "min_ms": 6.020802999998409, "p50_ms": 6.053305999998315, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['Control path only. Float weights do not satisfy static 8-bit deployment.']`

### `safetensors_weight_only_int8`

- runtime: `safetensors+torch_float_dequant`
- quantization_format: `weight_only_int8_affine`
- fallback_operators: `['entire_forward:float32_after_dequant']`
- unsupported_operators: `[]`
- float_to_export_error: `{'policy_mae': 0.015557250939309597, 'pass_logit_mae': 0.027483701705932617, 'wdl_mae': 0.008859376423060894, 'hidden_owner_mae': 0.016147812828421593, 'enemy_army_bins_mae': 0.016711588948965073, 'enemy_general_mae': 0.01604282297194004, 'hidden_castle_mae': 0.016246017068624496, 'land_margin_mae': 0.0037797093391418457, 'army_margin_mae': 0.009498119354248047, 'castle_margin_mae': 0.0004897117614746094, 'turns_to_termination_mae': 0.028708338737487793}`
- batch_results: `{"enemy_proposal": {"batch": 64, "latency": {"max_ms": 586.1750809999933, "mean_ms": 565.3621246666631, "min_ms": 526.3501609999963, "p50_ms": 583.5611319999998, "reps": 3.0}, "ok": true, "output_shapes": [[64, 9, 21, 21], [64, 1], [64, 3], [64, 1, 21, 21], [64, 16, 21, 21], [64, 1, 21, 21], [64, 1, 21, 21], [64, 1], [64, 1], [64, 1], [64, 1]]}, "leaf": {"batch": 4, "latency": {"max_ms": 19.548892999992518, "mean_ms": 19.539307666661898, "min_ms": 19.527100999994218, "p50_ms": 19.54192899999896, "reps": 3.0}, "ok": true, "output_shapes": [[4, 9, 21, 21], [4, 1], [4, 3], [4, 1, 21, 21], [4, 16, 21, 21], [4, 1, 21, 21], [4, 1, 21, 21], [4, 1], [4, 1], [4, 1], [4, 1]]}, "root": {"batch": 1, "latency": {"max_ms": 7.5069170000006125, "mean_ms": 7.45871700000104, "min_ms": 7.405900999998494, "p50_ms": 7.463333000004013, "reps": 3.0}, "ok": true, "output_shapes": [[1, 9, 21, 21], [1, 1], [1, 3], [1, 1, 21, 21], [1, 16, 21, 21], [1, 1, 21, 21], [1, 1, 21, 21], [1, 1], [1, 1], [1, 1], [1, 1]]}}`
- notes: `['Weight-only int8 with float runtime dequant. Not a static 8-bit graph.']`

## Pin audit

- judge matches sandbox: `True`
- installed vs sandbox: `{"jax": {"installed": "0.11.0", "match": true, "sandbox": "0.11.0"}, "numpy": {"installed": "2.4.6", "match": true, "sandbox": "2.4.6"}, "safetensors": {"installed": "0.8.0", "match": true, "sandbox": "0.8.0"}, "torch": {"installed": "2.13.0+cpu", "match": true, "sandbox": "2.13.0"}}`
- unavailable packages (not probed): `['onnxruntime', 'onnx', 'tensorrt', 'torchao', 'openvino']`
