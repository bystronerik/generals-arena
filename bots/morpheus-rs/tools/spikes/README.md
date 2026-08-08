# M3 engine spikes

Throwaway crates kept on purpose. They are the evidence behind the one place
this port departs from the rewrite plan's stated ordering — §3 ranked candle
first and a bespoke kernel last, and the measurement reversed that. A decision
that overrides a written plan should be re-derivable by someone who doubts it,
which a deleted spike is not.

None of this is built or linked by the bot. `tools/` is excluded from the
content hash (`fingerprint._SKIP_DIRS`), so these crates cannot re-identify
`morpheus-rs`, and their dependencies never reach the submission zip.

| crate | what it measures |
| --- | --- |
| `candle/` | the whole Morpheus graph in candle-core + candle-nn, batch 1/4/8 |
| `tract/` | the same graph through tract-onnx, from a fixed-shape ONNX export |
| `candle-ops/` | candle op by op on the Morpheus shapes — the file that found *why* candle loses |

## Running them

They need network (crates.io) and live outside the workspace, so build each on
its own:

```bash
cargo run --release --manifest-path bots/morpheus-rs/tools/spikes/candle/Cargo.toml -- bots/morpheus-rs/artifact/model.safetensors 200
```

The tract spike needs ONNX exports first, at the fixed batch sizes it times:

```bash
.venv/bin/python -c "
import torch
m = torch.jit.load('bots/morpheus/artifact/model_policy_wdl.pt', map_location='cpu').eval()
for b in (1, 4, 8):
    torch.onnx.export(m, (torch.randn(b, 49, 21, 21),), f'/tmp/policy_wdl_b{b}.onnx', input_names=['x'], opset_version=17, dynamo=False)
"
```

then point the spike at the directory holding them. `pip install onnx` is
required for the export and is not a repo dependency.

Set `RAYON_NUM_THREADS=1` for every run. One dedicated core is a competition
constraint, and a multi-threaded figure would answer a question nobody is
asking.

Results and what they decided:
[`morpheus-rs-inference-bench.md`](../../../../docs/research/measurements/morpheus-rs-inference-bench.md).
