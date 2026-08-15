# What is not here any more

Three tools left `bots/morpheus-joe/tools/` at N1 with the network they
described. All three still exist, unchanged, in `bots/morpheus-rs/tools/`,
which this fork never edits — nothing below is recoverable only from git
history (joe-net-plan §10).

| gone | why | what replaces it |
| --- | --- | --- |
| `convert_artifact.py` | It converted morpheus's `.pt` checkpoint to safetensors. This bot converts nothing: it **syncs**, one hop down the chain, from the bot that does (§8.7). | `scripts/joe_artifact_fanout.py`, plus `tests/test_joe_source_fanout.py`, which is the part that actually fires |
| `bench_inference.py` | It was the local half of M3's engine shoot-out — the Rust CNN against TorchScript, three head sets against three batch sizes. There is no TorchScript arm, no head switch and no batch axis left to shoot out. | the binary's own `bench` subcommand, for catching a build wrong by an order of magnitude; and joe-rs's `bench --stages`, which is the **authority** on the forward's cost and is what N0 measured |
| `spikes/` | Three probe crates (candle, candle-ops, tract) that timed alternative engines on morpheus's 49-plane graph against a `model.safetensors` this bot no longer carries. | nothing. The engine question was settled twice — by M3 for morpheus and by joe-rs's port plan §9 for joe — and both times against a framework |

`capture_morpheus.py`, `bench_belief.py`, `dependency_budget.py`,
`vendor_probe.py`, `mutation_check.py`, `package_submission.py` and
`run_parity.sh` all stay: every one of them is about the search, the belief
filter, the packaging, or the harness, and none of those changed.
