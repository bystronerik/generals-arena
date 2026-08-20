# joe-rs weight export

The joe-rs artifact is **derived from the joe artifact**, never trained or
edited on its own. `bots/joe-rs/tools/convert_artifact.py` turns
`bots/joe/artifact/ema.eqx` into `bots/joe-rs/artifact/model.safetensors`
plus a manifest.

```bash
.venv/bin/python bots/joe-rs/tools/convert_artifact.py
```

## Why the converter never parses the .eqx

An `.eqx` file has 100 unnamed leaves in pytree order; a converter that read
the framing itself could permute two same-shaped tensors (`q_proj` vs
`k_proj`) and load cleanly while playing plausibly badly (port-plan R2). So
the converter goes through `eqx.tree_deserialise_leaves` into the exact
template `bots/joe/agent.py` builds — the code path the deployed Python bot
trusts — then names each leaf from its pytree path.

Checks, all refusing to write on failure:

1. `weights_sha256` of the source `.eqx` matches joe's manifest.
2. Exactly 132 tensors with the `joe-net-v1` names/shapes, all float32,
   summing to 13,581,658 parameters (= the manifest's `n_params`). Those are
   the `M7F4` tier (depth 7, ff x4). Earlier tiers: `M7` (depth 7, ff x3) was
   132 / 11,514,586, and `M` (depth 5, ff x3) was 100 / 8,556,250 — note the
   leaf count is unchanged by an ff graft, which only widens existing
   tensors. The numbers are pinned rather than read from the manifest so a
   mis-shaped artifact still refuses, and a tier change is a deliberate edit
   here plus, in both Rust crates' `src/nn/net.rs`, `DEPTH`, `FF_DIM`, and
   the `ff_factor` entry of the manifest cross-check.
3. Round-trip: the written safetensors reloads bit-exact against the
   deserialised leaves.

`bin_centers` is a serialized leaf and is **exported, not recomputed** — a
`linspace` reimplementation would be a parity risk with zero upside.

## The manifest

`bots/joe-rs/artifact/manifest.json` copies joe's manifest (checkpoint
provenance intact) and adds:

- `tensor_schema: joe-net-v1` — the Rust loader (`src/nn/net.rs`) refuses to
  start on any schema-tag, name, shape, or dtype mismatch, and on any
  unexpected extra tensor.
- `safetensors_sha256`, `safetensors_size` — pins the derived file.
- `weights_sha256` (inherited) — pins the source `.eqx`.

## Checkout policy

Same rule as joe's `.eqx` (see `.gitignore`): the `.safetensors` is
gitignored and re-creatable; the committed manifest pins exactly which bytes
a checkout is missing. Fetch joe's `.eqx` first (its manifest's `r2_key`),
then run the converter. The content hash covers `model.safetensors`, so a
re-conversion forks joe-rs's rating identity — intended behavior, same as a
joe re-export.

## Quantization

Since 2026-08-18 the lineage is f16-in-f32: after fetching a new `.eqx`
from R2, `bots/joe/tools/quantize_artifact.py` rounds every weight through
IEEE f16 **before anything downstream runs**. The values stay float32 on
disk; only their precision changes. The `.eqx` is the single quantization
point — joe loads it directly and the converter below reads the same rounded
leaves — so no loader rounds at run time and there is one rounding
implementation in the repo.

Joe's manifest records the step: `quantized: "f16"`,
`pre_quantization_weights_sha256` (the R2 f32 original), and
`weights_sha256` moves to the rounded bytes. The tool refuses to run twice.
Evidence for the change: 3,544 rated games, regression excluded in two
rounds ([measurement](../../research/measurements/joe-rs-f16-quantization.md)).

## After a joe re-export

The converter alone is not the whole job. joe-rs tracks joe's artifact only
because someone re-runs this sequence; skipping it leaves the two bots
playing different networks while the docs claim otherwise.

```bash
.venv/bin/python bots/joe/tools/quantize_artifact.py     # f16 first, always
.venv/bin/python bots/joe-rs/tools/convert_artifact.py
export PATH="$HOME/.cargo/bin:$PATH"          # cargo is off the default PATH
cargo build --release --manifest-path bots/joe-rs/Cargo.toml
# the parity corpus is keyed to the network — rebuild it, do not reuse.
# archive first: capture_fixtures.py writes in place and skips existing
# .npz, so a partial run silently mixes two checkpoints' oracles.
mv data/joe/joe-rs-parity/games data/joe/joe-rs-parity/games.step<OLD>
.venv/bin/python bots/joe-rs/tools/capture_fixtures.py --play
.venv/bin/python bots/joe-rs/tools/make_synthetic_long.py  # needs the logs
.venv/bin/python bots/joe-rs/tools/capture_fixtures.py --capture
.venv/bin/python bots/joe-rs/tools/capture_fixtures.py --selection-golden
.venv/bin/python bots/joe-rs/tools/make_smoke_fixture.py   # committed fixture + golden
.venv/bin/pytest bots/joe-rs/tests/ -m joe
.venv/bin/python bots/joe-rs/tools/mutation_check.py
.venv/bin/pytest tests/test_joe_artifact_manifest.py
```

**No bot is downstream of joe-rs any more.** Two forks carried a byte copy of
these weights and both are gone: `morpheus-joe` on 2026-08-18, `unclejoe` on
2026-08-20. With them went the copy tool `scripts/joe_artifact_fanout.py` and
the staleness comparison it fed. Nothing detects a stale copy at play time — a
downstream bot loads the older weights, plays, and looks healthy — so a new
fork restores the tool, the checklist line, and the test, and reads
`joe-net-plan` §8.7 first. A sync forks the downstream bot's content hash,
exactly as a re-conversion forks joe-rs's; the registry records it, and a sync
between two rounds of the same contrast **voids both**, because the arms are
then different programs (joe-net-plan §8.7, §9).

**The checklist is not the mechanism.** Discipline is what failed here twice,
so `tests/test_joe_artifact_manifest.py` runs in the default suite and turns a
skipped step into a red result. What survives the forks is the half that a
digest comparison could never see anyway: it checks joe-rs's own manifest
against the bytes beside it, which catches a manifest that landed before its
weights or beside a truncated file. It costs milliseconds.

`--play` and `--capture` are split so `make_synthetic_long.py` can run
between them: it needs the natural games' `.in.log` files to pick the
longest, and the capture phase globs `*.in.log`, so it then picks the
fixture up with no extra flag.

Expect the tier-2 relative pins in `tests/test_parity.py` to need
re-measuring. They are a max over frames, so they do not transfer between
corpora — a rebuilt corpus can exceed them with a checkpoint that passed
before. Re-pin from the printed `[tier2]` line, then re-run
`mutation_check.py`: 13/13 killed is what makes a looser pin defensible.

The `.npz` surfaces are the JAX oracle's outputs for particular weights, and
the committed smoke fixture is a slice of them, so both go stale the moment
joe's weights change. The `<name>.joe-rs.log` self-goldens are the binary's
own replies for particular weights *and* a particular selection layer
(selection-plan S1), so they go stale on either change —
`--selection-golden` rebuilds them, and `make_smoke_fixture.py` records the
committed slice's. A stale committed fixture is the dangerous kind: a clean
checkout would compare the new binary against the old recorded replies.

**A capture or wire-replay mismatch is not always a weights problem.** joe-rs
plays its own Gumbel selection (2026-08-20) while Python joe takes the plain
argmax, so the two bots emit different moves *by design*. joe-rs's network is
graded on the `.npz` surfaces and its played path on its own goldens. Before
chasing a mismatch as a port bug, read
[parity.md](parity.md#what-parity-covers-the-network-not-the-selection-layer)
— on the step-29000 rebuild this looked exactly like a conversion fault and
was not one. A `.out.log` mismatch inside `capture_fixtures.py` has one more
cause: logs older than joe's current selection layer. Re-play them.
