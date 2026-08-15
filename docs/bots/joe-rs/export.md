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
2. Exactly 100 tensors with the `joe-net-v1` names/shapes, all float32,
   summing to 8,556,250 parameters (= the manifest's `n_params`).
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

## After a joe re-export

The converter alone is not the whole job. joe-rs tracks joe's artifact only
because someone re-runs this sequence; skipping it leaves the two bots
playing different networks while the docs claim otherwise.

```bash
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
.venv/bin/python bots/joe-rs/tools/make_smoke_fixture.py   # committed fixture
.venv/bin/pytest bots/joe-rs/tests/ -m joe
.venv/bin/python bots/joe-rs/tools/mutation_check.py
# Two bots downstream now: joe -> joe-rs -> {unclejoe, morpheus-joe}. Both
# track joe-rs's converted artifact, not joe's .eqx, because the joe-rs vs
# unclejoe contrast has to isolate the tactics layer and the joe-rs vs
# morpheus-joe contrast the search stack, and neither isolates anything
# unless the arms run identical weights. One script walks both.
.venv/bin/python scripts/joe_artifact_fanout.py
cargo build --release --manifest-path bots/unclejoe/Cargo.toml
cargo build --release --manifest-path bots/morpheus-joe/Cargo.toml
# morpheus-joe also copies joe-rs's *source* byte for byte, which is how its
# forward pass inherits this corpus's proof. Both copies are asserted here:
.venv/bin/pytest tests/test_joe_source_fanout.py
```

Nothing detects a skipped sync at play time — a downstream bot would load the
older weights, play, and look healthy — so the step belongs here or nowhere.
`joe_artifact_fanout.py` verifies `safetensors_sha256` on both sides and writes
the weights through a temporary file, which makes a *partial* sync (manifest
copied, weights not) fail loudly instead of playing. `--check` answers "is
every downstream current?" without writing, and `--bot <name>` narrows it to
one. `bots/unclejoe/tools/sync_artifact.py` still does unclejoe alone and is
unchanged. A sync forks the downstream bot's content hash, exactly as a
re-conversion forks joe-rs's; the registry records it.

**The checklist is not the mechanism.** Discipline is what failed here twice,
so `tests/test_joe_source_fanout.py` runs in the default suite and turns a
skipped step into a red result: it compares every downstream manifest's digest
against joe-rs's, checks each manifest against the bytes beside it, and asserts
that morpheus-joe's eleven copied source files are byte-identical to joe-rs's.
It costs milliseconds. Note the consequence for a measurement round: a sync
between two rounds of the same contrast **voids both**, because the arms are
then different programs (joe-net-plan §8.7, §9).

`--play` and `--capture` are split so `make_synthetic_long.py` can run
between them: it needs the natural games' `.in.log` files to pick the
longest, and the capture phase globs `*.in.log`, so it then picks the
fixture up with no extra flag.

Expect the tier-2 relative pins in `tests/test_parity.py` to need
re-measuring. They are a max over frames, so they do not transfer between
corpora — a rebuilt corpus can exceed them with a checkpoint that passed
before. Re-pin from the printed `[tier2]` line, then re-run
`mutation_check.py`: 9/9 killed is what makes a looser pin defensible.

The `.npz` surfaces are the JAX oracle's outputs for particular weights, and
the committed smoke fixture is a slice of them, so both go stale the moment
joe's weights change. A stale smoke fixture is the dangerous one: it is
committed, so a clean checkout would compare the new binary against the old
oracle's recorded replies.
