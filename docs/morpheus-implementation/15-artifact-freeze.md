# Part 15: Artifact freeze

## Deliverable

Freeze one calibrated checkpoint as a self-contained competition artifact under
`bots/morpheus/`.

The closure contains inference and search code, exact quantized weights, a
model manifest, stdio entry points, and deterministic build input.

**Touches**

- `bots/`: replace the prior candidate weights and manifest with exact frozen
  bytes.
- `arena/`: use bundle, fingerprint, registry, and submission harness paths.
- `scripts/`: use the download and freeze entry point.
- `data/bot_versions/`: add the content hash through the normal parent match
  registration path.

## Prerequisites

- [Part 08: Submission-shaped harness](08-submission-harness.md)
- [Part 09: Online qualification](09-online-qualification.md)
- [Part 14: Trainer and checkpoints](14-trainer-checkpoints.md)

## Source specifications

- [Frozen artifact](../bots/morpheus/evaluation.md#frozen-artifact)
- [Competition verification gate](../bots/morpheus/evaluation.md#competition-verification-gate)
- [Bot version registry](../arena/bot-version-registry.md)

## Defaults and replacement measurement

This part has no open default. It uses the exact tensor, action, architecture,
quantization, runtime, and deployment configuration accepted by earlier parts.

Any change to code, weights, manifest, or deterministic build input creates a
new artifact and requires all gates again.

## Implementation boundary

The manifest identifies schema versions, architecture, quantization, runtime,
training run, deployment configuration, and full weight SHA-256.

Forbid downloads, mutable external paths, and `latest` links. The competition
build has no network.

Build the deterministic submission zip through `arena.bundle`. Run the
submission harness on the extracted zip, not only on the repository
`run.sh`.

Let `arena.matches.run_match` or the tournament parent register the content
hash. Do not write `data/bot_versions/morpheus.json` by hand.

## Isolated test

```bash
python scripts/morpheus_freeze.py \
  --run-id <run_id> --checkpoint <checkpoint_id> \
  --bot-dir bots/morpheus
```

```bash
python -m pytest bots/morpheus/tests/test_manifest.py \
  -m morpheus -q
python -m pytest tests/test_fingerprint.py tests/test_bundle.py -q
```

```bash
python -m arena.bundle morpheus --force
```

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

```bash
python -m arena.matches.submission \
  data/bundles/morpheus-<content_hash>.zip \
  --opponent bots/smoke/run.sh \
  --mode competition --seed 0
```

```bash
python -m arena.matches.run_match \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0 --round morpheus-freeze
python -m arena.records.registry --verify
```

## Specification gaps

The current specs name the manifest fields but do not define a JSON schema,
format version, or compatibility policy. Part 04 must define them before this
part.

The judge CPU remains unspecified. A successful local submission check is
necessary but cannot prove identical judge latency.

## Exit criterion

Answer `yes` only if the weight SHA matches the manifest, a one-byte weight
change moves the content hash, the standalone bundle has no external
dependency, the competition match ends normally, the submission harness
accepts with zero faults, and the registry verifies the exact hash.

Answer `no` for any mutable input, hash mismatch, bundle difference, resource
failure, forced EOF exit, or registry error.
