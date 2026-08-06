# Scraped classes 1–5 → Part 14 research train

Build a stratified classes 1–4 curriculum from reconstruction rounds, add
class-5 full starts on new competition seeds, materialize the sample buffer
(reuse existing class 1–3 `*.sample.npz` via `skip_existing`), and train with
equal class weight on 1–5.

## Defaults (locked)

- **Classes:** 1–5 (`--full-start-count 512` → 1024 seat samples).
- **Stratify:** global 1:1 queried-player wins and losses (`n = n_lose`).
- **Top wins:** exactly 50% of kept wins from **ResBot**, **Kubic**, **thor**,
  split in equal thirds (remainder by sorted name: Kubic, ResBot, thor).
- **Other wins:** remaining 50% from all other reconstruction players.
- **Seats:** `corpus-index.sample_seat` / queried player for 1–4; both seats
  for class 5.
- **Materialize:** `n_particles=4`; skip existing class 1–3 files; write new
  class 4 and class 5. Class 5 needs no trajectory.
- **Train:** `class_balance: {"1":1,"2":1,"3":1,"4":1}` (class 5 omitted;
  pass-policy placeholder). `promotable_main_run: false`.
  `batch_size: 32`, `max_steps: 6000` (~5 passes), `snapshot_every_steps: 1000`.

## Operator sequence

```bash
python scripts/morpheus_curriculum.py build \
  --panel scripts/configs/morpheus/scraped-classes15.json \
  --trajectories-parent data/trajectories \
  --classes 1 2 3 4 5 \
  --full-start-count 512 \
  --require-sample-seat \
  --stratify-global-wdl \
  --top-win-players ResBot Kubic thor \
  --top-win-fraction 0.5 \
  --output training/morpheus/manifests/scraped-classes15.json

python scripts/morpheus_curriculum.py verify \
  --manifest training/morpheus/manifests/scraped-classes15.json \
  --require-sample-seat

# Local smoke (optional): only new classes, small cap
python scripts/morpheus_materialize.py \
  --manifest training/morpheus/manifests/scraped-classes15.json \
  --output data/morpheus/trainer/buffer \
  --classes 4 5 --max-items 64 --n-particles 4

# Modal: upload manifest, rematerialize (skip_existing keeps 1–3)
modal volume put morpheus-training \
  training/morpheus/manifests/scraped-classes15.json \
  /morpheus/manifests/scraped-classes15.json

modal run scripts/morpheus_modal_materialize.py \
  --manifest /vol/morpheus/manifests/scraped-classes15.json \
  --shards 96 --n-particles 4 --classes 4,5

modal run scripts/morpheus_modal.py::train \
  --config training/morpheus/configs/scraped-classes15-run.json \
  --run-id scraped-classes15-<date>
```

Reuse class 1–3 samples only when the seed contract and format match the
current materialize path. Rematerialize those classes if the contract changed.

## Related

- Classes 1–3 only: [scraped-classes13.md](scraped-classes13.md)
- Curriculum rules: [../bots/morpheus/curriculum.md](../bots/morpheus/curriculum.md)
