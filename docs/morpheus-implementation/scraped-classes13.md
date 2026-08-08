# Scraped classes 1–5 → Part 14 research train

Build a stratified classes 1–4 curriculum from reconstruction rounds, add
class-5 full starts, materialize on Modal (96 CPU shards), and train with equal
class weights on 1, 2, 3, 4, and 5.

## Defaults (locked)

- **Classes:** 1, 2, 3, 4, and 5 (`--full-start-count 512` → 1024 class-5
  seat samples).
- **Stratify:** global 1:1 queried-player wins and losses (`n = n_lose`, then
  capped to the largest n the top/other win pools can fill); draws excluded.
- **Top wins:** exactly 75% of kept wins from **Kubic**, **ResBot**, **bca**,
  **nanomena**, **thor**, **Mattz**, split equally (remainder by sorted name).
- **Other wins:** remaining 25% from all other reconstruction players.
- **Seats:** `corpus-index.sample_seat` / queried player.
- **Materialize:** `n_particles=4`; 96 Modal containers × `cpu=1` × `memory=512`.
  Skip samples already under `/vol/morpheus/buffer`. Purge orphans not in the
  new manifest before writing.
- **Train:** `class_balance` for classes 1–5; `window_size` ≥ buffer size;
  `promotable_main_run: false`.

## Operator sequence

```bash
python scripts/morpheus_curriculum.py build \
  --panel scripts/configs/morpheus/scraped-classes13.json \
  --trajectories-parent data/trajectories \
  --classes 1 2 3 4 5 \
  --full-start-count 512 \
  --require-sample-seat \
  --stratify-global-wdl \
  --top-win-players Kubic ResBot bca nanomena thor Mattz \
  --top-win-fraction 0.75 \
  --output training/morpheus/manifests/scraped-classes13.json

python scripts/morpheus_curriculum.py verify \
  --manifest training/morpheus/manifests/scraped-classes13.json \
  --require-sample-seat

modal volume put morpheus-training data/trajectories /morpheus/trajectories
modal volume put morpheus-training \
  training/morpheus/manifests/scraped-classes13.json \
  /morpheus/manifests/scraped-classes13.json

modal run scripts/morpheus_modal_materialize.py --shards 96 --n-particles 4

modal run scripts/morpheus_modal.py::train \
  --config training/morpheus/configs/scraped-classes13-run.json \
  --run-id scraped-classes13-<date>
```

## Related

- Class-1 pilot: [pilot-class1-scraped.md](pilot-class1-scraped.md)
- Curriculum rules: [../bots/morpheus/curriculum.md](../bots/morpheus/curriculum.md)
