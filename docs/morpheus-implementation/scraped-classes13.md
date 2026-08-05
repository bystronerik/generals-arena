# Scraped classes 1–3 → Part 14 research train

Build a stratified classes 1–3 curriculum from all reconstruction rounds,
materialize the sample buffer on Modal (64 CPU shards), and train with class-1
sampling weight only.

## Defaults (locked)

- **Classes:** 1–3 (`--full-start-count 0`).
- **Stratify:** global 1:1 queried-player wins and losses (`n = n_lose`).
- **Top wins:** exactly 50% of kept wins from **ResBot**, **Kubic**, **thor**,
  split in equal thirds (remainder by sorted name: Kubic, ResBot, thor).
- **Other wins:** remaining 50% from all other reconstruction players.
- **Seats:** `corpus-index.sample_seat` / queried player.
- **Materialize:** `n_particles=4`; 64 Modal containers × `cpu=1` × `memory=512`.
- **Train:** `class_balance: {"1": 1.0}`; `promotable_main_run: false`.

## Operator sequence

```bash
python scripts/morpheus_curriculum.py build \
  --panel scripts/configs/morpheus/scraped-classes13.json \
  --trajectories-parent data/trajectories \
  --classes 1 2 3 \
  --full-start-count 0 \
  --require-sample-seat \
  --stratify-global-wdl \
  --top-win-players ResBot Kubic thor \
  --top-win-fraction 0.5 \
  --output training/morpheus/manifests/scraped-classes13.json

python scripts/morpheus_curriculum.py verify \
  --manifest training/morpheus/manifests/scraped-classes13.json \
  --require-sample-seat

modal volume put morpheus-training data/trajectories /morpheus/trajectories
modal volume put morpheus-training \
  training/morpheus/manifests/scraped-classes13.json \
  /morpheus/manifests/scraped-classes13.json

# Smoke (optional):
# modal run scripts/morpheus_modal_materialize.py \
#   --shards 2 --n-particles 4 --max-items 32

modal run scripts/morpheus_modal_materialize.py --shards 64 --n-particles 4

modal run scripts/morpheus_modal.py::train \
  --config training/morpheus/configs/scraped-classes13-run.json \
  --run-id scraped-classes13-<date>
```

Local materialize (`scripts/morpheus_materialize.py` → `data/morpheus/trainer/buffer`)
remains available for debugging. The Modal path writes `/vol/morpheus/buffer`
on volume `morpheus-training` for Part 14 train.

## Related

- Class-1 pilot: [pilot-class1-scraped.md](pilot-class1-scraped.md)
- Curriculum rules: [../bots/morpheus/curriculum.md](../bots/morpheus/curriculum.md)
