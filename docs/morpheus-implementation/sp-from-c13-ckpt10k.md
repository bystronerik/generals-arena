# Self-play research train from scraped-classes13 ckpt-00010000

Research-scope only (`promotable_main_run: false`). Seed checkpoint:
`scraped-classes13-2026-08-08` / `ckpt-00010000-300811cf4add` (exported at
`bots/morpheus/artifact`).

## Locked choices

- Goal: research strength gain; stop on training loss
- 10k natural-end games target; 2-hour wall budget; ≤99 Modal CPUs
- 70/30 league + bootstrap panel; training-small search; pilot root noise on
- New run id; SP buffer only; export + competition gate at the end

Progress: workers emit `[sp] heartbeat` / `game_done` lines; the local
entrypoint prints `worker_finished` JSON as each worker returns
(`order_outputs=False`).

## Stage 0 — prepare + probe

```bash
python scripts/morpheus_sp_prepare_league.py \
  --checkpoint bots/morpheus/artifact \
  --output data/morpheus/leagues/sp-from-c13-ckpt10k/league.json

modal run scripts/morpheus_modal_self_play.py \
  --config training/morpheus/configs/sp-from-c13-ckpt10k-self-play.json \
  --games 20 --workers 5 --run-id sp-probe-2026-08-08
```

Read `mean_turns` and `projected_games_per_hour_at_99`. Go/no-go for 10k inside
the remaining wall time.

## Stage 1 — produce

```bash
modal run scripts/morpheus_modal_self_play.py \
  --config training/morpheus/configs/sp-from-c13-ckpt10k-self-play.json \
  --games 10000 --workers 99 --run-id sp-from-c13-ckpt10k-2026-08-08
```

Shards: `/vol/morpheus/self_play/<run_id>/`.

## Stage 2 — ingest

```bash
modal run scripts/morpheus_modal_sp_ingest.py \
  --run-id sp-from-c13-ckpt10k-2026-08-08 \
  --output-name sp-from-c13-ckpt10k
```

Buffer: `/vol/morpheus/buffer_sp/sp-from-c13-ckpt10k`.

## Stage 3 — train

Requires the source checkpoint on the volume:

`/vol/morpheus/runs/scraped-classes13-2026-08-08/ckpt-00010000-300811cf4add`

```bash
modal run scripts/morpheus_modal.py::train \
  --config training/morpheus/configs/sp-from-c13-ckpt10k-train.json \
  --run-id sp-train-from-c13-ckpt10k-2026-08-08
```

## Stage 4 — export + gate

Export the new snapshot into `bots/morpheus/artifact`, then:

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

## Configs

- Produce: `training/morpheus/configs/sp-from-c13-ckpt10k-self-play.json`
- Train: `training/morpheus/configs/sp-from-c13-ckpt10k-train.json`
