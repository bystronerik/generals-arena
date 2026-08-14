# Joe vast.ai training (interruptible, R2-backed)

How to launch a Joe PPO run on a vast.ai interruptible instance. Durable
state lives in the Cloudflare R2 bucket `joe-training` under `joe/<run_name>/`.
This is **competition-rules training**, not live generals.io. It does not
feed `data/games/` or `data/ratings/`.

The Modal entry (`scripts/joe_modal_train.py`) stays the prototyping path.
The plan is
[`docs/research/strategies/joe-vast-training-plan.md`](../research/strategies/joe-vast-training-plan.md).

---

## Prerequisites

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt   # includes boto3 and the vastai CLI
vastai set api-key <key>              # once; key from console.vast.ai
```

The vast.ai CLI is the pip package `vastai`. The console script is
`.venv/bin/vastai`. The launcher finds it on `PATH`, then next to the
running interpreter, then at `.venv/bin/vastai`. You do not need a
separate curl install.

The API key must include the **secrets** permission (`api.secrets`).
A key without that route makes `sync-env` look successful while storing
nothing: `show env-vars` stays empty and `launch` reports the vars as
missing. Create a key with secrets access at
[console.vast.ai/manage-keys](https://console.vast.ai/manage-keys/), then
`vastai set api-key <key>`.

R2 credentials stay in the gitignored `.env` (or the shell environment) and
never appear in a script, a template, or a commit. The token is scoped to the
one `joe-training` bucket:

| Variable | Required | Purpose |
| --- | --- | --- |
| `R2_ENDPOINT_URL` | yes | S3-compatible endpoint for the `joe-training` bucket |
| `R2_ACCESS_KEY_ID` | yes | Token scoped to that one bucket |
| `R2_SECRET_ACCESS_KEY` | yes | Matching secret |
| `R2_BUCKET` | no | Defaults to `joe-training` |

Copy those into vast.ai **account-level encrypted env vars** so the
instance sees them without putting secrets on `--env` or in the image:

```bash
python scripts/joe_vast_train.py sync-env
```

`sync-env` prints key names only. It never prints values.

---

## Commands

```bash
# Phase 2 micro-smoke (4090-class, few iterations, S-sized net)
python scripts/joe_vast_train.py launch --smoke

# Upload code + launch.json only (no instance, no GPU spend)
python scripts/joe_vast_train.py launch --smoke --pack-only

# Full M run on interruptible H100 (Phase 3)
python scripts/joe_vast_train.py launch --tier M --run-name joe-M-vast-...

python scripts/joe_vast_train.py status --run-name <run>
python scripts/joe_vast_train.py resume --run-name <run>
python scripts/joe_vast_train.py destroy --run-name <run>
python scripts/joe_vast_train.py destroy --run-name <run> --purge-r2
```

`launch` packs `competition-module/` plus `training/joe` (same set as the
Modal `add_local_dir` mounts), uploads `code/<git_sha>.tar.gz`, writes
`launch.json` and `config.yaml`, then creates an interruptible instance
with `scripts/joe_vast_onstart.sh`. `resume` destroys the old instance
before it creates a replacement for the same `run_name`.

The bid for an offer is 5% more than the `min_bid` of that offer. A bid
equal to `min_bid` loses the machine to the next bidder that gives one
cent more. Give `--bid <$/hr>` to set the price yourself. The launcher
writes the price it sends to `launch.json` and to the instance record.

Never pipe the output through `tail` or `head`. Redirect to a file.

### Adopting an existing instance

`launch` and `resume` accept `--instance-id <id>` to use a vast.ai
instance you already rented instead of searching offers and creating one:

```bash
python scripts/joe_vast_train.py launch --smoke --instance-id 1234567
python scripts/joe_vast_train.py resume --run-name <run> --instance-id 1234567
```

The R2 side is unchanged: `launch` still uploads the code and writes
`launch.json` first, and `resume` still requires an existing
`launch.json`. The launcher then verifies the instance exists and is
running (`vastai show instance`), labels it with the run label, records
it in R2 (instance record + heartbeat, with no `offer_id` or
`bid_price` — there is no offer), and runs
`scripts/joe_vast_onstart.sh` on it over SSH (`vastai ssh-url`) with
`RUN_NAME` and `JOE_ROOT` exported. The remote onstart log is
`/workspace/joe-adopt-onstart.log`.

Constraints:

- `--bid` does not combine with `--instance-id`; there is no offer to
  bid on.
- `resume --instance-id` destroys any *other* instances recorded for
  the run (destroy-before-adopt) but never the adopted one.
- Teardown skips instances that vast.ai no longer lists and instances
  whose state (`cur_state`, `next_state`, `intended_status`) shows a
  destroy already in progress. A second `destroy instance` call on
  such an instance blocks until the teardown finishes.
- Adoption needs SSH access to the instance (your vast.ai account SSH
  key). The `R2_*` account env vars must already be synced
  (`sync-env`); the bootstrap reads them from the container
  environment without printing them.

### Restarting the trainer in place

`resume --instance-id <id>` restarts the training process without a new
instance: it keeps the adopted instance, then runs onstart again over
SSH. Onstart is idempotent, so a warm box skips the venv, the pip stack,
and the code download, and goes to `training.joe.vast_boot`, which
restores `state/latest.json`. You lose the steps after the last
checkpoint and nothing else.

Two rules for that path:

- **Stop the old trainer first** (`pkill -f training.joe.vast_boot`).
  The launcher does not stop it. Two trainers on one GPU write to the
  same R2 prefix.
- **Each boot needs a new `JOE_BOOT_ID`.** It names the local log and
  the `logs/train-<boot_id>.log` object in R2, so a repeated id writes
  over the log of the boot before it. Onstart makes a new id on every
  run and keeps the per-boot variables out of `/etc/environment`,
  which every later SSH login reads.

New code does not go to a running instance through `resume`: onstart
skips the download while `repo/.joe-code-sha` agrees with `launch.json`.
Use `launch --force` with the same `--run-name` and `--instance-id` to
upload a new tarball. `state/` stays, so the run continues.

---

## What the instance does

On every boot (including a same-machine resume after an outbid):

1. Install `boto3` if needed and write `logs/boot.json` (`phase=
   installing_pip_stack`). This is the first R2 object from the instance.
   `state/latest.json` is not a boot signal; it appears at the first
   checkpoint (`save_every`).
2. Install the pinned pip stack if it is not already present. The stock
   PyTorch image is Python 3.11; `jax==0.11.0` needs 3.12, so onstart
   creates `/workspace/joe/venv` with CPython 3.12 via `uv`. First-boot
   `jax[cuda12]` can take 10-20 min. Onstart tees stdout so `vastai logs`
   shows pip progress.
3. Download and unpack the code tarball from R2 (skip if the checksum
   marker matches).
4. Read `state/latest.json`. If it is present, restore that checkpoint
   set. If it is absent, start fresh from `config.yaml`.
5. Refuse to start if `lease/heartbeat.json` is fresher than five minutes
   and carries a different instance id.
6. Train through `training.joe.main.run` with the R2 uploader as
   `on_checkpoint`.

The training loop is the same loop Modal calls. The only difference is
the checkpoint hook and where the files live.
