# Joe GCP training (G4 spot, R2-backed)

How to run a Joe PPO run on a Google Cloud G4 VM (RTX PRO 6000 Blackwell
Server Edition, 96 GB). Same durable-state model as
[joe-vast-train.md](joe-vast-train.md): everything lives in the R2 bucket
`joe-training` under `joe/<run_name>/`, and the VM runs the unchanged
`scripts/joe_vast_onstart.sh` + `training.joe.vast_boot`. A run can move
between vast.ai and GCP by running `resume` on the other launcher — R2
carries the checkpoints, config, and lease.

Measured context for this GPU class:
[joe-x16-pilots.md](../research/measurements/joe-x16-pilots.md).

---

## Prerequisites

```bash
brew install --cask google-cloud-sdk   # once
gcloud auth login                      # the account with the credits
gcloud config set project <project-id>
```

GPU quota (per region, separate metrics):

| Provisioning | Quota metric |
| --- | --- |
| Spot / Flex-start | `PREEMPTIBLE_NVIDIA_RTX_PRO_6000_GPUS` |
| On-demand | `NVIDIA_RTX_PRO_6000` family quota, plus global `GPUS_ALL_REGIONS` |

A freshly upgraded free-trial account often gets quota requests
auto-denied; retry with 1 GPU, one region, and a concrete justification.
Check what a region actually granted:

```bash
gcloud compute regions describe us-central1 \
  --format="table(quotas.filter('RTX_PRO_6000' in metric))"
```

R2 credentials stay in the gitignored `.env` exactly as for vast. The
launcher ships them over SSH into a root-only `/root/joe-env` on the VM —
never through VM metadata, the gcloud command line, or a commit.

## Commands

```bash
python scripts/joe_gcp_train.py zones          # zones offering G4
python scripts/joe_gcp_train.py launch --tier M7F4 --zone us-central1-b \
    --run-name joe-... --overrides '{...}'
python scripts/joe_gcp_train.py status  --run-name <run>
python scripts/joe_gcp_train.py resume  --run-name <run>
python scripts/joe_gcp_train.py watch   --run-name <run>   # auto-restart
python scripts/joe_gcp_train.py destroy --run-name <run> --purge-r2
```

`launch` packs the checkout, uploads `code/<sha>.tar.gz`, `config.yaml`,
and `launch.json` to R2 (same objects the vast launcher writes, plus
`provider: gcp`, zone, machine type), creates the VM, then bootstraps it
over SSH. Spot is the default; `--on-demand` needs the on-demand quota.

## Spot semantics (different from vast)

There is no bidding. A preemption **stops** the VM
(`--instance-termination-action=STOP`); the disk — venv, code, pip stack —
survives. `resume` (or the `watch` loop) restarts the same VM warm:
`gcloud compute instances start`, then the bootstrap reruns onstart, which
skips everything cached and restores `state/latest.json` from R2. You
lose the steps after the last uploaded checkpoint and nothing else.

The bootstrap kills any running trainer first (`pkill -f
training.joe.vast_boot`) — the two-trainers-one-prefix rule from the vast
doc applies unchanged — and waits up to 15 minutes for the NVIDIA driver
the Deep Learning VM image installs on first boot.

## Image

Default: family `common-cu129-ubuntu-2404-nvidia-580` in project
`deeplearning-platform-release` (a Deep Learning VM image whose R580
driver supports Blackwell; family checked 2026-08-25). If the family name
has rotated, list the current ones and pass `--image-family`:

```bash
gcloud compute images list --project deeplearning-platform-release \
  --filter="family~common" --format="value(family)" | sort -u
```

## Costs (2026-08-25 snapshot)

g4-standard-48 (1 GPU): ~$4.50/h on-demand, ~$1.94/h spot, identical
across US regions. Checkpoint uploads to R2 are ordinary internet egress
(~$0.12/GB — a few dollars per run). Both bill against promotional
credits. Prices move; re-check before a long run.
