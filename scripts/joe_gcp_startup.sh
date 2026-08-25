#!/bin/bash
# GCP startup script for Joe training (VM metadata, runs as root on every
# boot — including every spot restart). Carries NO secrets: the launcher
# delivers /root/joe-env and /root/joe-onstart.sh once over SSH, and both
# persist on the boot disk across spot stops, so a preemption recovery
# needs no SSH at all — `instances start` is enough.
exec > >(tee -a /root/joe-startup.log) 2>&1
echo "joe gcp startup $(date -u +%Y-%m-%dT%H:%M:%SZ)"

while [ ! -f /root/joe-env ] || [ ! -f /root/joe-onstart.sh ]; do
  echo "waiting for /root/joe-env and /root/joe-onstart.sh"
  sleep 15
done
chmod 600 /root/joe-env

# The Deep Learning VM image installs the NVIDIA driver on first boot.
n=0
while [ "$n" -lt 60 ]; do
  nvidia-smi -L >/dev/null 2>&1 && break
  n=$((n + 1))
  sleep 15
done
nvidia-smi -L >/dev/null 2>&1 || { echo "no NVIDIA driver after wait"; exit 1; }

# Two trainers on one GPU write to the same R2 prefix (vast rule).
pkill -f training.joe.vast_boot || true
sleep 2

set -a
. /root/joe-env
set +a
# Metadata startup scripts run without HOME; onstart's `set -u` needs it
# (uv installs under $HOME). Measured 2026-08-25: unset HOME killed
# onstart at the PATH export.
export HOME=/root
echo "starting onstart run=${RUN_NAME:-UNSET}"
exec bash /root/joe-onstart.sh > /root/joe-onstart.log 2>&1
