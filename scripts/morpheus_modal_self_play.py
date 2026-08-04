#!/usr/bin/env python3
"""Modal entry point for Part 11: Morpheus self-play shard production.

Usage:
    modal run scripts/morpheus_modal_self_play.py \\
      --config training/morpheus/configs/self-play-smoke.json \\
      --games 2

Writes shards under /vol/morpheus/self_play/<run_id>/ on the Modal Volume
``morpheus-training`` (create it once with ``modal volume create morpheus-training``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from dataclasses import replace

import modal

REPO = Path(__file__).resolve().parents[1]

VOLUME = modal.Volume.from_name("morpheus-training", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "jax==0.11.0",
        "jaxlib==0.11.0",
        "torch==2.6.0",
    )
    .add_local_dir(
        str(REPO / "competition-module"),
        remote_path="/root/competition-module",
        copy=True,
    )
    .add_local_dir(str(REPO / "arena"), remote_path="/root/arena", copy=True)
    .add_local_dir(str(REPO / "bots"), remote_path="/root/bots", copy=True)
    .add_local_dir(str(REPO / "training"), remote_path="/root/training", copy=True)
    .add_local_dir(
        str(REPO / "scripts" / "configs"),
        remote_path="/root/scripts/configs",
        copy=True,
    )
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root"})
)

app = modal.App("morpheus-self-play")


@app.function(
    image=IMAGE,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    volumes={"/vol": VOLUME},
)
def produce(config_json: str, games: int, run_id: str) -> dict:
    from training.morpheus.self_play.driver import DriverConfig, run_batch
    from training.morpheus.self_play.verify import verify_directory

    config = DriverConfig.from_dict(json.loads(config_json))
    if games > 0:
        config = replace(config, games=int(games))
    out = Path("/vol/morpheus/self_play") / run_id
    result = run_batch(config, output=out, repo_root=Path("/root"))
    report = verify_directory(out)
    VOLUME.commit()
    return {
        "ok": report.ok,
        "output": str(result.output),
        "games": len(result.shards),
        "proportions": result.proportions,
        "verify": report.to_dict(),
    }


@app.local_entrypoint()
def main(
    config: str = "training/morpheus/configs/self-play-smoke.json",
    games: int = 2,
    run_id: str = "",
) -> None:
    cfg_path = Path(config)
    if not cfg_path.is_file():
        cfg_path = REPO / config
    payload = cfg_path.read_text(encoding="utf-8")
    rid = run_id or time.strftime("run-%Y%m%d-%H%M%S")
    result = produce.remote(payload, games, rid)
    print(json.dumps(result, indent=2))
