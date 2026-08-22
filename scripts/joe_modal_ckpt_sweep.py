#!/usr/bin/env python3
"""Checkpoint-strength screen for a joe run (Modal, GPU).

Stage 1/2 of docs/research/strategies/joe-M-checkpoint-selection-plan.md:
narrow ~98 EMA checkpoints of one run down to a shortlist, cheaply, so that
only two or three ever reach a rated arena round.

The screen plays checkpoint against checkpoint inside the JAX env, batched:
``training/joe/train/evaluations.py::_evaluate_side_vs_ref`` under vmap +
lax.scan. Both seats play greedy, which is the deployment policy, so an
outcome is deterministic given a map.

**Nothing this script produces is an arena match.** Results go to
docs/research/measurements/ and never to data/games/, data/ratings/, or
data/bot_versions/ (root AGENTS.md, "What is never stored or rated"). The
screen has no verdict power; it only decides what is worth an arena round.

Two entry points. Stage the weights once, then sweep as often as you like:

    modal run scripts/joe_modal_ckpt_sweep.py::stage \
        --steps 5000,6000,10000 > /tmp/joe_stage.log 2>&1

    modal run --detach scripts/joe_modal_ckpt_sweep.py::sweep \
        --steps 5000,6000,10000 --pairs 6000:5000,10000:6000,10000:5000 \
        --n-maps 256 --tag pilot > /tmp/joe_sweep.log 2>&1 &
    modal app list          # then check startup a few minutes in (AGENTS.md)
    modal app logs <app-id>

**Always --detach for a real sweep.** Without it, Modal cancels the remote
call the moment the local process dies, and an hour of H100 time goes with
it. Every finished pair is also committed to the Volume, so a sweep that
loses its client is recoverable rather than lost:

    modal run scripts/joe_modal_ckpt_sweep.py::collect --tag s2

and re-launching ``sweep`` with the same --tag resumes from what the Volume
already holds instead of replaying it.

Never pipe the output through tail/head — redirect to a file (AGENTS.md).
All repo imports live inside functions: Modal re-imports this file in the
container, and module scope must survive both sides.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "joe-M-vast-20260813-0213"

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "jax[cuda12]==0.11.0",
        "equinox",
        "optax",
        "pyyaml",
    )
    .add_local_dir(
        str(REPO / "competition-module"),
        remote_path="/root/competition-module",
        copy=True,
    )
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root"})
    .add_local_file(str(REPO / "training" / "__init__.py"),
                    remote_path="/root/training/__init__.py")
    .add_local_dir(str(REPO / "training" / "joe"),
                   remote_path="/root/training/joe",
                   ignore=["**/__pycache__", "tests/**"])
)

app = modal.App("joe-ckpt-sweep")
# Weights live here, staged from R2 by ``stage``. A Volume rather than a
# modal.Secret carrying R2_*: the R2 token can write the bucket that holds
# every joe training run, and a one-time upload is cheaper than copying that
# token into a third-party service (plan section 3).
CKPT_VOL = modal.Volume.from_name("joe-M-ckpts", create_if_missing=True)


def _elo(score: float) -> float | None:
    """Pairwise Elo from a score rate. None at 0 or 1 (unbounded)."""
    if not 0.0 < score < 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval — honest at the extremes, where a screen often sits."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


@app.function(image=IMAGE, gpu="H100", timeout=6 * 3600,
              volumes={"/ckpts": CKPT_VOL})
def sweep_remote(run_name: str, steps: list[int], pairs: list[list[int]],
                 n_maps: int, scan_steps: int, map_seed: int,
                 pool_size: int, tag: str = "screen") -> dict:
    """Runs in the container: load every checkpoint, play every pair.

    Every finished pair is committed to the Volume before the next one
    starts. A sweep is an hour of H100 time and the local client holding it
    is the fragile end -- Modal cancels the remote call when that process
    dies, which cost 25 finished pairs once. Results now outlive the client,
    and re-launching with the same --tag skips what the Volume already has.
    """
    import os
    from dataclasses import replace

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import yaml

    from training.joe.config import Config
    from training.joe.env import (make_competition_env,
                                  preset_min_generals_distance)
    from training.joe.networks import build_network, get_network_bundle
    from training.joe.train.evaluations import _evaluate_side_vs_ref

    # Pool-generation and scan kernels cost real time to compile; the cache
    # makes a re-sweep at the same shape start playing almost immediately.
    os.makedirs("/ckpts/jax-cache", exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", "/ckpts/jax-cache")
    print(f"devices: {jax.devices()}", flush=True)

    root = f"/ckpts/{run_name}"
    with open(f"{root}/config.yaml") as f:
        cfg = Config.from_dict(yaml.safe_load(f), source="config.yaml")

    # Deployment parity, and the subtlest setting in this file. Training ran
    # use_bf16=True, but joe_export_bot.py does not forward use_bf16 to the
    # bot's network, so the *rated* bot runs an f32 forward. The screen must
    # match the bot, not the trainer, or it screens a program nobody plays.
    # use_bf16 is a static field, so the same weights load into either.
    cfg = replace(cfg, use_bf16=False)

    bundle = get_network_bundle(cfg.network)
    augment_fn = bundle["augment_obs"]
    greedy_fn = bundle["greedy_action"]
    init_obs_state_fn = bundle["init_obs_state"]

    # The final curriculum stage *is* the competition preset (distance 17+).
    # The pool only feeds auto-reset after a game ends, and we score first
    # finishes only, so it stays small.
    t0 = time.time()
    env = make_competition_env(preset_min_generals_distance(), None, pool_size)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    jax.block_until_ready(pool.armies)
    print(f"pool ready in {time.time() - t0:.1f}s (size {pool_size})",
          flush=True)

    single = init_obs_state_fn(cfg.pad_to, cfg.pad_to)
    obs_state = jax.tree.map(
        lambda x: jnp.tile(x, (n_maps, *([1] * x.ndim))), single)

    t0 = time.time()
    template = build_network(cfg, jrandom.PRNGKey(0))
    nets = {}
    for s in steps:
        path = f"{root}/ema_{s}.eqx"
        nets[s] = eqx.tree_deserialise_leaves(path, template)
    n_params = sum(x.size for x in jax.tree.leaves(
        eqx.filter(template, eqx.is_array)))
    print(f"loaded {len(nets)} checkpoints in {time.time() - t0:.1f}s "
          f"({n_params:,} params, use_bf16={cfg.use_bf16})", flush=True)

    # Anything a previous attempt at this tag already played.
    os.makedirs("/ckpts/results", exist_ok=True)
    partial = f"/ckpts/results/{tag}.jsonl"
    results = []
    if os.path.exists(partial):
        with open(partial) as f:
            results = [json.loads(ln) for ln in f if ln.strip()]
        print(f"resuming: {len(results)} pairs already on the volume",
              flush=True)
    done = {(r["a"], r["b"]) for r in results}

    for a, b in pairs:
        if (a, b) in done:
            continue
        # One key for both orientations, so A and B meet on the *same* maps
        # in both seats. That is the arena's --seat-policy alternate: seat
        # balance by construction, and map difficulty cancels inside the
        # matched pair instead of only in expectation. The same key across
        # every pair also puts the whole sweep on one map set.
        key = jrandom.PRNGKey(map_seed)
        t0 = time.time()
        agg = dict(wins=0, losses=0, draws=0, finished=0)
        for net_player in (0, 1):
            fin, won, lost, drew = _evaluate_side_vs_ref(
                env, nets[a], nets[b], key, scan_steps, n_maps,
                obs_state, obs_state, augment_fn, greedy_fn, pool, net_player)
            agg["finished"] += int(jnp.sum(fin))
            agg["wins"] += int(jnp.sum(won))
            agg["losses"] += int(jnp.sum(lost))
            agg["draws"] += int(jnp.sum(drew))
        dt = time.time() - t0
        games = 2 * n_maps
        w, l, d, fin_n = (agg["wins"], agg["losses"], agg["draws"],
                          agg["finished"])
        decisive = w + l
        score = (w + 0.5 * d) / fin_n if fin_n else float("nan")
        lo, hi = _wilson(w, decisive)
        row = {
            "a": a, "b": b, "games": games, "finished": fin_n,
            "wins": w, "losses": l, "draws": d,
            "decisive": decisive,
            "decisive_win_rate": (w / decisive) if decisive else None,
            "decisive_ci95": [lo, hi],
            "score_rate": score,
            "elo_a_minus_b": _elo(score),
            "elo_ci95": [_elo(lo), _elo(hi)],
            "draw_rate": d / fin_n if fin_n else None,
            "seconds": dt,
            "games_per_second": games / dt if dt else None,
        }
        results.append(row)
        with open(partial, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        CKPT_VOL.commit()
        note = "" if fin_n == games else f"  !! {games - fin_n} UNFINISHED"
        print(f"  {a} vs {b}: {w}W {l}L {d}D  "
              f"score {score:.3f}  elo {row['elo_a_minus_b'] or float('nan'):+.1f}  "
              f"{dt:.1f}s ({row['games_per_second']:.1f} games/s){note}",
              flush=True)

    return {
        "run_name": run_name, "steps": steps, "n_maps": n_maps,
        "games_per_pair": 2 * n_maps, "scan_steps": scan_steps,
        "map_seed": map_seed, "pool_size": pool_size,
        "n_params": n_params, "use_bf16": cfg.use_bf16,
        "gpu": "H100", "results": results,
    }


def bradley_terry(results: list[dict], anchor: int | None = None) -> dict:
    """Joint BT fit over the whole pair matrix, in Elo, with standard errors.

    Per-pair Elo read off a score rate does *not* chain: the pilot measured
    +33.7 (6000>5000) and +287.5 (10000>6000) against a direct +394.6 for
    10000>5000, a 73 Elo transitivity gap. The arena's numbers add up exactly
    because they come from one joint fit; so must these, or a ranking over
    many checkpoints inherits that gap at every hop.

    Draws count as half a game to each side (the screen's draw rate is under
    1%, far from where a Davidson tie term would earn its parameter).
    """
    import numpy as np

    ids = sorted({r["a"] for r in results} | {r["b"] for r in results})
    idx = {s_: i for i, s_ in enumerate(ids)}
    n = len(ids)
    scale = math.log(10.0) / 400.0

    S = np.zeros((n, n))   # score of i against j
    N = np.zeros((n, n))   # games between i and j
    for r in results:
        i, j = idx[r["a"]], idx[r["b"]]
        s_ij = r["wins"] + 0.5 * r["draws"]
        S[i, j] += s_ij
        S[j, i] += r["finished"] - s_ij
        N[i, j] += r["finished"]
        N[j, i] += r["finished"]

    theta = np.zeros(n)
    for _ in range(500):
        d = theta[:, None] - theta[None, :]
        p = 1.0 / (1.0 + np.exp(-scale * d))
        grad = scale * np.sum(S - N * p, axis=1)
        w = scale * scale * N * p * (1.0 - p)
        H = np.diag(w.sum(axis=1)) - w
        # Rank-deficient by one (ratings are relative); pin the mean.
        H = H + np.ones((n, n)) / n
        step = np.linalg.solve(H, grad)
        theta = theta + step
        theta = theta - theta.mean()
        if np.max(np.abs(step)) < 1e-10:
            break

    d = theta[:, None] - theta[None, :]
    p = 1.0 / (1.0 + np.exp(-scale * d))
    w = scale * scale * N * p * (1.0 - p)
    H = np.diag(w.sum(axis=1)) - w
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    if anchor is not None and anchor in idx:
        theta = theta - theta[idx[anchor]]

    order = sorted(range(n), key=lambda i: -theta[i])
    return {
        "anchor": anchor,
        "ratings": [{"step": ids[i], "elo": float(theta[i]),
                     "se": float(se[i])} for i in order],
        "cov": cov.tolist(), "ids": ids,
    }


@app.local_entrypoint()
def collect(tag: str, run_name: str = DEFAULT_RUN, n_maps: int = 1024,
            map_seed: int = 20260821):
    """Rebuild a sweep JSON from the per-pair rows on the Volume.

    The recovery path when a client dies mid-sweep: the remote committed
    every finished pair, so the games are not lost even though the call that
    would have returned them is gone.
    """
    raw = b"".join(CKPT_VOL.read_file(f"results/{tag}.jsonl"))
    results = [json.loads(ln) for ln in raw.decode().splitlines() if ln.strip()]
    steps = sorted({r["a"] for r in results} | {r["b"] for r in results})
    out = {
        "run_name": run_name, "steps": steps, "n_maps": n_maps,
        "games_per_pair": 2 * n_maps, "map_seed": map_seed,
        "gpu": "H100", "tag": tag, "recovered_from_volume": True,
        "results": results,
    }
    out["bradley_terry"] = bradley_terry(results)
    dest = REPO / "docs" / "research" / "measurements" / f"joe-ckpt-{tag}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    games = sum(r["games"] for r in results)
    print(f"recovered {len(results)} pairs, {games:,} games")
    _print_fit(out["bradley_terry"])
    print(f"wrote {dest.relative_to(REPO)}")


@app.local_entrypoint()
def fit(path: str, anchor: int = 0):
    """Re-fit a saved sweep JSON without spending GPU time."""
    src = Path(path)
    out = json.loads(src.read_text())
    out["bradley_terry"] = bradley_terry(out["results"],
                                         anchor=anchor or None)
    src.write_text(json.dumps(out, indent=1) + "\n")
    _print_fit(out["bradley_terry"])


def _print_fit(bt: dict) -> None:
    print(f"\nBradley-Terry (anchor {bt['anchor'] or 'mean'}):")
    for row in bt["ratings"]:
        print(f"  step {row['step']:>6}  {row['elo']:+9.1f} ± {row['se']:.1f}")


@app.local_entrypoint()
def stage(steps: str, run_name: str = DEFAULT_RUN):
    """Copy R2 EMA blobs + the run config into the Modal Volume (local)."""
    import sys
    import tempfile

    sys.path.insert(0, str(REPO))
    from training.joe.store import R2Store, load_dotenv

    load_dotenv(str(REPO / ".env"))
    store = R2Store.from_env()
    want = [int(s) for s in steps.split(",") if s.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg_local = tmp / "config.yaml"
        store.download_run_file(run_name, "config.yaml", str(cfg_local))
        uploads = [(cfg_local, f"/{run_name}/config.yaml")]

        for s in want:
            state = store._get_json(store.key(run_name, "state",
                                              f"state-{s}.json"))
            if state is None:
                raise SystemExit(f"No state-{s}.json in R2 for '{run_name}'")
            local = tmp / f"ema_{s}.eqx"
            store.download_file(
                store.key(run_name, "checkpoints", state["files"]["ema"]),
                str(local))
            print(f"  fetched step {s} ({local.stat().st_size:,} bytes)",
                  flush=True)
            uploads.append((local, f"/{run_name}/ema_{s}.eqx"))

        with CKPT_VOL.batch_upload(force=True) as batch:
            for local, remote in uploads:
                batch.put_file(str(local), remote)
    print(f"staged {len(want)} checkpoints + config to volume joe-M-ckpts")


@app.local_entrypoint()
def sweep(steps: str, pairs: str = "", run_name: str = DEFAULT_RUN,
          n_maps: int = 256, scan_steps: int = 0, map_seed: int = 20260821,
          pool_size: int = 1024, tag: str = "screen"):
    """pairs: "a:b,a:b,..." — empty means every unordered pair (round-robin)."""
    want = [int(s) for s in steps.split(",") if s.strip()]
    if pairs:
        plist = [[int(x) for x in p.split(":")] for p in pairs.split(",")]
    else:
        plist = [[a, b] for i, a in enumerate(want) for b in want[i + 1:]]
    # +2 ticks past the env's own truncation so a game that ends exactly on
    # the draw turn still registers as finished inside the scan.
    scan = scan_steps or 1202

    print(f"sweep {tag}: {len(want)} checkpoints, {len(plist)} pairs, "
          f"{2 * n_maps} games/pair = {2 * n_maps * len(plist)} games",
          flush=True)
    out = sweep_remote.remote(run_name, want, plist, n_maps, scan, map_seed,
                              pool_size, tag)

    out["tag"] = tag
    out["bradley_terry"] = bradley_terry(out["results"])
    dest = REPO / "docs" / "research" / "measurements" / f"joe-ckpt-{tag}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    tot = sum(r["seconds"] for r in out["results"])
    games = sum(r["games"] for r in out["results"])
    print(f"\n{games} games in {tot:.1f}s play time "
          f"({games / tot:.1f} games/s overall)")
    _print_fit(out["bradley_terry"])
    print(f"wrote {dest.relative_to(REPO)}")
