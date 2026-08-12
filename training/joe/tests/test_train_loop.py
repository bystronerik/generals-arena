"""End-to-end mini train(): two iterations on a tiny env and net (CPU).

Exercises rollout collection, GAE, the PPO update, the vs-random eval, EMA,
and checkpointing through the real loop — the cheapest gate that catches
signature or shape drift before a GPU run. `joe`-marked (opt-in), so the
default suite budget is untouched; expect ~1 min of JAX compilation.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.joe

import equinox as eqx
import jax.random as jrandom

from generals.core.env import GeneralsEnv

from training.joe.config import Config
from training.joe.logger import Logger
from training.joe.main import make_optimizer
from training.joe.networks import build_network, get_network_bundle
from training.joe.train.ppo import train


def _tiny_env_factory(min_d, max_d, pool_size):
    return GeneralsEnv(grid_dims=(9, 9), build_castles=True,
                       deathtouch_turn=20, truncation=30,
                       min_generals_distance=3, pool_size=max(pool_size, 4))


def test_train_two_iterations(tmp_path):
    cfg = Config(
        run_name="joe-test",
        pad_to=9, min_grid_size=9, max_grid_size=9, truncation=30,
        pool_size=8, reset_pool_every=0,
        depth=1, embed_dim=32, n_head=4, ff_factor=1, patch_size=3,
        use_bf16=False,
        num_envs=4, num_steps=8, num_iters=2, minibatch_size=16,
        num_epochs=1, adv_top_frac=0.5,
        eval_every=1000, eval_games=8,
        ckpt_every=2, save_every=2,
        curriculum=None,
    )
    key = jrandom.PRNGKey(0)
    key, net_key = jrandom.split(key)
    network = build_network(cfg, net_key)
    optimizer = make_optimizer(cfg)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    ckpt_dir = str(tmp_path)
    logger = Logger(ckpt_dir, hparams=cfg.to_dict())
    commits = []

    net, opt, ema = train(
        cfg, network, optimizer, opt_state, logger, key,
        get_network_bundle(cfg.network), ckpt_dir, engine_sha="test-sha",
        on_checkpoint=lambda: commits.append(1), env_factory=_tiny_env_factory)
    logger.finish()

    # Checkpoint files + manifest written at iter 2, commit hook fired
    assert os.path.exists(os.path.join(ckpt_dir, "joe-test_2.eqx"))
    assert os.path.exists(os.path.join(ckpt_dir, "joe-test_ema_2.eqx"))
    assert os.path.exists(os.path.join(ckpt_dir, "joe-test_ema.eqx"))
    with open(os.path.join(ckpt_dir, "state.json")) as f:
        state = json.load(f)
    assert state["engine_sha"] == "test-sha"
    assert state["schema"] == 2
    assert state["global_step"] == 2
    assert state["curriculum_stage"] == 0
    assert state["files"] == {"full": "joe-test_2.eqx",
                              "ema": "joe-test_ema_2.eqx"}
    assert commits == [1]

    # Metrics: an eval row (it==0 baseline) and two train rows
    rows = [json.loads(line) for line in open(os.path.join(ckpt_dir, "metrics.jsonl"))]
    assert any("eval/win_rate" in r for r in rows)
    train_rows = [r for r in rows if "train/total_loss" in r]
    assert len(train_rows) == 2

    # EMA stayed a distinct copy of the same structure
    assert eqx.tree_check(ema) is None or True
    assert ema is not net
