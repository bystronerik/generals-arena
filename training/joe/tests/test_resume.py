"""Kill-and-resume through the real loop (joe-marked, opt-in).

Phase 0 gate of docs/research/strategies/joe-vast-training-plan.md: a run
resumed from ``state.json`` continues at the same global step and
curriculum stage, and never rewrites a step-named checkpoint an earlier
process wrote. Uses ``main.run()`` so resume detection itself is under
test. Expect a few minutes of JAX compilation on the tiny env.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

pytestmark = pytest.mark.joe

from generals.core.env import GeneralsEnv

from training.joe.config import Config
from training.joe.main import run


def _tiny_env_factory(min_d, max_d, pool_size):
    return GeneralsEnv(grid_dims=(9, 9), build_castles=True,
                       deathtouch_turn=20, truncation=30,
                       min_generals_distance=3, pool_size=max(pool_size, 4))


def _cfg(**overrides):
    base = dict(
        run_name="joe-resume",
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
    base.update(overrides)
    return Config(**base)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _state(ckpt_dir):
    with open(os.path.join(ckpt_dir, "state.json")) as f:
        return json.load(f)


def test_kill_and_resume_continues_global_step(tmp_path):
    ckpt_dir = str(tmp_path)
    run(_cfg(), ckpt_dir, engine_sha="test-sha", env_factory=_tiny_env_factory)
    ckpt2 = os.path.join(ckpt_dir, "joe-resume_2.eqx")
    hash_before = _sha(ckpt2)
    assert _state(ckpt_dir)["global_step"] == 2

    # "Kill": the first process is gone; a new one starts from the same
    # ckpt_dir with the run's total target raised to 4.
    run(_cfg(num_iters=4), ckpt_dir, engine_sha="test-sha",
        env_factory=_tiny_env_factory)

    state = _state(ckpt_dir)
    assert state["global_step"] == 4
    assert state["files"] == {"full": "joe-resume_4.eqx",
                              "ema": "joe-resume_ema_4.eqx"}
    assert os.path.exists(os.path.join(ckpt_dir, "joe-resume_4.eqx"))
    # No clobbering: the step-2 checkpoint is byte-identical
    assert _sha(ckpt2) == hash_before

    # The resumed loop ran global steps 3 and 4, not 1 and 2 again
    with open(os.path.join(ckpt_dir, "metrics.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    train_steps = [r["step"] for r in rows if "train/total_loss" in r]
    assert train_steps == [1, 2, 3, 4]


def test_curriculum_stage_survives_resume(tmp_path, capsys):
    ckpt_dir = str(tmp_path)
    curriculum = [
        dict(min_generals_distance=3, max_generals_distance=6),
        dict(min_generals_distance=3, max_generals_distance=6,
             win_rate_threshold=0.0),
    ]
    # The it==0 baseline eval always runs; threshold 0.0 advances to
    # stage 1 immediately, so the state at step 2 records stage 1.
    run(_cfg(curriculum=curriculum), ckpt_dir, engine_sha="test-sha",
        env_factory=_tiny_env_factory)
    state = _state(ckpt_dir)
    assert state["curriculum_stage"] == 1
    saved_wr = state["last_eval_wr"]

    capsys.readouterr()
    run(_cfg(curriculum=curriculum, num_iters=4), ckpt_dir,
        engine_sha="test-sha", env_factory=_tiny_env_factory)
    out = capsys.readouterr().out
    assert "curriculum stage 1" in out          # resume banner
    assert "stage 1)" in out                    # pool built for stage 1

    state = _state(ckpt_dir)
    assert state["curriculum_stage"] == 1
    # No eval ran during the resumed steps (eval_every=1000, it>0), so the
    # gate counter must have survived the restart exactly.
    assert state["last_eval_wr"] == saved_wr


def test_resume_rejects_shrunken_curriculum(tmp_path):
    ckpt_dir = str(tmp_path)
    curriculum = [
        dict(min_generals_distance=3, max_generals_distance=6),
        dict(min_generals_distance=3, max_generals_distance=6,
             win_rate_threshold=0.0),
    ]
    run(_cfg(curriculum=curriculum), ckpt_dir, engine_sha="test-sha",
        env_factory=_tiny_env_factory)
    assert _state(ckpt_dir)["curriculum_stage"] == 1

    # The config lost a stage since the checkpoint: fail loudly, no clamp
    with pytest.raises(ValueError, match="out of range"):
        run(_cfg(curriculum=curriculum[:1], num_iters=4), ckpt_dir,
            engine_sha="test-sha", env_factory=_tiny_env_factory)
