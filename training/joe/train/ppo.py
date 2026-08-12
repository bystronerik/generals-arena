"""PPO algorithm: GAE, clipped surrogate update, and the training loop.

Ported from AverageJoe ``train/ppo.py`` with the plan's resolutions:

- D2: plain entropy bonus. The released code unconditionally replaces it
  with reverse KL toward an ``expander_magnet`` heuristic; the magnet is not
  ported at all.
- D3: sample filtering keeps the top ``adv_top_frac`` by |advantage|.
- D4: win-rate-gated curriculum stages. Each stage constructs a NEW env
  from explicit competition kwargs (see ``training/joe/env.py``) instead of
  mutating distance attributes — the fork's pool generator is jit-cached
  with the env static, so mutation would reuse stale kernels.
- Checkpoints carry learner + optimizer + EMA + global step + engine SHA;
  ``on_checkpoint`` lets the Modal entry commit the Volume after each save.
  Full saves refresh ``state.json`` (schema v2, ``training/joe/state.py``)
  so a killed run resumes at the same global step and curriculum stage.
- No gamma annealing and no reference-Elo eval (see evaluations.py).
"""

import os
import time
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from training.joe.config import CurriculumStage
from training.joe.env import make_competition_env, preset_min_generals_distance
from training.joe.state import check_curriculum_stage, write_state
from training.joe.train.evaluations import periodic_eval
from training.joe.train.rollout_selfplay import collect_rollout


# ---- GAE ----


@partial(jax.jit, static_argnames=["gamma", "gae_lambda"])
def compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    _, N = rewards.shape

    def scan_fn(last_adv, inputs):
        reward, value, next_value, terminated, truncated = inputs
        done = (terminated | truncated).astype(jnp.float32)

        bootstrap = jnp.where(terminated, 0.0, next_value)
        delta = reward + gamma * bootstrap - value

        nonterminal = 1.0 - done
        adv = delta + gamma * gae_lambda * nonterminal * last_adv
        # Zero carry on truncation: delta is wrong (bootstraps from reset state)
        # and must not leak backwards through the GAE chain
        carry = jnp.where(truncated, 0.0, adv)
        return carry, adv

    inputs = (rewards[::-1], values[::-1], next_values[::-1],
              terminated[::-1], truncated[::-1])
    _, advs_rev = jax.lax.scan(scan_fn, jnp.zeros(N), inputs)
    return advs_rev[::-1]


def compute_mc_returns(rewards, terminated, truncated, gamma):
    """True discounted MC returns (no bootstrapping) plus a validity mask:
    1.0 for timesteps whose episode completed within the rollout."""
    _, N = rewards.shape

    def scan_fn(carry, inputs):
        next_ret, next_valid = carry
        rew, term, trunc = inputs
        done = (term | trunc).astype(jnp.float32)
        ret = rew + gamma * next_ret * (1.0 - done)
        valid = jnp.where(done, 1.0, next_valid)
        return (ret, valid), (ret, valid)

    init = (jnp.zeros(N), jnp.zeros(N))
    _, (mc_rets, valid) = jax.lax.scan(
        scan_fn, init, (rewards[::-1], terminated[::-1], truncated[::-1]))
    return mc_rets[::-1], valid[::-1]


# ---- Value loss ----


def make_value_loss_fn(cfg):
    """Return a jittable value loss function based on config.

    CE:  val_aux is (num_bins,) logits, returns HL-Gauss cross-entropy.
    MSE: val_aux is a scalar, returns 0.5 * (val - ret)^2.
    """
    if cfg.value_loss not in ("mse", "ce"):
        raise ValueError(f"Unknown value_loss '{cfg.value_loss}'. Must be 'mse' or 'ce'.")
    if cfg.value_loss == "ce":
        bin_centers = jnp.linspace(cfg.v_min, cfg.v_max, cfg.num_bins)
        half_width = (cfg.v_max - cfg.v_min) / (cfg.num_bins - 1) / 2.0
        sigma = cfg.hl_sigma

        def value_loss_fn(logits, ret):
            upper = (bin_centers + half_width - ret) / sigma
            lower = (bin_centers - half_width - ret) / sigma
            target_probs = jax.scipy.stats.norm.cdf(upper) - jax.scipy.stats.norm.cdf(lower)
            target_probs = target_probs / jnp.maximum(jnp.sum(target_probs), 1e-8)
            log_probs = jax.nn.log_softmax(logits)
            return -jnp.sum(target_probs * log_probs)

        return value_loss_fn

    def value_loss_fn(val, ret):
        return 0.5 * (val - ret) ** 2

    return value_loss_fn


def hl_gauss_targets(cfg, ret):
    """The HL-Gauss target distribution for one return (test/analysis helper)."""
    bin_centers = jnp.linspace(cfg.v_min, cfg.v_max, cfg.num_bins)
    half_width = (cfg.v_max - cfg.v_min) / (cfg.num_bins - 1) / 2.0
    upper = jax.scipy.stats.norm.cdf((bin_centers + half_width - ret) / cfg.hl_sigma)
    lower = jax.scipy.stats.norm.cdf((bin_centers - half_width - ret) / cfg.hl_sigma)
    probs = upper - lower
    return probs / jnp.maximum(jnp.sum(probs), 1e-8)


# ---- PPO update ----


def ppo_update(network, opt_state, batch, optimizer, key, clip_eps, vf_coef,
               ent_coef, minibatch_size, value_loss_fn, sample_idx):
    """One PPO epoch: shuffle sample indices, lax.scan over minibatches."""
    (obs, move_masks, build_masks, temporal, actions, old_lps, advs, rets,
     train_mask) = batch
    total = obs.shape[0] * obs.shape[1]

    # Flatten time and env dims (views — no allocation)
    obs_f = obs.reshape(total, *obs.shape[2:])
    move_f = move_masks.reshape(total, *move_masks.shape[2:])
    build_f = build_masks.reshape(total, *build_masks.shape[2:])
    temporal_f = temporal.reshape(total, *temporal.shape[2:])
    actions_f = actions.reshape(total, -1)
    old_lps_f = old_lps.reshape(-1)
    advs_f = advs.reshape(-1)
    rets_f = rets.reshape(-1)
    mask_f = train_mask.reshape(-1)

    # Shuffle indices only (not data — avoids full-size copies)
    n_samples = sample_idx.shape[0]
    num_batches = n_samples // minibatch_size
    perm = jrandom.permutation(key, n_samples)
    shuffled_idx = sample_idx[perm]
    idx_mb = shuffled_idx.reshape(num_batches, minibatch_size)

    def scan_body(carry, mb_idx):
        network, opt_state = carry
        mb_obs = obs_f[mb_idx]
        mb_move = move_f[mb_idx]
        mb_build = build_f[mb_idx]
        mb_temporal = temporal_f[mb_idx]
        mb_actions = actions_f[mb_idx]
        mb_old_lps = old_lps_f[mb_idx]
        mb_advs = advs_f[mb_idx]
        mb_rets = rets_f[mb_idx]
        mb_mask = mask_f[mb_idx]

        def loss_fn(net):
            def single_loss(o, mm, bm, td, a, old_lp, adv, ret):
                _, val, lp, ent, val_aux, _ = net(o, mm, bm, td, None, a)
                log_ratio = lp - old_lp
                ratio = jnp.exp(log_ratio)
                pg1 = -adv * ratio
                pg2 = -adv * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                policy_loss = jnp.maximum(pg1, pg2)
                value_loss = value_loss_fn(val_aux, ret)
                clipped = (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)
                approx_kl = ratio - 1.0 - log_ratio
                # D2: plain entropy bonus, no magnet
                total = policy_loss + vf_coef * value_loss - ent_coef * ent
                return total, {
                    "policy_loss": policy_loss, "value_loss": value_loss,
                    "entropy": ent, "clip_fraction": clipped,
                    "approx_kl": approx_kl, "ratio": ratio,
                    "log_ratio": log_ratio, "lp": lp, "old_lp": old_lp,
                }

            all_losses, s = jax.vmap(single_loss)(
                mb_obs, mb_move, mb_build, mb_temporal, mb_actions,
                mb_old_lps, mb_advs, mb_rets)
            # Mask out truncated steps (their delta/advantage is wrong)
            masked_losses = all_losses * mb_mask
            mean_loss = masked_losses.sum() / jnp.maximum(mb_mask.sum(), 1.0)
            # Per-minibatch summary; key name carries the cross-minibatch
            # reduction (max_*/min_* -> max/min, else mean).
            stats = {
                "total_loss": mean_loss,
                "policy_loss": s["policy_loss"].mean(),
                "value_loss": s["value_loss"].mean(),
                "entropy": s["entropy"].mean(),
                "clip_fraction": s["clip_fraction"].mean(),
                "approx_kl": s["approx_kl"].mean(),
                "mean_ratio": s["ratio"].mean(),
                "max_kl": s["approx_kl"].max(),
                "max_ratio": s["ratio"].max(),
                "min_log_ratio": s["log_ratio"].min(),
                "max_log_ratio": s["log_ratio"].max(),
                "min_lp": s["lp"].min(),
                "max_lp": s["lp"].max(),
                "min_old_lp": s["old_lp"].min(),
                "max_old_lp": s["old_lp"].max(),
            }
            return mean_loss, stats

        (loss, stats), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
        grads = jax.lax.pmean(grads, axis_name="devices")

        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        stats["grad_norm"] = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves))
        policy_grads = jax.tree.leaves(eqx.filter(grads.policy_head, eqx.is_array))
        stats["actor_grad_norm"] = jnp.sqrt(sum(jnp.sum(g**2) for g in policy_grads))
        value_grads = jax.tree.leaves(eqx.filter(grads.value_head, eqx.is_array))
        stats["critic_grad_norm"] = jnp.sqrt(sum(jnp.sum(g**2) for g in value_grads))

        updates, opt_state = optimizer.update(grads, opt_state, network)
        network = eqx.apply_updates(network, updates)
        return (network, opt_state), stats

    (network, opt_state), stacked = jax.lax.scan(
        scan_body, (network, opt_state), idx_mb)

    def _reduce(k, v):
        if k.startswith("max_"):
            return jnp.max(v)
        if k.startswith("min_"):
            return jnp.min(v)
        return jnp.mean(v)

    result = {k: _reduce(k, v) for k, v in stacked.items()}
    return network, opt_state, result


# ---- Checkpointing ----


def save_checkpoint(ckpt_dir, run_name, it, network, opt_state, ema_network,
                    engine_sha, full, curriculum_stage=0, last_eval_wr=0.0):
    """Write checkpoint files and, on full saves, the resume state.

    full=True writes (network, opt_state) + EMA + latest-EMA and refreshes
    ``state.json``; full=False writes the EMA snapshot only (the cheap
    ckpt_every cadence). ``it`` is the global step, so a resumed run never
    rewrites a step-named file an earlier process wrote.
    """
    ema_path = os.path.join(ckpt_dir, f"{run_name}_ema_{it}.eqx")
    eqx.tree_serialise_leaves(ema_path, ema_network)
    written = [ema_path]
    if full:
        path = os.path.join(ckpt_dir, f"{run_name}_{it}.eqx")
        eqx.tree_serialise_leaves(path, (network, opt_state))
        ema_latest = os.path.join(ckpt_dir, f"{run_name}_ema.eqx")
        eqx.tree_serialise_leaves(ema_latest, ema_network)
        written = [path] + written + [ema_latest]
        write_state(ckpt_dir, run_name, it, curriculum_stage, last_eval_wr,
                    engine_sha,
                    files={"full": os.path.basename(path),
                           "ema": os.path.basename(ema_path)})
    return written


# ---- Training loop ----


def _replicate(tree, num_devices):
    """Add a leading device axis for pmap (jax >= 0.11 dropped
    ``jax.device_put_replicated``; pmap places the shards itself)."""
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + jnp.shape(x)), tree)


def train(cfg, network, optimizer, opt_state, logger, key, bundle,
          ckpt_dir, engine_sha, on_checkpoint=None, env_factory=None,
          start_step=0, start_stage=0, start_eval_wr=0.0):
    """Main PPO training loop with multi-GPU data parallelism via pmap.

    ``env_factory(min_generals_distance, max_generals_distance, pool_size)``
    defaults to the competition preset; tests inject a tiny env instead.

    ``start_step`` / ``start_stage`` / ``start_eval_wr`` restore a resumed
    run: the loop runs the global steps ``start_step..num_iters``, so
    ``num_iters`` is the run's total target, and checkpoint names,
    schedules, and cadences all use the global step.
    """
    if env_factory is None:
        env_factory = make_competition_env
    num_envs = cfg.num_envs
    num_devices = jax.device_count()
    run_name = cfg.run_name

    init_obs_state_fn = bundle["init_obs_state"]
    augment_fn = bundle["augment_obs"]
    greedy_fn = bundle["greedy_action"]
    value_loss_fn = make_value_loss_fn(cfg)

    # Curriculum stages; without one, a single stage at the preset window.
    stages = cfg.curriculum_stages or [CurriculumStage(
        min_generals_distance=preset_min_generals_distance())]
    check_curriculum_stage(start_stage, len(stages))
    current_stage_idx = start_stage
    last_eval_wr = start_eval_wr

    def stage_envs(stage, pool_key):
        """Fresh train + eval envs and pools for one curriculum stage."""
        env = env_factory(
            stage.min_generals_distance, stage.max_generals_distance,
            cfg.pool_size)
        eval_env = env_factory(
            stage.min_generals_distance, stage.max_generals_distance,
            min(cfg.pool_size, 1024))
        k1, k2 = jrandom.split(pool_key)
        pool, _ = env.reset(k1)
        eval_pool, _ = eval_env.reset(k2)
        return env, eval_env, pool, eval_pool

    key, pool_key = jrandom.split(key)
    stage = stages[current_stage_idx]
    print(f"Curriculum: {len(stages)} stages")
    for i, s in enumerate(stages):
        gate = f"wr>={s.win_rate_threshold:.0%}" if i > 0 else "start"
        print(f"  stage {i}: {gate} -> dist={s.min_generals_distance}-"
              f"{s.max_generals_distance}")
    if start_step > 0:
        print(f"Resuming at global step {start_step}, stage "
              f"{current_stage_idx} (last eval wr {last_eval_wr:.0%})",
              flush=True)
    t0 = time.time()
    env, eval_env, pool, eval_pool = stage_envs(stage, pool_key)
    jax.block_until_ready(pool.armies)
    print(f"Pool generated in {time.time() - t0:.1f}s "
          f"(size {env.pool_size}, stage {current_stage_idx})", flush=True)

    # Partition network for pmap: array leaves replicated, static in closures
    params, static = eqx.partition(network, eqx.is_array)
    params = _replicate(params, num_devices)
    opt_state = _replicate(opt_state, num_devices)

    def make_p_init_envs(env):
        def _init_envs(key):
            keys = jrandom.split(key, num_envs)
            return jax.vmap(env.init_state)(keys)
        return jax.pmap(_init_envs)

    def make_p_rollout(env):
        def _rollout(params, states, key, osp0, osp1, pool_r):
            net = eqx.combine(params, static)
            return collect_rollout(
                states, env, net, key, cfg.num_steps, osp0, osp1,
                augment_fn, pool_r)
        return jax.pmap(_rollout)

    p_init_envs = make_p_init_envs(env)
    p_rollout = make_p_rollout(env)

    key, init_key = jrandom.split(key)
    states = p_init_envs(jrandom.split(init_key, num_devices))

    # Per-network observation state (batched across envs, replicated across devices)
    single_state = init_obs_state_fn(cfg.pad_to, cfg.pad_to)
    batched_obs_state = jax.tree.map(
        lambda x: jnp.tile(x, (num_envs, *([1] * x.ndim))), single_state)
    obs_state_p0 = _replicate(batched_obs_state, num_devices)
    obs_state_p1 = _replicate(batched_obs_state, num_devices)

    keys = jrandom.split(key, num_devices)

    pool_rep = _replicate(pool, num_devices)

    def _compute_gae(rews, vals, next_vals, terminated, truncated):
        return compute_gae(rews, vals, next_vals, terminated, truncated,
                           cfg.gamma, cfg.gae_lambda)
    p_gae = jax.pmap(_compute_gae)
    p_mc_returns = jax.pmap(lambda r, te, tr: compute_mc_returns(r, te, tr, cfg.gamma))

    @partial(jax.pmap, axis_name="devices")
    def _normalize_advs(advs):
        mean = jax.lax.pmean(advs.mean(), axis_name="devices")
        mean_sq = jax.lax.pmean((advs ** 2).mean(), axis_name="devices")
        std = jnp.sqrt(jnp.maximum(mean_sq - mean ** 2, 0.0))
        return (advs - mean) / (std + 1e-8)

    def _ppo_step(params, opt_state, batch, key, ent_coef, sample_idx):
        net = eqx.combine(params, static)
        net, opt_state, result = ppo_update(
            net, opt_state, batch, optimizer, key, cfg.clip_eps, cfg.vf_coef,
            ent_coef, cfg.minibatch_size, value_loss_fn, sample_idx)
        new_params, _ = eqx.partition(net, eqx.is_array)
        result = jax.lax.pmean(result, axis_name="devices")
        return new_params, opt_state, result
    p_ppo_step = jax.pmap(_ppo_step, axis_name="devices")

    p_split_key = jax.pmap(lambda k: jrandom.split(k))

    def _get_network():
        return eqx.combine(jax.tree.map(lambda x: x[0], params), static)

    def _get_opt_state():
        return jax.tree.map(lambda x: x[0], opt_state)

    # Sample filtering: top-k by |advantage| (D3)
    per_device_total = cfg.num_steps * 2 * num_envs
    n_keep = int(per_device_total * cfg.adv_top_frac)
    n_keep = (n_keep // cfg.minibatch_size) * cfg.minibatch_size

    @jax.pmap
    def _compute_top_idx(advs):
        flat_advs = advs.reshape(-1)
        _, top_idx = jax.lax.top_k(jnp.abs(flat_advs), n_keep)
        return top_idx

    # EMA network (updated every iteration; the deployment policy)
    ema_decay = cfg.ema_decay
    if cfg.ema_checkpoint:
        ema_network = eqx.tree_deserialise_leaves(cfg.ema_checkpoint, _get_network())
        ema_params, _ = eqx.partition(ema_network, eqx.is_array)
        print(f"Loaded EMA weights from {cfg.ema_checkpoint}")
    else:
        ema_params = jax.tree.map(lambda x: x[0].copy(), params)

    print(f"Training (self-play, {num_devices} device(s))...", flush=True)
    train_start = time.time()
    for it in range(start_step, cfg.num_iters):
        # Eval before training so it==0 gives a baseline
        network = _get_network()
        on_last_stage = current_stage_idx >= len(stages) - 1
        eval_freq = cfg.eval_every_after if (cfg.eval_every_after and on_last_stage) \
            else cfg.eval_every
        eval_ran, last_eval_wr, key = periodic_eval(
            it, cfg, eval_freq, network, eval_env, eval_pool, single_state,
            augment_fn, greedy_fn, logger, key, last_eval_wr)

        t0 = time.time()

        # Curriculum: advance when the win-rate threshold is met (eval iters only)
        if eval_ran and current_stage_idx < len(stages) - 1:
            next_stage = stages[current_stage_idx + 1]
            if last_eval_wr >= next_stage.win_rate_threshold:
                current_stage_idx += 1
                stage = next_stage
                print(f"CURRICULUM stage {current_stage_idx}/{len(stages) - 1} "
                      f"(iter {it}, wr={last_eval_wr:.0%}>="
                      f"{stage.win_rate_threshold:.0%}): "
                      f"dist={stage.min_generals_distance}-"
                      f"{stage.max_generals_distance}", flush=True)
                # New env objects: the pool generator's jit cache keys on the
                # env, so a fresh env is required for a fresh distance window.
                key, pool_key = jrandom.split(key)
                env, eval_env, pool, eval_pool = stage_envs(stage, pool_key)
                pool_rep = _replicate(pool, num_devices)
                p_init_envs = make_p_init_envs(env)
                p_rollout = make_p_rollout(env)
                key, reinit_key = jrandom.split(key)
                states = p_init_envs(jrandom.split(reinit_key, num_devices))
                obs_state_p0 = _replicate(batched_obs_state, num_devices)
                obs_state_p1 = _replicate(batched_obs_state, num_devices)
                print("CURRICULUM: regenerated pool", flush=True)

        # Periodic pool refresh for map diversity (pool is traced — no recompile)
        if cfg.reset_pool_every > 0 and it > 0 and it % cfg.reset_pool_every == 0:
            key, pool_key = jrandom.split(key)
            pool, _ = env.reset(pool_key)
            pool_rep = _replicate(pool, num_devices)

        # Collect rollout — pmapped across devices
        t_rollout = time.time()
        states, rollout_data, keys, obs_state_p0, obs_state_p1 = p_rollout(
            params, states, keys, obs_state_p0, obs_state_p1, pool_rep)
        jax.block_until_ready(states)
        t_rollout = time.time() - t_rollout

        # Shapes: (D, num_steps, 2*num_envs, ...)
        (obs, move_masks, build_masks, temporal, actions, lps, vals,
         next_vals, rews, terminated, truncated, winners,
         owned_castles) = rollout_data
        del rollout_data
        dones = terminated | truncated

        # Episode stats from the p0 seat (summed across devices)
        dones_p0 = dones[:, :, :num_envs]
        terminated_p0 = terminated[:, :, :num_envs]
        winners_p0 = winners[:, :, :num_envs]

        advs = p_gae(rews, vals, next_vals, terminated, truncated)
        rets = advs + vals
        adv_std_raw = float(advs.std())
        advs = _normalize_advs(advs)

        # Rollout-level diagnostics (before filtering)
        mean_owned_castles = float(owned_castles.mean()) / num_envs
        mean_val = float(vals.mean())
        ret_mean = float(rets.mean())
        ret_std = float(rets.std())
        var_returns = float(jnp.var(rets))
        explained_var = 1.0 - float(jnp.var(rets - vals)) / max(var_returns, 1e-8)
        value_bias = float(jnp.mean(vals - rets))

        mc_rets, mc_valid = p_mc_returns(rews, terminated, truncated)
        mc_valid_count = float(mc_valid.sum())
        mc_valid_frac = mc_valid_count / max(float(mc_valid.size), 1.0)
        if mc_valid_count > 100:
            mc_diff = (vals - mc_rets) * mc_valid
            mc_mean_ret = float(jnp.sum(mc_rets * mc_valid)) / mc_valid_count
            mc_var_ret = float(jnp.sum(mc_valid * (mc_rets - mc_mean_ret) ** 2)
                               / mc_valid_count)
            mc_ev = 1.0 - float(jnp.sum(mc_diff ** 2) / mc_valid_count) / max(mc_var_ret, 1e-8)
            mc_value_bias = float(jnp.sum(mc_diff) / mc_valid_count)
        else:
            mc_ev = float("nan")
            mc_value_bias = float("nan")

        # PPO update
        t_ppo = time.time()
        train_mask = 1.0 - truncated.astype(jnp.float32)

        sample_idx = _compute_top_idx(advs)
        advs_flat = advs.reshape(num_devices, -1)
        filtered_advs = jnp.take_along_axis(advs_flat, sample_idx, axis=1)
        filtered_adv_std = float(filtered_advs.std())
        filtered_adv_mean = float(jnp.abs(filtered_advs).mean())

        batch = (obs, move_masks, build_masks, temporal, actions, lps,
                 advs, rets, train_mask)
        metrics = {}
        epochs_used = 0

        # Entropy coefficient schedule (global step)
        if cfg.ent_schedule == "power_law":
            current_ent_coef = max(
                cfg.ent_coef_start / (it + 1) ** cfg.ent_power,
                cfg.ent_coef_min)
        else:
            t = min(it / max(cfg.ent_coef_decay_iters, 1), 1.0)
            current_ent_coef = cfg.ent_coef_start + t * (cfg.ent_coef_end - cfg.ent_coef_start)

        # Learning rate (for logging; the optax schedule applies it)
        if cfg.lr_schedule == "power_law":
            iteration = it + 1.0
            raw = cfg.lr_power_law_numerator / (iteration ** cfg.lr_power_law_exponent)
            current_lr = max(min(raw, cfg.lr_power_law_max), cfg.lr_power_law_min)
        elif cfg.lr_decay_iters > 0:
            t_lr = min(it / cfg.lr_decay_iters, 1.0)
            current_lr = cfg.lr + t_lr * (cfg.final_lr - cfg.lr)
        else:
            current_lr = cfg.lr
        ent_coef_arr = jnp.full(num_devices, current_ent_coef)

        for _ in range(cfg.num_epochs):
            split = p_split_key(keys)
            keys, epoch_keys = split[:, 0], split[:, 1]
            params, opt_state, metrics = p_ppo_step(
                params, opt_state, batch, epoch_keys, ent_coef_arr, sample_idx)
            epochs_used += 1
            if cfg.target_kl is not None and float(metrics["approx_kl"][0]) > cfg.target_kl:
                break
        jax.block_until_ready(params)
        t_ppo = time.time() - t_ppo

        m = jax.tree.map(lambda x: x[0], metrics)

        elapsed = time.time() - t0
        eps = int(dones_p0.sum())
        wins = int(jnp.sum(terminated_p0 & (winners_p0 == 0)))
        losses = int(jnp.sum(terminated_p0 & (winners_p0 == 1)))
        draws = eps - wins - losses
        wr = wins / max(eps, 1)
        lr_rate = losses / max(eps, 1)
        dr = draws / max(eps, 1)
        samples_per_iter = num_devices * 2 * num_envs * cfg.num_steps
        sps = samples_per_iter / elapsed
        mean_ep_len = float(dones_p0.size) / max(eps, 1)

        wall = int(time.time() - train_start)
        hh, mm, ss = wall // 3600, wall % 3600 // 60, wall % 60
        print(
            f"[{hh:02d}:{mm:02d}:{ss:02d}] Iter {it + 1:3d}/{cfg.num_iters} | "
            f"Loss: {float(m['total_loss']):.4f} | "
            f"PG: {float(m['policy_loss']):.4f} | "
            f"VF: {float(m['value_loss']):.4f} | "
            f"Ent: {float(m['entropy']):.3f} | "
            f"KL: {float(m['approx_kl']):.4f} | "
            f"Clip: {float(m['clip_fraction']):.2f} | "
            f"GNorm: {float(m['grad_norm']):.2f} | "
            f"EV: {explained_var:.2f} | "
            f"Reward: {float(rews.mean()):+.4f} | "
            f"Eps: {eps:3d} | W/L/D: {wins}/{losses}/{draws} "
            f"({wr * 100:.0f}%/{lr_rate * 100:.0f}%/{dr * 100:.0f}%) | "
            f"EpLen: {mean_ep_len:.0f} | "
            f"Castles: {mean_owned_castles:.2f} | "
            f"Epochs: {epochs_used}/{cfg.num_epochs} | "
            f"LR: {current_lr:.1e} | "
            f"SPS: {sps:.0f} | {elapsed:.2f}s "
            f"(rollout {t_rollout:.2f}s, ppo {t_ppo:.2f}s)",
            flush=True,
        )

        log_metrics = {
            "train/total_loss": m["total_loss"],
            "train/policy_loss": m["policy_loss"],
            "train/value_loss": m["value_loss"],
            "train/entropy": m["entropy"],
            "train/n_actions": jnp.exp(m["entropy"]),
            "train/clip_fraction": m["clip_fraction"],
            "train/approx_kl": m["approx_kl"],
            "train/mean_ratio": m["mean_ratio"],
            "train/max_ratio": m["max_ratio"],
            "train/grad_norm": m["grad_norm"],
            "train/actor_grad_norm": m["actor_grad_norm"],
            "train/critic_grad_norm": m["critic_grad_norm"],
            "train/explained_variance": explained_var,
            "train/mc_explained_variance": mc_ev,
            "train/value_bias": value_bias,
            "train/mc_value_bias": mc_value_bias,
            "train/mc_valid_frac": mc_valid_frac,
            "train/mean_value": mean_val,
            "train/return_mean": ret_mean,
            "train/return_std": ret_std,
            "train/adv_std_raw": adv_std_raw,
            "train/filtered_adv_std": filtered_adv_std,
            "train/filtered_adv_mean_abs": filtered_adv_mean,
            "train/mean_ep_length": mean_ep_len,
            "train/mean_reward": float(rews.mean()),
            "train/win_rate": wr,
            "train/loss_rate": lr_rate,
            "train/draw_rate": dr,
            "train/mean_owned_castles": mean_owned_castles,
            "train/sps": sps,
            "train/ent_coef": current_ent_coef,
            "train/lr": current_lr,
            "train/curriculum_stage": current_stage_idx,
        }
        if cfg.debug:
            log_metrics.update({
                "train/max_kl": m["max_kl"],
                "train/min_log_ratio": m["min_log_ratio"],
                "train/max_log_ratio": m["max_log_ratio"],
                "train/min_lp": m["min_lp"],
                "train/max_lp": m["max_lp"],
                "train/min_old_lp": m["min_old_lp"],
                "train/max_old_lp": m["max_old_lp"],
                "train/epochs_used": epochs_used,
            })
        logger.log(it + 1, log_metrics)

        # Update EMA params
        current_params = jax.tree.map(lambda x: x[0], params)
        ema_params = jax.tree.map(
            lambda e, c: ema_decay * e + (1 - ema_decay) * c,
            ema_params, current_params)

        network = _get_network()
        save_full = (it + 1) % cfg.save_every == 0
        save_ema = (it + 1) % cfg.ckpt_every == 0
        if save_full or save_ema:
            ema_network = eqx.combine(ema_params, static)
            written = save_checkpoint(
                ckpt_dir, run_name, it + 1, network, _get_opt_state(),
                ema_network, engine_sha, full=save_full,
                curriculum_stage=current_stage_idx, last_eval_wr=last_eval_wr)
            print(f"  SAVED: {', '.join(written)}", flush=True)
            if on_checkpoint is not None:
                on_checkpoint()

        # Free large arrays to limit allocator fragmentation on the next rollout
        del obs, move_masks, build_masks, temporal, actions, lps, advs, rets
        del train_mask, batch, sample_idx, vals, next_vals, rews
        del terminated, truncated, winners, owned_castles
        del dones, dones_p0, terminated_p0, winners_p0

    return _get_network(), _get_opt_state(), eqx.combine(ema_params, static)
