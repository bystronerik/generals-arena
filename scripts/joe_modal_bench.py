#!/usr/bin/env python3
"""Phase 1 throughput benchmark for the joe pipeline (Modal, GPU).

Measures, per GPU type (A10G and H100), in competition mode (build_castles +
deathtouch, pad 21):

  1. Raw env steps/s under vmap + lax.scan with random valid actions,
     at 1k / 4k / 16k parallel envs.
  2. Pool-generation wall time and device memory at pool_size=200k, for the
     mode preset (distance 17+) and for the early curriculum window (2-6).
  3. Net-in-the-loop samples/s with randomly initialized S / M / L nets:
     self-play rollout (augmented 39-ch obs, 10-action head, build mask)
     plus one PPO epoch (GAE, |adv| top-25% filter, HL-Gauss value CE).

Usage (never pipe through tail/head — redirect to a file, AGENTS.md):

    modal run scripts/joe_modal_bench.py > /tmp/joe_bench.log 2>&1 &
    modal app list          # then check startup a few minutes in
    modal app logs <app-id>

Results land in docs/research/measurements/joe-phase1-throughput.md.

All jax/equinox code lives inside the remote functions: Modal re-imports this
file in the container, and the local process must not need GPU deps.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "jax[cuda12]==0.11.0",
        "equinox",
        "optax",
    )
    .add_local_dir(
        str(REPO / "competition-module"),
        remote_path="/root/competition-module",
        copy=True,
    )
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root"})
)

app = modal.App("joe-bench")

# Board/tier constants mirror training/joe/configs/{S,M}.yaml (Phase 0) and
# plan section 2. L exists here only as the paper-scale reference point.
TIERS = {
    "S": dict(depth=4, embed_dim=352, n_head=8, ff_factor=2),
    "M": dict(depth=5, embed_dim=384, n_head=8, ff_factor=3),
    "L": dict(depth=7, embed_dim=448, n_head=8, ff_factor=3),
}


def _bench_impl(gpu_label: str, env_counts, pool_size_big: int,
                net_num_envs: int, net_num_steps: int,
                tiers: str = "SML") -> dict:
    """Runs in the container. All heavy imports and jax code live here."""
    import time
    from typing import NamedTuple

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import optax

    from generals.core.action import compute_valid_move_mask
    from generals.core.env import GeneralsEnv
    from generals.core.game import get_observation
    from generals.modifiers.build_castles import build_cost_grid

    P = 21  # competition pad_to
    dev = jax.devices()[0]
    results = {"gpu": gpu_label, "device": str(dev), "jax": jax.__version__}

    def log(msg):
        print(f"[bench:{gpu_label}] {msg}", flush=True)

    def mem_stats():
        s = dev.memory_stats()
        if not s:
            return {}
        return {
            "bytes_in_use_gb": round(s.get("bytes_in_use", 0) / 2**30, 3),
            "peak_bytes_in_use_gb": round(s.get("peak_bytes_in_use", 0) / 2**30, 3),
        }

    def comp_env(pool_size, **overrides):
        """Competition preset via explicit kwargs so distance can vary.

        Mirrors _MODE_PRESETS["competition"]; the curriculum will construct
        envs the same way (plan section 4).
        """
        kwargs = dict(
            min_grid_size=18, max_grid_size=21, pad_to=P,
            truncation=1200, perfect_info=False,
            mountain_density_range=(0.24, 0.26),
            num_castles_range=(9, 11),
            min_generals_distance=17, max_generals_distance=None,
            castle_val_range=(20, 26),
            build_castles=True, deathtouch_turn=800,
            pool_size=pool_size,
        )
        kwargs.update(overrides)
        return GeneralsEnv(**kwargs)

    log(f"device={dev} jax={jax.__version__}")

    # ------------------------------------------------------------------
    # 1. Raw env steps/s, random valid actions
    # ------------------------------------------------------------------

    def random_valid_action(key, obs):
        """Uniform over valid moves (categorical on the masked flat grid)."""
        mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
        logits = jnp.where(mask.reshape(-1), 0.0, -1e9)
        k1, k2 = jrandom.split(key)
        idx = jrandom.categorical(k1, logits)
        r = idx // (P * 4)
        c = (idx % (P * 4)) // 4
        d = idx % 4
        no_valid = ~jnp.any(mask)
        ih = jrandom.randint(k2, (), 0, 2)
        return jnp.array([no_valid, r, c, d, ih], dtype=jnp.int32)

    raw_rows = []
    for n_envs in env_counts:
        env = comp_env(pool_size=n_envs)
        key = jrandom.PRNGKey(0)
        t0 = time.time()
        pool, _ = env.reset(key)
        jax.block_until_ready(pool.armies)
        small_pool_s = time.time() - t0
        states = pool  # one state per env

        scan_len = 50

        def one_chunk(states, key, env=env, pool=pool):
            def body(carry, _):
                states, key = carry
                key, k = jrandom.split(key)
                obs0 = jax.vmap(lambda s: get_observation(s, 0))(states)
                obs1 = jax.vmap(lambda s: get_observation(s, 1))(states)
                n = states.armies.shape[0]
                keys = jrandom.split(k, 2 * n).reshape(2, n, 2)
                a0 = jax.vmap(random_valid_action)(keys[0], obs0)
                a1 = jax.vmap(random_valid_action)(keys[1], obs1)
                actions = jnp.stack([a0, a1], axis=1)
                ts, states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)
                return (states, key), ts.terminated.sum()
            (states, key), terms = jax.lax.scan(body, (states, key), None, length=scan_len)
            return states, key, terms.sum()

        chunk = jax.jit(one_chunk)
        key = jrandom.PRNGKey(1)
        t0 = time.time()
        states, key, _ = chunk(states, key)
        jax.block_until_ready(states.armies)
        compile_s = time.time() - t0

        t0 = time.time()
        reps = 0
        while time.time() - t0 < 5.0 and reps < 40:
            states, key, _ = chunk(states, key)
            reps += 1
        jax.block_until_ready(states.armies)
        elapsed = time.time() - t0
        sps = n_envs * scan_len * reps / elapsed
        raw_rows.append({
            "num_envs": n_envs, "env_steps_per_s": round(sps),
            "compile_s": round(compile_s, 1),
            "pool_gen_s_at_num_envs": round(small_pool_s, 1),
            **mem_stats(),
        })
        log(f"raw envs={n_envs}: {sps:,.0f} steps/s (compile {compile_s:.1f}s)")
    results["raw_env_steps"] = raw_rows

    # ------------------------------------------------------------------
    # 2. Pool generation at 200k
    # ------------------------------------------------------------------

    pool_rows = []
    for label, overrides in ([
        ("distance-17-preset", {}),
        ("distance-2-6-early", dict(min_generals_distance=2, max_generals_distance=6)),
    ] if pool_size_big > 0 else []):
        env = comp_env(pool_size=pool_size_big, **overrides)
        t0 = time.time()
        pool, _ = env.reset(jrandom.PRNGKey(10))
        jax.block_until_ready(pool.armies)
        cold_s = time.time() - t0
        t0 = time.time()
        pool, _ = env.reset(jrandom.PRNGKey(11))
        jax.block_until_ready(pool.armies)
        warm_s = time.time() - t0
        nbytes = sum(x.nbytes for x in jax.tree.leaves(pool))
        pool_rows.append({
            "window": label, "pool_size": pool_size_big,
            "cold_s": round(cold_s, 1), "warm_s": round(warm_s, 1),
            "pool_gb": round(nbytes / 2**30, 2), **mem_stats(),
        })
        log(f"pool {label}: cold {cold_s:.1f}s warm {warm_s:.1f}s "
            f"({nbytes / 2**30:.2f} GB)")
        del pool
    results["pool_gen"] = pool_rows

    # ------------------------------------------------------------------
    # 3. Net-in-the-loop: rollout + PPO update
    # ------------------------------------------------------------------
    # Shapes match the Phase 2 port: 39 obs channels (38 + build cost),
    # 10-action head (4 full, 4 half, pass=8, build=9), HL-Gauss value CE,
    # |adv| top-25% filter, minibatch 1024, 1 epoch. Random init.

    N_CH = 39
    HIST = 7
    TWIN = 512
    NUM_BINS = 128

    class ObsState(NamedTuple):
        army_stack: jnp.ndarray
        enemy_stack: jnp.ndarray
        last_army: jnp.ndarray
        last_enemy_army: jnp.ndarray
        castles: jnp.ndarray
        generals: jnp.ndarray
        mountains: jnp.ndarray
        seen: jnp.ndarray
        enemy_seen: jnp.ndarray
        last_enemy_seen_value: jnp.ndarray
        last_enemy_seen_age: jnp.ndarray
        army_hist: jnp.ndarray
        land_hist: jnp.ndarray

    def init_obs_state():
        z = lambda *s: jnp.zeros(s)
        zb = lambda *s: jnp.zeros(s, dtype=jnp.bool_)
        return ObsState(
            z(HIST, P, P), z(HIST, P, P), z(P, P), z(P, P),
            zb(P, P), zb(P, P), zb(P, P), zb(P, P), zb(P, P),
            z(P, P), z(P, P), z(TWIN), z(TWIN),
        )

    def _pool2d(x):
        return jax.lax.reduce_window(
            x[None], -jnp.inf, jax.lax.max, (1, 3, 3), (1, 1, 1), "SAME")[0]

    def obs_to_array(o):
        shape = o.armies.shape
        return jnp.stack([
            o.armies, o.generals, o.castles, o.mountains,
            o.neutral_cells, o.owned_cells, o.opponent_cells,
            o.fog_cells, o.structures_in_fog,
            jnp.broadcast_to(o.owned_land_count, shape),
            jnp.broadcast_to(o.owned_army_count, shape),
            jnp.broadcast_to(o.opponent_land_count, shape),
            jnp.broadcast_to(o.opponent_army_count, shape),
            jnp.broadcast_to(o.timestep, shape),
        ], axis=0).astype(jnp.float32)

    def augment(obs, cost, st):
        """(14,P,P) raw obs + build cost + memory -> (39,P,P), new state.

        Same channel recipe as AverageJoe augment_obs (boards arrive already
        mountain-padded to P, so the pad block is unnecessary), plus the
        build-cost channel (decision 4).
        """
        army = obs[0] * obs[5]
        earmy = obs[0] * obs[6]
        astack = jnp.concatenate([(army - st.last_army)[None], st.army_stack[:-1]])
        estack = jnp.concatenate([(earmy - st.last_enemy_army)[None], st.enemy_stack[:-1]])
        seen = st.seen | (_pool2d(obs[5]) > 0)
        eseen = st.enemy_seen | (_pool2d(obs[6]) > 0)
        castles = st.castles | (obs[2] > 0)
        generals = st.generals | (obs[1] > 0)
        mountains = st.mountains | (obs[3] > 0)
        lesv = jnp.where(earmy > 0, earmy, st.last_enemy_seen_value)
        lesa = jnp.where(earmy > 0, 0.0, st.last_enemy_seen_age + 1.0)
        ah = jnp.roll(st.army_hist, -1).at[-1].set(obs[12, 0, 0])
        lh = jnp.roll(st.land_hist, -1).at[-1].set(obs[11, 0, 0])
        ones = jnp.ones((P, P))
        cx = jnp.broadcast_to(jnp.arange(P, dtype=jnp.float32)[None] / (P - 1), (P, P))
        cy = jnp.broadcast_to(jnp.arange(P, dtype=jnp.float32)[:, None] / (P - 1), (P, P))
        chans = jnp.stack([
            obs[0], army, earmy, obs[0] * obs[4],
            seen.astype(jnp.float32), eseen.astype(jnp.float32),
            generals.astype(jnp.float32), castles.astype(jnp.float32),
            mountains.astype(jnp.float32),
            obs[4], obs[5], obs[6], obs[7], obs[8],
            obs[13] * ones, (obs[13] % 50) * ones / 50,
            obs[9] * ones, obs[10] * ones, obs[11] * ones, obs[12] * ones,
            lesv, jnp.log1p(lesa) / 5.0, cx, cy,
            cost / 50.0,                       # 24: own build cost (39th channel)
        ], axis=0)
        aug = jnp.concatenate([chans, astack, estack], axis=0)
        new = ObsState(astack, estack, army, earmy, castles, generals,
                       mountains, seen, eseen, lesv, lesa, ah, lh)
        return aug, new

    def normalize(obs):
        army_ch = [0, 1, 2, 3, 17, 19, 20] + list(range(25, N_CH))
        obs = obs.at[jnp.array(army_ch)].divide(50.0)
        obs = obs.at[jnp.array([14, 16, 18])].divide(50.0)
        return obs

    def action_penalty(move_mask, build_mask, allow_pass=True):
        """(P,P,4) moves + (P,P) builds -> (10,P,P) additive penalty."""
        m = 1 - jnp.transpose(move_mask, (2, 0, 1)).astype(jnp.float32)
        pass_pen = jnp.full((1, P, P), 0.0 if allow_pass else 1.0)
        build_pen = (1 - build_mask.astype(jnp.float32))[None]
        return jnp.concatenate([m, m, pass_pen, build_pen], axis=0) * -1e9

    def decode10(idx):
        gc = P * P
        d, pos = idx // gc, idx % gc
        r, c = pos // P, pos % P
        is_pass, is_build = d == 8, d == 9
        is_half = (d >= 4) & (d < 8)
        pf = jnp.where(is_build, 2, jnp.where(is_pass, 1, 0))
        ad = jnp.where(is_half, d - 4, jnp.where(d < 4, d, 0))
        return jnp.array([pf, r, c, ad, is_half], dtype=jnp.int32)

    def encode10(a):
        gc = P * P
        d = jnp.where(a[0] == 2, 9,
                      jnp.where(a[0] == 1, 8,
                                jnp.where(a[4] > 0, a[3] + 4, a[3])))
        return d * gc + a[1] * P + a[2]

    def to_bf16(tree):
        return jax.tree.map(
            lambda x: x.astype(jnp.bfloat16)
            if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating) else x,
            tree)

    class MHSA(eqx.Module):
        qkv: eqx.nn.Linear
        out: eqx.nn.Linear
        n_head: int = eqx.field(static=True)

        def __init__(self, d, n_head, *, key):
            k1, k2 = jrandom.split(key)
            self.qkv = eqx.nn.Linear(d, 3 * d, key=k1)
            self.out = eqx.nn.Linear(d, d, key=k2)
            self.n_head = n_head

        def __call__(self, x):
            t, d = x.shape
            hd = d // self.n_head
            qkv = jax.vmap(self.qkv)(x).reshape(t, 3, self.n_head, hd)
            q, k, v = (jnp.transpose(qkv[:, i], (1, 0, 2)) for i in range(3))
            att = q @ jnp.transpose(k, (0, 2, 1)) / (hd ** 0.5)
            att = jax.nn.softmax(att.astype(jnp.float32), axis=-1).astype(q.dtype)
            o = jnp.transpose(att @ v, (1, 0, 2)).reshape(t, d)
            return jax.vmap(self.out)(o)

    class Block(eqx.Module):
        n1: eqx.nn.LayerNorm
        attn: MHSA
        n2: eqx.nn.LayerNorm
        f1: eqx.nn.Linear
        f2: eqx.nn.Linear

        def __init__(self, d, n_head, ff, *, key):
            k1, k2, k3 = jrandom.split(key, 3)
            self.n1 = eqx.nn.LayerNorm(d)
            self.attn = MHSA(d, n_head, key=k1)
            self.n2 = eqx.nn.LayerNorm(d)
            self.f1 = eqx.nn.Linear(d, ff * d, key=k2)
            self.f2 = eqx.nn.Linear(ff * d, d, key=k3)

        def __call__(self, x):
            x = x + self.attn(jax.vmap(self.n1)(x))
            h = jax.nn.silu(jax.vmap(self.f1)(jax.vmap(self.n2)(x)))
            return x + jax.vmap(self.f2)(h)

    class Temporal(eqx.Module):
        a1: eqx.nn.Linear
        a2: eqx.nn.Linear
        l1: eqx.nn.Linear
        l2: eqx.nn.Linear

        def __init__(self, d, *, key):
            ks = jrandom.split(key, 4)
            hid = 512
            self.a1 = eqx.nn.Linear(TWIN, hid, key=ks[0])
            self.a2 = eqx.nn.Linear(hid, d, key=ks[1])
            self.l1 = eqx.nn.Linear(TWIN, hid, key=ks[2])
            self.l2 = eqx.nn.Linear(hid, d, key=ks[3])

        def __call__(self, td):
            a = self.a2(jax.nn.silu(self.a1(td[0] / 50.0)))
            l = self.l2(jax.nn.silu(self.l1(td[1] / 50.0)))
            return jnp.stack([a, l])

    class Net(eqx.Module):
        embed: eqx.nn.Linear
        value_token: jnp.ndarray
        pos: jnp.ndarray
        layers: list
        norm: eqx.nn.LayerNorm
        policy: eqx.nn.Linear
        value: eqx.nn.Linear
        temporal: Temporal
        ttype: jnp.ndarray
        bins: jnp.ndarray
        M: int = eqx.field(static=True)

        def __init__(self, depth, embed_dim, n_head, ff_factor, patch=3, *, key):
            ks = jrandom.split(key, depth + 7)
            self.M = patch
            gp = P // patch
            self.embed = eqx.nn.Linear(N_CH * patch * patch, embed_dim, key=ks[0])
            self.value_token = jrandom.normal(ks[1], (1, embed_dim)) * 0.02
            self.pos = jrandom.truncated_normal(
                ks[2], -2.0, 2.0, (gp * gp + 3, embed_dim)) * 0.1
            self.layers = [Block(embed_dim, n_head, ff_factor, key=ks[3 + i])
                           for i in range(depth)]
            self.norm = eqx.nn.LayerNorm(embed_dim)
            self.policy = eqx.nn.Linear(embed_dim, 10 * patch * patch, key=ks[3 + depth])
            self.value = eqx.nn.Linear(embed_dim, NUM_BINS, key=ks[4 + depth])
            self.temporal = Temporal(embed_dim, key=ks[5 + depth])
            self.ttype = jrandom.normal(ks[6 + depth], (2, embed_dim)) * 0.02
            self.bins = jnp.linspace(-1.0, 1.0, NUM_BINS)

        def _forward(self, obs, move_mask, build_mask, td):
            M = self.M
            gp = P // M
            obs = normalize(obs).astype(jnp.bfloat16)
            pen = action_penalty(move_mask, build_mask)
            net = to_bf16(self)
            x = obs.reshape(N_CH, gp, M, gp, M).transpose(1, 3, 0, 2, 4).reshape(gp * gp, -1)
            x = jax.vmap(net.embed)(x)
            tt = net.temporal(td.astype(jnp.bfloat16)) + net.ttype
            x = jnp.concatenate([net.value_token, tt, x]) + net.pos
            for layer in net.layers:
                x = layer(x)
            x = jax.vmap(net.norm)(x)
            vlogits = net.value(x[0]).astype(jnp.float32)
            value = jnp.sum(jax.nn.softmax(vlogits) * self.bins)
            pl = jax.vmap(net.policy)(x[3:]).astype(jnp.float32)
            logits = pl.reshape(gp, gp, 10, M, M).transpose(2, 0, 3, 1, 4).reshape(10, P, P)
            return (logits + pen).reshape(-1), value, vlogits

        def __call__(self, obs, move_mask, build_mask, td, key, action=None):
            logits, value, vlogits = self._forward(obs, move_mask, build_mask, td)
            if action is None:
                idx = jrandom.categorical(key, logits)
                action = decode10(idx)
            else:
                idx = encode10(action)
            lp = jax.nn.log_softmax(logits)
            p = jax.nn.softmax(logits)
            return action, value, lp[idx], -jnp.sum(p * lp), vlogits

    def hl_gauss_ce(vlogits, ret):
        bins = jnp.linspace(-1.0, 1.0, NUM_BINS)
        hw = 2.0 / (NUM_BINS - 1) / 2.0
        up = jax.scipy.stats.norm.cdf((bins + hw - ret) / 0.04)
        lo = jax.scipy.stats.norm.cdf((bins - hw - ret) / 0.04)
        tp = (up - lo) / jnp.maximum(jnp.sum(up - lo), 1e-8)
        return -jnp.sum(tp * jax.nn.log_softmax(vlogits))

    def build_masks(obs, cost):
        move = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
        build = (obs.owned_cells & ~(obs.generals > 0) & ~(obs.castles > 0)
                 & (obs.armies > cost))
        return move, build

    # pool_size must stay a multiple of the 16 (h, w) combos and >= num_envs
    net_env = comp_env(pool_size=max(4 * net_num_envs, 64))
    net_pool, _ = net_env.reset(jrandom.PRNGKey(20))
    jax.block_until_ready(net_pool.armies)
    n = net_num_envs
    T = net_num_steps
    minibatch = min(1024, max(1, (2 * n * T) // 8))  # 1024 at real sizes
    top_frac = 0.25
    n_keep = int(2 * n * T * top_frac) // minibatch * minibatch

    def make_rollout(static, pool):
        def rollout(params, states, osp, key):
            net = eqx.combine(params, static)
            return _rollout_inner(net, states, osp, key, pool)
        return eqx.filter_jit(rollout)

    def _rollout_inner(net, states, osp, key, pool):
        def body(carry, _):
            states, osp, key = carry
            obs0 = jax.vmap(lambda s: get_observation(s, 0))(states)
            obs1 = jax.vmap(lambda s: get_observation(s, 1))(states)
            cat = lambda a, b: jax.tree.map(lambda x, y: jnp.concatenate([x, y]), a, b)
            obs = cat(obs0, obs1)
            cost0 = jax.vmap(lambda s: build_cost_grid(s, 0))(states)
            cost1 = jax.vmap(lambda s: build_cost_grid(s, 1))(states)
            cost = jnp.concatenate([cost0, cost1]).astype(jnp.float32)
            arr = jax.vmap(obs_to_array)(obs)
            move, build = jax.vmap(build_masks)(obs, cost)
            aug, osp = jax.vmap(augment)(arr, cost, osp)
            aug = aug.astype(jnp.bfloat16)
            td = jnp.stack([osp.army_hist, osp.land_hist], axis=1)
            key, k = jrandom.split(key)
            keys = jrandom.split(k, 2 * n)
            acts, vals, lps, _, _ = jax.vmap(net, in_axes=(0, 0, 0, 0, 0, None))(
                aug, move, build, td, keys, None)
            env_actions = jnp.stack([acts[:n], acts[n:]], axis=1)
            ts, states = jax.vmap(lambda s, a: net_env.step(s, a, pool))(states, env_actions)
            rew = jnp.concatenate([ts.reward[:, 0], ts.reward[:, 1]])
            done = ts.terminated | ts.truncated
            done2 = jnp.concatenate([done, done])
            osp = jax.tree.map(
                lambda x: jnp.where(done2.reshape(-1, *([1] * (x.ndim - 1))),
                                    jnp.zeros_like(x), x), osp)
            trunc2 = jnp.concatenate([ts.truncated, ts.truncated])
            term2 = jnp.concatenate([ts.terminated, ts.terminated])
            return (states, osp, key), (aug, move, build, td, acts, vals, lps,
                                        rew, term2, trunc2)

        (states, osp, key), data = jax.lax.scan(
            body, (states, osp, key), None, length=T)
        return states, osp, key, data

    def gae(rew, vals, term, trunc, lam=0.9):
        next_vals = jnp.concatenate([vals[1:], vals[-1:]])

        def scan_fn(last, inp):
            r, v, nv, te, tr = inp
            boot = jnp.where(te, 0.0, nv)
            delta = r + boot - v
            adv = delta + lam * (1.0 - (te | tr).astype(jnp.float32)) * last
            return jnp.where(tr, 0.0, adv), adv

        _, advs = jax.lax.scan(
            scan_fn, jnp.zeros(rew.shape[1]),
            (rew[::-1], vals[::-1], next_vals[::-1], term[::-1], trunc[::-1]))
        return advs[::-1]

    def make_ppo(static, optimizer):
        def ppo(params, opt_state, data, key):
            return _ppo_inner(params, static, opt_state, optimizer, data, key)
        return eqx.filter_jit(ppo)

    def _ppo_inner(params, static, opt_state, optimizer, data, key):
        aug, move, build, td, acts, vals, lps, rew, term, trunc = data
        advs = gae(rew, vals, term, trunc)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        rets = advs + vals
        train_mask = 1.0 - trunc.astype(jnp.float32)
        total = 2 * n * T
        flat = lambda x: x.reshape(total, *x.shape[2:])
        aug_f, move_f, build_f, td_f = flat(aug), flat(move), flat(build), flat(td)
        acts_f, lps_f = flat(acts), lps.reshape(-1)
        advs_f, rets_f, mask_f = advs.reshape(-1), rets.reshape(-1), train_mask.reshape(-1)
        _, top_idx = jax.lax.top_k(jnp.abs(advs_f), n_keep)
        perm = jrandom.permutation(key, n_keep)
        idx_mb = top_idx[perm].reshape(n_keep // minibatch, minibatch)

        def mb_body(carry, mb_idx):
            params, opt_state = carry

            def loss_fn(p):
                net = eqx.combine(p, static)

                def single(o, m, b, t, a, olp, adv, ret):
                    _, _, lp, ent, vlog = net(o, m, b, t, None, a)
                    ratio = jnp.exp(lp - olp)
                    pg = jnp.maximum(-adv * ratio,
                                     -adv * jnp.clip(ratio, 0.8, 1.2))
                    return pg + 0.5 * hl_gauss_ce(vlog, ret) - 0.05 * ent

                losses = jax.vmap(single)(
                    aug_f[mb_idx], move_f[mb_idx], build_f[mb_idx], td_f[mb_idx],
                    acts_f[mb_idx], lps_f[mb_idx], advs_f[mb_idx], rets_f[mb_idx])
                m = mask_f[mb_idx]
                return (losses * m).sum() / jnp.maximum(m.sum(), 1.0)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(mb_body, (params, opt_state), idx_mb)
        return params, opt_state, losses.mean()

    net_rows = []
    for tier, spec in ((t, TIERS[t]) for t in tiers):
        key = jrandom.PRNGKey(30)
        net = Net(**spec, key=key)
        params, static = eqx.partition(net, eqx.is_array)
        n_params = sum(x.size for x in jax.tree.leaves(params))
        optimizer = optax.chain(optax.clip_by_global_norm(0.267), optax.adam(1e-4))
        opt_state = optimizer.init(params)
        states = jax.tree.map(lambda x: x[:n], net_pool)
        osp1 = init_obs_state()
        osp = jax.tree.map(lambda x: jnp.tile(x, (2 * n, *([1] * x.ndim))), osp1)
        r_jit = make_rollout(static, net_pool)
        p_jit = make_ppo(static, optimizer)
        key = jrandom.PRNGKey(31)

        # Warmup (compile)
        t0 = time.time()
        states, osp, key, data = r_jit(params, states, osp, key)
        jax.block_until_ready(data[0])
        roll_compile = time.time() - t0
        t0 = time.time()
        key, k = jrandom.split(key)
        params, opt_state, loss = p_jit(params, opt_state, data, k)
        jax.block_until_ready(loss)
        ppo_compile = time.time() - t0

        roll_ts, ppo_ts = [], []
        for _ in range(3):
            t0 = time.time()
            states, osp, key, data = r_jit(params, states, osp, key)
            jax.block_until_ready(data[0])
            roll_ts.append(time.time() - t0)
            t0 = time.time()
            key, k = jrandom.split(key)
            params, opt_state, loss = p_jit(params, opt_state, data, k)
            jax.block_until_ready(loss)
            ppo_ts.append(time.time() - t0)
        roll_s = sum(roll_ts) / len(roll_ts)
        ppo_s = sum(ppo_ts) / len(ppo_ts)
        samples = 2 * n * T
        row = {
            "tier": tier, "params_m": round(n_params / 1e6, 2),
            "num_envs": n, "num_steps": T, "samples_per_iter": samples,
            "rollout_s": round(roll_s, 2), "ppo_s": round(ppo_s, 2),
            "samples_per_s": round(samples / (roll_s + ppo_s)),
            "rollout_env_steps_per_s": round(n * T / roll_s),
            "compile_s": round(roll_compile + ppo_compile, 1),
            **mem_stats(),
        }
        net_rows.append(row)
        log(f"net {tier} ({n_params / 1e6:.1f}M): {row['samples_per_s']:,} samples/s "
            f"(rollout {roll_s:.2f}s, ppo {ppo_s:.2f}s)")
        del net, params, static, opt_state, data
    results["net_in_loop"] = net_rows

    return results


@app.function(image=IMAGE, gpu="A10G", timeout=3600)
def bench_a10g() -> dict:
    return _bench_impl("A10G", env_counts=[1024, 4096, 16384],
                       pool_size_big=200_000, net_num_envs=1024, net_num_steps=64)


@app.function(image=IMAGE, gpu="H100", timeout=3600)
def bench_h100() -> dict:
    return _bench_impl("H100", env_counts=[1024, 4096, 16384],
                       pool_size_big=200_000, net_num_envs=1024, net_num_steps=64)


@app.function(image=IMAGE, gpu="A100-80GB", timeout=3600)
def bench_a100() -> dict:
    return _bench_impl("A100-80GB", env_counts=[1024, 4096, 16384],
                       pool_size_big=200_000, net_num_envs=1024, net_num_steps=64)


@app.function(image=IMAGE, gpu="A100-80GB", timeout=3600)
def bench_a100_big() -> dict:
    """Net loop at the frozen configs' 2048-env batch (S/M only)."""
    return _bench_impl("A100-80GB-2048", env_counts=[], pool_size_big=0,
                       net_num_envs=2048, net_num_steps=64, tiers="SM")


@app.function(image=IMAGE, gpu="H100", timeout=3600)
def bench_h100_big() -> dict:
    """Net loop at the frozen configs' 2048-env batch (S/M only)."""
    return _bench_impl("H100-2048", env_counts=[], pool_size_big=0,
                       net_num_envs=2048, net_num_steps=64, tiers="SM")


@app.local_entrypoint()
def main(gpus: str = "A10G,H100"):
    wanted = [g.strip().upper() for g in gpus.split(",") if g.strip()]
    fns = {"A10G": bench_a10g, "H100": bench_h100, "A100": bench_a100,
           "A100BIG": bench_a100_big, "H100BIG": bench_h100_big}
    calls = [(g, fns[g].spawn()) for g in wanted]
    out = {}
    for g, call in calls:
        out[g] = call.get()
        print(f"===== {g} done =====", flush=True)
    print("RESULTS_JSON_BEGIN", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print("RESULTS_JSON_END", flush=True)
