"""Does the PPO update (forward+backward) depend on the transformer trunk?

The rollout ablation showed removing all 16 blocks leaves the rollout time
unchanged. PPO is the other half of the iteration and is the only place the
backward pass runs, so it gets the same treatment: identical rollout data,
identical optimizer, only SelfAttentionLayer.__call__ swapped.
"""
import json, os, sys, time
REPO = os.environ.get("JOE_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, optax
from training.joe.config import Config
from training.joe.env import make_competition_env, preset_min_generals_distance
from training.joe.networks import build_network, get_network_bundle
from training.joe.networks.transformer import SelfAttentionLayer
from training.joe.train.ppo import (_replicate, compute_gae,
                                    make_value_loss_fn, ppo_update)
from training.joe.train.rollout_selfplay import collect_rollout

_ORIG = SelfAttentionLayer.__call__


def identity_layer(self, x):
    return x


def log(m):
    print(f"[ppo-ab] {m}", flush=True)


cfg = Config.from_yaml(f"{REPO}/training/joe/configs/X16.yaml")
object.__setattr__(cfg, "pool_size", 8192)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
ND = jax.device_count()
N, T = cfg.num_envs, cfg.num_steps
bundle = get_network_bundle(cfg.network)
augment_fn, init_obs = bundle["augment_obs"], bundle["init_obs_state"]
vlf = make_value_loss_fn(cfg)

env = make_competition_env(preset_min_generals_distance(), None, cfg.pool_size)
pool, _ = env.reset(jrandom.PRNGKey(0)); jax.block_until_ready(pool.armies)
net = build_network(cfg, jrandom.PRNGKey(1))
params, static = eqx.partition(net, eqx.is_array)
opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                  optax.adam(cfg.lr))
p_params, p_opt = _replicate(params, ND), _replicate(opt.init(params), ND)
pool_rep = _replicate(pool, ND)
states = jax.pmap(lambda k: jax.vmap(env.init_state)(jrandom.split(k, N)))(
    jrandom.split(jrandom.PRNGKey(2), ND))
single = init_obs(cfg.pad_to, cfg.pad_to)
b = jax.tree.map(lambda x: jnp.tile(x, (N, *([1] * x.ndim))), single)
osp0, osp1 = _replicate(b, ND), _replicate(b, ND)
keys = jrandom.split(jrandom.PRNGKey(3), ND)

log("collecting one rollout with the real net for PPO input ...")
t0 = time.perf_counter()
_, rd, _, _, _ = jax.pmap(lambda p, s, k, a, bb, pl: collect_rollout(
    s, env, eqx.combine(p, static), k, T, a, bb, augment_fn, pl))(
    p_params, states, keys, osp0, osp1, pool_rep)
jax.block_until_ready(rd[0]); log(f"rollout {time.perf_counter()-t0:.1f}s")
(obs, mm, bm, tp, act, lps, vals, nvals, rews, term, trunc, win, own) = rd

advs = jax.pmap(lambda r, v, nv, te, tr: compute_gae(
    r, v, nv, te, tr, cfg.gamma, cfg.gae_lambda))(rews, vals, nvals, term, trunc)
rets = advs + vals
advs = (advs - advs.mean()) / (advs.std() + 1e-8)
tm = 1.0 - trunc.astype(jnp.float32)
tot = T * 2 * N
nk = (int(tot * cfg.adv_top_frac) // cfg.minibatch_size) * cfg.minibatch_size
sidx = jax.pmap(lambda a: jax.lax.top_k(jnp.abs(a.reshape(-1)), nk)[1])(advs)
batch = (obs, mm, bm, tp, act, lps, advs, rets, tm)
log(f"minibatches {nk // cfg.minibatch_size}, kept {nk:,} of {tot:,}")

def build():
    def _step(prm, ost, bt, key, si):
        n = eqx.combine(prm, static)
        n, ost, r = ppo_update(n, ost, bt, opt, key, cfg.clip_eps, cfg.vf_coef,
                               0.01, cfg.minibatch_size, vlf, si)
        np_, _ = eqx.partition(n, eqx.is_array)
        return np_, ost, jax.lax.pmean(r, axis_name="devices")
    return jax.pmap(_step, axis_name="devices")

fns = {}
for arm, impl in (("base", _ORIG), ("none", identity_layer)):
    SelfAttentionLayer.__call__ = impl
    t0 = time.perf_counter()
    f = build()
    o = f(p_params, p_opt, batch, keys, sidx)
    jax.block_until_ready(o[2]["total_loss"])
    fns[arm] = f
    log(f"{arm} compiled+ran {time.perf_counter()-t0:.1f}s")
    del o
SelfAttentionLayer.__call__ = _ORIG

times = {a: [] for a in fns}
for rnd in range(3):
    for arm, f in fns.items():
        t0 = time.perf_counter()
        for _ in range(2):
            o = f(p_params, p_opt, batch, keys, sidx)
            jax.block_until_ready(o[2]["total_loss"])
        dt = (time.perf_counter() - t0) / 2
        del o
        times[arm].append(round(dt, 4))
        log(f"round {rnd} {arm:<5} {dt:.3f}s per PPO update")

best = {a: min(v) for a, v in times.items()}
log("=" * 54)
log("PPO UPDATE ABLATION (best of 3 rounds)")
for a, v in best.items():
    log(f"  {a:<5} {v:8.3f}s   {best['base']/v:.3f}x vs base")
log(f"  transformer trunk share of PPO: "
    f"{100*(best['base']-best['none'])/best['base']:.1f}%")
log("=" * 54)
json.dump({"times_s": times, "best_s": best,
           "trunk_share_pct": round(100*(best['base']-best['none'])/best['base'], 2)},
          open(os.environ.get("JOE_OUT", "/tmp/joe-ppo-ablate.json"),
               "w"), indent=2)
print("AB_JSON_END", flush=True)
