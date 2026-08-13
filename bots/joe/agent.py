"""Joe: greedy EMA policy of the Average Joe self-play PPO run.

Phase 5 of docs/research/strategies/averagejoe-competition-plan.md. Each
turn is the training eval path (train/evaluations.py) run on one sample:
rebuild the engine's 14-channel observation tensor from the wire frame,
augment to 39 channels with the persistent AugmentedObsState, compute move
and build masks, one float32 forward pass, greedy argmax over the
10-channel action head.

The network weights are the EMA checkpoint exported by
``scripts/joe_export_bot.py`` into ``artifact/`` (EMA is the deployment
policy — plan section 5). ``artifact/manifest.json`` records the
architecture and the checkpoint provenance; the weights file is part of the
bot's content hash, so a re-export forks the rating identity.

Inference is jax-CPU float32, single-threaded (run.sh pins the pools; the
Phase 2 measurement is the budget evidence: M p99 ~19 ms on one x86 core
against the 150 ms limit, JIT compile absorbed by the first-move grace).
"""
import json
from pathlib import Path

import numpy as np

BOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BOT_DIR / "artifact"

# Wire protocol cell types (competition-module/competition/protocol.py).
TYPE_FOG = 0
TYPE_PLAIN = 1
TYPE_MOUNTAIN = 2
TYPE_CASTLE = 3
TYPE_GENERAL = 4
TYPE_STRUCTURE_IN_FOG = 5

OWNER_ME = 1
OWNER_OPP = 2


def frame_to_raw(obs) -> np.ndarray:
    """Wire frame -> the engine's (14, H, W) observation tensor.

    Inverts ``competition/protocol.py encode_observation``: every channel of
    ``Observation.as_tensor`` is recovered exactly. ``neutral_cells`` is
    ``ownership_neutral & visible``; ownership_neutral is passable cells
    owned by neither player, so on the wire it is a visible non-mountain
    cell with owner 0 (generals are always owned, so type 1 and type 3
    cover it).
    """
    type_grid = np.asarray(obs.type_grid, dtype=np.int32)
    owner_grid = np.asarray(obs.owner_grid, dtype=np.int32)
    army_grid = np.asarray(obs.army_grid, dtype=np.float32)

    raw = np.zeros((14, obs.H, obs.W), dtype=np.float32)
    raw[0] = army_grid
    raw[1] = type_grid == TYPE_GENERAL
    raw[2] = type_grid == TYPE_CASTLE
    raw[3] = type_grid == TYPE_MOUNTAIN
    raw[4] = (owner_grid == 0) & ((type_grid == TYPE_PLAIN) | (type_grid == TYPE_CASTLE))
    raw[5] = owner_grid == OWNER_ME
    raw[6] = owner_grid == OWNER_OPP
    raw[7] = type_grid == TYPE_FOG
    raw[8] = type_grid == TYPE_STRUCTURE_IN_FOG
    raw[9] = obs.my_land
    raw[10] = obs.my_army
    raw[11] = obs.opp_land
    raw[12] = obs.opp_army
    raw[13] = obs.turn
    return raw


class Agent:
    def __init__(self, player_id: int, H: int, W: int, artifact_dir=None):
        # Heavy imports live here, not at module scope: the wire handshake
        # arrives before the first observation, so all of jax init, weight
        # loading, and the JIT compile land inside the first-move grace.
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom

        from joe_net import HistoryTransformer
        from joe_obs import (
            augment_obs,
            build_cost_from_raw,
            compute_build_mask_from_raw,
            compute_valid_move_mask,
            decode_action,
            init_obs_state,
        )

        artifact_dir = Path(artifact_dir or ARTIFACT_DIR)
        with open(artifact_dir / "manifest.json") as f:
            manifest = json.load(f)
        arch = manifest["network"]
        pad_to = int(arch["pad_to"])
        if H > pad_to or W > pad_to:
            raise ValueError(f"board {H}x{W} exceeds the net's pad_to {pad_to}")

        template = HistoryTransformer(
            grid_size=pad_to,
            pad_to=pad_to,
            history_size=int(arch["history_size"]),
            patch_size=int(arch["patch_size"]),
            depth=int(arch["depth"]),
            embed_dim=int(arch["embed_dim"]),
            n_head=int(arch["n_head"]),
            ff_factor=int(arch["ff_factor"]),
            use_bf16=False,  # deployment is float32 (plan section 5)
            value_loss=arch["value_loss"],
            num_bins=int(arch["num_bins"]),
            v_min=float(arch["v_min"]),
            v_max=float(arch["v_max"]),
            key=jrandom.PRNGKey(0),
        )
        net = eqx.tree_deserialise_leaves(
            str(artifact_dir / manifest["weights"]), template)

        self._jnp = jnp
        self._np = np
        self._decode = lambda idx: decode_action(idx, pad_to)
        self.pad_to = pad_to
        self.H, self.W = H, W
        self.obs_state = init_obs_state(pad_to)
        self._init_obs_state = lambda: init_obs_state(pad_to)

        @eqx.filter_jit
        def step(net, raw, obs_state):
            """Full per-move path: cost, masks, augment, forward, greedy."""
            cost = build_cost_from_raw(raw)
            aug, obs_state = augment_obs(raw, cost, obs_state)
            move = compute_valid_move_mask(
                raw[0], raw[5] > 0, raw[3] > 0)
            build = compute_build_mask_from_raw(raw, cost)
            temporal = jnp.stack(
                [obs_state.opponent_army_history,
                 obs_state.opponent_land_history])
            logits, _, _ = net._forward(aug, move, build, temporal)
            action = decode_action(jnp.argmax(logits), pad_to)
            return action, obs_state

        self._net = net
        self._step = step

        # Compile now (first-move grace), on a frame-shaped dummy; then drop
        # the polluted obs_state — the real game starts from zeros.
        dummy = jnp.zeros((14, H, W), dtype=jnp.float32)
        action, _ = self._step(self._net, dummy, self.obs_state)
        np.asarray(action)
        self.obs_state = self._init_obs_state()

    def act(self, obs):
        raw = self._jnp.asarray(frame_to_raw(obs))
        action, self.obs_state = self._step(self._net, raw, self.obs_state)
        p, r, c, d, s = (int(x) for x in self._np.asarray(action))
        if p == 1:
            # Pass row/col may point into the pad region (the pass channel is
            # never masked); the engine ignores them, but keep the reply in
            # bounds anyway.
            return 1, 0, 0, 0, 0
        return p, r, c, d, s
