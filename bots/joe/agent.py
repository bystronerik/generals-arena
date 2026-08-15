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

# Repetition penalty. The network is a deterministic function of the board, so
# where the board is locally periodic the greedy argmax is periodic too: joe
# walks one stack around a structure it owns and never leaves. PPO collected its
# data by *sampling* this policy, which left such a loop within a few turns
# every time, so the gradient never had to learn an escape. See
# docs/research/measurements/joe-argmax-limit-cycle.md.
#
# A decayed count of the cells joe recently moved from, subtracted from every
# action logit at those cells, removes the fixed point without any randomness.
# A cell revisited every other turn settles at PENALTY / (1 - DECAY**2), so
# ~10.5 logits here — the same order as the ~11.3 logit median margin with which
# the net pulls army back onto its own structures, which is the force that
# closes the loop. A whole corridor accumulates more than any one of its cells,
# and one escape is enough. Isolated repeats stay far below this and are left
# alone.
#
# **Do not hand-tune these.** Neither is fitted, and game outcomes are not
# monotone in them: on the seed this was diagnosed from, 2.0 and 4.0 both win
# and 3.0 loses. Single games are chaotic in these constants, so a change needs
# a measurement round with both arms, not a seed or two.
REPEAT_PENALTY = 2.0
REPEAT_DECAY = 0.90

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
            N_ACTION_CHANNELS,
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
        def step(net, raw, obs_state, visits):
            """Full per-move path: cost, masks, augment, forward, selection.

            `logits` is what the network returned, before the repetition
            penalty — the probe reads it, and the fidelity test pins its argmax
            to the training path. The emitted action is the argmax of the
            *penalised* logits, which differ only at cells joe moved from
            recently.
            """
            cost = build_cost_from_raw(raw)
            aug, obs_state = augment_obs(raw, cost, obs_state)
            move = compute_valid_move_mask(
                raw[0], raw[5] > 0, raw[3] > 0)
            build = compute_build_mask_from_raw(raw, cost)
            temporal = jnp.stack(
                [obs_state.opponent_army_history,
                 obs_state.opponent_land_history])
            logits, value, _ = net._forward(aug, move, build, temporal)

            # One penalty per cell, applied to that cell's whole action column.
            # The masked entries are already -1e9, and the penalty is bounded by
            # REPEAT_PENALTY / (1 - REPEAT_DECAY) = 40, so this can never lift an
            # illegal action into contention.
            penalised = (logits.reshape(N_ACTION_CHANNELS, pad_to, pad_to)
                         - REPEAT_PENALTY * visits[None, :, :]).reshape(-1)
            idx = jnp.argmax(penalised)
            action = decode_action(idx, pad_to)

            # Count the cell joe moved *from*. Builds are one-shot per cell and
            # pass carries no cell, so neither accumulates a penalty.
            cells = pad_to * pad_to
            channel, position = idx // cells, idx % cells
            row, col = position // pad_to, position % pad_to
            new_visits = (visits * REPEAT_DECAY).at[row, col].add(
                jnp.where(channel < 8, 1.0, 0.0))

            overrode = idx != jnp.argmax(logits)
            return (action, obs_state, logits, value, move, build,
                    new_visits, overrode, idx)

        self._net = net
        self._step = step

        # Last decision, for bots/joe/probe.py. Left as device arrays: nothing
        # on the move path reads them, so no transfer happens in a normal game.
        self.logits = None
        self.value = None
        self.move_mask = None
        self.build_mask = None
        self.overrode = None
        self.action_idx = None

        self._init_visits = lambda: jnp.zeros((pad_to, pad_to), dtype=jnp.float32)
        self.visits = self._init_visits()

        # Compile now (first-move grace), on a frame-shaped dummy; then drop
        # the polluted obs_state and visit counts — the real game starts from
        # zeros, so joe's first move is the network's own argmax.
        dummy = jnp.zeros((14, H, W), dtype=jnp.float32)
        action, *_ = self._step(self._net, dummy, self.obs_state, self.visits)
        np.asarray(action)
        self.obs_state = self._init_obs_state()
        self.visits = self._init_visits()

    def act(self, obs):
        raw = self._jnp.asarray(frame_to_raw(obs))
        (action, self.obs_state, self.logits, self.value,
         self.move_mask, self.build_mask, self.visits,
         self.overrode, self.action_idx) = self._step(
            self._net, raw, self.obs_state, self.visits)
        p, r, c, d, s = (int(x) for x in self._np.asarray(action))
        if p == 1:
            # Pass row/col may point into the pad region (the pass channel is
            # never masked); the engine ignores them, but keep the reply in
            # bounds anyway.
            return 1, 0, 0, 0, 0
        return p, r, c, d, s
