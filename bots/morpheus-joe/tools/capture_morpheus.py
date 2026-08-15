"""
Capture module for the morpheus-rs parity corpus and the M0 latency baseline.

Loaded by `arena.instrument.runner` through `arena.instrument.capture`, never
imported by anything that plays. It reaches into the Python morpheus agent's
internals to record, per turn, what a Rust reimplementation has to reproduce:
the observation it decided on, the RNG draws it consumed, the belief it held,
the tensor and priors the network saw, the heuristic shaping scores, the action
it played, and the per-component timings behind all of it.

Two rules shape everything here.

**Observation only.** The oracle is frozen (rewrite-plan §2); a capture that
changed a drawn value, an array, or an execution order would invalidate the
corpus it exists to produce. The RNG proxy delegates every call to the real
`numpy.random.Generator` and records the result. The tensor hook stashes the
reference the evaluator already returned. Everything else is read after the
move is on the wire, from state the agent left behind.

**Cheap every turn, heavy on selected turns.** Timings and scalars are recorded
for every turn — they are the M0 latency baseline, and a sampled baseline is a
worse baseline. Belief particles, root tensors, and shaping scores are recorded
only on turns the stratifier picks (§5's pre-contact / post-contact / recovery /
late-game / first-move strata), because a full-fidelity frame is ~100 KB and a
20-game corpus of them would be tens of gigabytes.

Format and field meanings: docs/bots/morpheus-rs/parity-corpus.md.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np

# Morpheus's own modules (`runtime`, `tactics`, `belief`) are imported lazily,
# never at module scope. The runner loads this file into *every* traced seat's
# process, and a bare `import runtime` there resolves against whichever bot
# that seat is playing — a foreign module, or none at all.

# Turns between heavy frames outside the always-heavy strata. Override with
# MORPHEUS_CAPTURE_STRIDE=1 to capture every turn at full fidelity (expensive).
DEFAULT_HEAVY_STRIDE = 8

# Deathtouch regime opens at turn 800 (RULES.md); §5 wants that stratum covered
# because it is where the transition kernel's rarest branch lives.
LATE_GAME_TURN = 800


def _stride() -> int:
    try:
        return max(1, int(os.environ.get("MORPHEUS_CAPTURE_STRIDE", "")))
    except ValueError:
        return DEFAULT_HEAVY_STRIDE


class RecordingGenerator:
    """
    A `numpy.random.Generator` that logs what it hands out.

    Every draw is delegated to the wrapped generator, so the bot's stream is
    bit-identical to an uncaptured run; the log is a side effect. The recorded
    values — not a reimplementation of numpy's bit generator — are what the
    Rust bot replays in parity mode (rewrite-plan §5), which makes *draw-site
    order* part of the ported contract: a Rust port that calls the sites in a
    different order consumes the stream differently and diverges visibly.

    Attribute access falls through, so any generator method the bot starts
    using keeps working. Only the three it actually calls are recorded; a
    fourth would pass through unlogged, so `unrecorded` counts them and the
    corpus report fails loudly rather than replaying a short stream.
    """

    _RECORDED = ("integers", "choice", "random")

    def __init__(self, inner: np.random.Generator) -> None:
        self._inner = inner
        self.draws: list[dict[str, Any]] = []
        self.unrecorded: dict[str, int] = {}

    # -- recorded draw sites -------------------------------------------------

    def integers(self, low, high=None, size=None, *args, **kwargs):
        out = self._inner.integers(low, high, size, *args, **kwargs)
        self._log("integers", {"low": low, "high": high, "size": size}, out)
        return out

    def choice(self, a, size=None, replace=True, p=None, *args, **kwargs):
        out = self._inner.choice(a, size, replace, p, *args, **kwargs)
        self._log(
            "choice",
            {
                "n": int(a) if np.isscalar(a) else int(np.asarray(a).shape[0]),
                "size": size,
                "replace": bool(replace),
                "weighted": p is not None,
            },
            out,
        )
        return out

    def random(self, size=None, *args, **kwargs):
        out = self._inner.random(size, *args, **kwargs)
        self._log("random", {"size": size}, out)
        return out

    # -- passthrough ---------------------------------------------------------

    def __getattr__(self, name: str):
        if name == "_inner":  # only before __init__ finishes; never recurse
            raise AttributeError(name)
        attr = getattr(self._inner, name)
        if callable(attr) and not name.startswith("_"):
            self.unrecorded[name] = self.unrecorded.get(name, 0)

            def _counted(*args, **kwargs):
                self.unrecorded[name] += 1
                return attr(*args, **kwargs)

            return _counted
        return attr

    # -- internals -----------------------------------------------------------

    def _log(self, method: str, args: dict[str, Any], out: Any) -> None:
        self.draws.append({"m": method, "a": args, "r": _as_plain(out)})

    def take(self) -> list[dict[str, Any]]:
        """Drain the draws logged since the last call."""
        drawn, self.draws = self.draws, []
        return drawn


def _as_plain(value: Any) -> Any:
    """Draw results as JSON scalars or lists — small by construction."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class _TensorHook:
    """
    Wraps `NetworkEvaluator._tensor`, keeping a reference to what it returned.

    Recomputing the root tensor after the move would be the obvious
    alternative and is wrong: `previous_action` has moved on by then, so the
    rebuilt tensor is not the one the network saw. Holding the reference costs
    a list append on the deadline path and captures the real input.
    """

    def __init__(self, evaluator) -> None:
        self._inner = evaluator._tensor
        self.tensors: list[Any] = []

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        self.tensors.append(out)
        return out

    def take_root(self):
        """The first tensor built this turn — the root evaluation's input."""
        built, self.tensors = self.tensors, []
        return (built[0] if built else None), len(built)


# ------------------------------------------------------------------- install

_STATE: dict[str, Any] = {}


def install(agent) -> None:
    """Wrap the agent's RNG and tensor builder before the first move."""
    controller = getattr(agent, "_controller", None)
    if controller is None:  # not morpheus — capture nothing
        _STATE.clear()
        return

    from runtime import PASS
    from tactics import OSCILLATION_HISTORY

    rng = RecordingGenerator(controller.rng)
    # Both holders, and they must be the *same* proxy: the controller and the
    # search share one generator, so the draw stream is one interleaved
    # sequence. Wrapping them separately would record two streams that no
    # replay could reassemble in the right order.
    controller.rng = rng
    controller.search.rng = rng

    tensor_hook = None
    evaluator = getattr(controller, "evaluator", None)
    if evaluator is not None and hasattr(evaluator, "_tensor"):
        tensor_hook = _TensorHook(evaluator)
        evaluator._tensor = tensor_hook

    _STATE.update(
        {
            "controller": controller,
            "rng": rng,
            "tensor_hook": tensor_hook,
            "seat": int(controller.seat),
            "stride": _stride(),
            "seen_contact": False,
            "stratum_counts": {},
            # The controller overwrites `_last_action` and `_recent_actions`
            # before the capture runs, so their pre-move values — the ones the
            # shaping and hard-rule layers actually saw — are mirrored here
            # instead of read back. Same seeds the controller starts from.
            "prev_action": PASS,
            "recent": deque(maxlen=OSCILLATION_HISTORY),
        }
    )


# --------------------------------------------------------------------- frame


def _grids(obs) -> dict[str, np.ndarray]:
    return {
        "type": np.asarray(obs.type_grid, dtype=np.int8),
        "owner": np.asarray(obs.owner_grid, dtype=np.int8),
        "army": np.asarray(obs.army_grid, dtype=np.int32),
    }


def _state_arrays(state) -> dict[str, Any]:
    return {
        "armies": np.asarray(state.armies, dtype=np.int32),
        "ownership": np.asarray(state.ownership, dtype=bool),
        "ownership_neutral": np.asarray(state.ownership_neutral, dtype=bool),
        "generals": np.asarray(state.generals, dtype=bool),
        "castles": np.asarray(state.castles, dtype=bool),
        "mountains": np.asarray(state.mountains, dtype=bool),
        "passable": np.asarray(state.passable, dtype=bool),
        "general_positions": np.asarray(state.general_positions, dtype=np.int32),
        "time": int(state.time),
        "winner": int(state.winner),
    }


def _memory_arrays(memory) -> dict[str, Any]:
    """Only the planes a tensor or a transition reads; H/W are on the frame."""
    return {
        name: np.asarray(getattr(memory, name))
        for name in (
            "known_mountain",
            "known_passable_base",
            "known_castle",
            "own_general",
            "known_enemy_general",
            "ever_visible",
            "last_seen_turn",
            "remembered_owner",
            "remembered_army",
            "remembered_was_castle",
            "remembered_castle_owner",
        )
    }


def _belief_snapshot(belief) -> dict[str, Any]:
    """
    Particles at full fidelity, histories as their action pairs only.

    A history frame carries a whole `GameState`, and eight particles times an
    eight-deep history is two thirds of a megabyte per turn. The action pairs
    are enough: replay reconstructs the intermediate states from the oldest
    recorded state through the transition kernel, which is exactly the
    property M4's rejuvenation parity gate is trying to check anyway.
    """
    from belief import ess, ess_fraction

    weights = [float(p.weight) for p in belief.particles]
    return {
        "seat": int(belief.seat),
        "collapsed": bool(belief.collapsed),
        "n_particles_config": int(belief.config.n_particles),
        "weights": weights,
        "ess": float(ess(weights)),
        "ess_fraction": float(ess_fraction(belief)),
        "particles": [
            {
                "state": _state_arrays(p.state),
                "weight": float(p.weight),
                "enemy_memory": _memory_arrays(p.enemy_memory),
                "enemy_prev_action": list(p.enemy_prev_action)
                if p.enemy_prev_action is not None
                else None,
                "history_actions": [
                    {"mine": list(h.my_action), "enemy": list(h.enemy_action)}
                    for h in p.history
                ],
                "history_oldest_state": _state_arrays(p.history[0].state)
                if p.history
                else None,
            }
            for p in belief.particles
        ],
    }


def _stratum(agent, obs, turn: int, contact: bool) -> str:
    """
    The §5 stratum this turn belongs to: a game phase, tagged with recovery.

    Phase and recovery are crossed rather than ranked. Ranking them starves
    whichever comes second — on an unqualified host the belief update is
    deferred on most turns, so a flat "recovery wins" rule files nearly every
    post-contact turn as recovery and the post-contact stratum empties out.
    The two facts are independent, so the stratum records both.
    """
    if turn == 1:
        return "first_move"
    if contact and not _STATE.get("seen_contact", False):
        return "first_contact"
    if int(obs.turn) >= LATE_GAME_TURN:
        phase = "late_game"
    else:
        phase = "post_contact" if contact else "pre_contact"
    return f"{phase}+recovery" if int(getattr(agent, "recovery", 0)) else phase


def _is_heavy(stratum: str) -> bool:
    """
    Every stride-th turn *within each stratum*, so the first is always heavy.

    Striding per stratum rather than over the whole game is what keeps rare
    strata represented: a flat `turn % stride` gives late-game and recovery
    coverage in proportion to how common they happen to be on the capture
    host, which is exactly the bias the stratification exists to remove.
    """
    counts = _STATE["stratum_counts"]
    seen = counts.get(stratum, 0)
    counts[stratum] = seen + 1
    return seen % _STATE.get("stride", DEFAULT_HEAVY_STRIDE) == 0


def frame(agent, obs, action, turn: int) -> dict[str, Any] | None:
    """One captured turn, sampled after the move is already on the wire."""
    if not _STATE:
        return None

    controller = _STATE["controller"]
    rng: RecordingGenerator = _STATE["rng"]
    memory = controller.memory
    belief = controller.belief

    from tactics import enemy_is_visible

    contact = bool(memory is not None and enemy_is_visible(obs, memory))
    stratum = _stratum(agent, obs, turn, contact)
    heavy = _is_heavy(stratum)
    _STATE["seen_contact"] = _STATE["seen_contact"] or contact
    prev_action = _STATE["prev_action"]
    recent_actions = [list(a) for a in _STATE["recent"]]

    captured: dict[str, Any] = {
        "t": turn,
        "seat": _STATE["seat"],
        "stratum": stratum,
        "heavy": heavy,
        "contact": contact,
        "obs": {
            "H": int(obs.H),
            "W": int(obs.W),
            "turn": int(obs.turn),
            "my_land": int(obs.my_land),
            "my_army": int(obs.my_army),
            "opp_land": int(obs.opp_land),
            "opp_army": int(obs.opp_army),
            **_grids(obs),
        },
        "action": [int(x) for x in action],
        # Pre-move inputs to the hard-rule layer, which `constrain_nn_action`
        # consumes alongside the belief (rewrite-plan §5, tier 3).
        "prev_action": [int(x) for x in prev_action],
        "recent_actions": recent_actions,
        # Timings and counters: every turn, because these are the baseline.
        "timing": {
            "move_ms": int(getattr(agent, "move_ms", 0)),
            "component_ms": {
                k: float(v) for k, v in getattr(agent, "component_ms", {}).items()
            },
            "component_calls": dict(getattr(agent, "component_calls", {})),
            "cost_belief_ms": int(getattr(agent, "cost_belief_ms", 0)),
            "cost_root_ms": int(getattr(agent, "cost_root_ms", 0)),
            "cost_search_ms": int(getattr(agent, "cost_search_ms", 0)),
            "cost_reply_ms": int(getattr(agent, "cost_reply_ms", 0)),
        },
        "counters": {
            "completed_simulations": int(getattr(agent, "completed_simulations", 0)),
            "forward_equivalents": int(getattr(agent, "forward_equivalents", 0)),
            "belief_ess_milli": int(getattr(agent, "belief_ess", 0)),
            "recovery": int(getattr(agent, "recovery", 0)),
            "tree_size": int(getattr(agent, "tree_size", 0)),
            "fallback_level": str(getattr(agent, "fallback_level", "pass")),
            "belief_plus_root_ok": int(getattr(agent, "belief_plus_root_ok", 0)),
            "belief_n": int(belief.n) if belief is not None else 0,
        },
        # The draw stream is small and is the thing that makes a replay
        # deterministic, so it is recorded on every turn regardless of stratum.
        "rng_draws": rng.take(),
        "rng_unrecorded": {k: v for k, v in rng.unrecorded.items() if v},
    }

    # Advance the mirror the same way the controller just did, so the next
    # turn's frame reports the inputs that turn will really be given.
    played = tuple(int(x) for x in action)
    _STATE["prev_action"] = played
    if played[0] == 0:
        _STATE["recent"].append(played)

    prior = getattr(controller.search, "last_root_prior", None)
    unshaped = getattr(controller.evaluator, "last_unshaped_prior", None)
    captured["root"] = {
        "has_result": int(getattr(agent, "has_root_result", 0)),
        "prior": np.asarray(prior, dtype=np.float64) if prior is not None else None,
        "unshaped_prior": np.asarray(unshaped, dtype=np.float64)
        if unshaped is not None
        else None,
    }

    tensor_hook = _STATE.get("tensor_hook")
    root_tensor, n_tensors = (None, 0)
    if tensor_hook is not None:
        root_tensor, n_tensors = tensor_hook.take_root()
    captured["root"]["tensors_built"] = n_tensors

    if not heavy:
        return captured

    if root_tensor is not None:
        captured["root"]["tensor"] = np.asarray(
            root_tensor.detach().cpu().numpy(), dtype=np.float32
        )
    if memory is not None:
        captured["memory"] = {"H": int(memory.H), "W": int(memory.W)}
        captured["memory"].update(_memory_arrays(memory))

        # Recomputed after the move, off the deadline path. Both are pure
        # functions of (obs, memory), so recomputing is exact — unlike the
        # tensor, which depends on state the turn has already advanced past.
        from tactics import heuristic_action_scores, play_mask

        mask = play_mask(obs, memory)
        captured["play_mask"] = np.asarray(mask, dtype=bool)
        captured["shaping_scores"] = np.asarray(
            heuristic_action_scores(
                obs,
                memory,
                mask,
                belief=belief,
                prev_action=prev_action,
            ),
            dtype=np.float64,
        )
    if belief is not None and belief.n > 0:
        captured["belief"] = _belief_snapshot(belief)

    return captured


def close() -> None:
    """Put the agent's internals back the way we found them."""
    controller = _STATE.get("controller")
    rng = _STATE.get("rng")
    if controller is not None and isinstance(rng, RecordingGenerator):
        controller.rng = rng._inner
        controller.search.rng = rng._inner
    hook = _STATE.get("tensor_hook")
    if controller is not None and isinstance(hook, _TensorHook):
        controller.evaluator._tensor = hook._inner
    _STATE.clear()
