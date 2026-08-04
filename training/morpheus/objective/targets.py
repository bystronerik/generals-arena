"""Training targets from terminal WDL, root average strategy, and engine truth.

Auxiliary labels never enter reward. Enemy-army bins use Part 04
``ARMY_BIN_EDGES`` — never a second edge table.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training.morpheus.objective.config import ScalarNormalization
from training.morpheus.objective.reward import terminal_reward, wdl_one_hot
from training.morpheus.self_play.schema import SparsePolicy

REPO = Path(__file__).resolve().parents[3]
MORPHEUS_BOT = REPO / "bots" / "morpheus"

Array = np.ndarray


def _ensure_bot_path() -> None:
    for entry in (REPO, REPO / "bots", MORPHEUS_BOT):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


def _schema():
    _ensure_bot_path()
    from schema import ARMY_BIN_EDGES, N_ARMY_BINS  # type: ignore

    return ARMY_BIN_EDGES, N_ARMY_BINS


def _action_consts():
    _ensure_bot_path()
    from action import N_ACTIONS, PAD  # type: ignore

    return N_ACTIONS, PAD


def army_bin_index(
    army: float,
    edges: Sequence[float] | None = None,
) -> int:
    """Map an army count onto the versioned Part 04 logarithmic bins."""
    army_edges, n_bins = _schema()
    use = tuple(edges) if edges is not None else army_edges
    if len(use) != n_bins + 1:
        raise ValueError(
            f"army bin edges must have length {n_bins + 1}, got {len(use)}"
        )
    x = max(float(army), 0.0)
    # edges[i] <= x < edges[i+1]; clamp last bin for x == edges[-1]
    idx = int(np.searchsorted(use, x, side="right") - 1)
    return int(max(0, min(n_bins - 1, idx)))


def sparse_policy_to_dense(
    policy: SparsePolicy | Mapping[str, Any] | None,
    *,
    n_actions: int | None = None,
) -> Array:
    """Expand normalized root average strategy ``S_A`` to a dense vector."""
    n, _ = _action_consts() if n_actions is None else (int(n_actions), None)
    dense = np.zeros(n, dtype=np.float64)
    if policy is None:
        return dense
    if isinstance(policy, SparsePolicy):
        indices = policy.indices
        probs = policy.probs
    else:
        indices = tuple(int(x) for x in policy["indices"])
        probs = tuple(float(x) for x in policy["probs"])
    if len(indices) != len(probs):
        raise ValueError("policy indices/probs length mismatch")
    total = 0.0
    for idx, p in zip(indices, probs):
        if idx < 0 or idx >= n:
            raise ValueError(f"policy index {idx} out of range for n={n}")
        dense[idx] = float(p)
        total += float(p)
    if indices and abs(total - 1.0) > 1e-5:
        raise ValueError(f"policy probs must sum to 1, got {total}")
    return dense


def normalize_margin(own: float, enemy: float) -> float:
    """``(own - enemy) / (own + enemy + 1)`` — labels only, never reward."""
    return (float(own) - float(enemy)) / (float(own) + float(enemy) + 1.0)


@dataclass(frozen=True)
class SeatTargets:
    """All training targets for one seat at one sampled state."""

    policy: Array  # (N_ACTIONS,) dense S_A
    wdl: Array  # (3,) win/draw/loss
    value: float  # p_win - p_loss = terminal reward under discount 1
    hidden_owner: Array  # (PAD, PAD) enemy ownership probability target
    enemy_army_bin: Array  # (PAD, PAD) int16 bin index; -1 where not enemy
    enemy_general: Array  # (PAD, PAD) one-hot enemy general
    hidden_castle: Array  # (PAD, PAD) enemy castle ownership
    land_margin: float
    army_margin: float
    castle_margin: float
    turns_to_termination: float
    board_mask: Array  # (PAD, PAD)

    def scaled_scalars(self, norm: ScalarNormalization) -> dict[str, float]:
        return {
            "land_margin": self.land_margin / norm.land_margin_scale,
            "army_margin": self.army_margin / norm.army_margin_scale,
            "castle_margin": self.castle_margin / norm.castle_margin_scale,
            "turns_to_termination": (
                self.turns_to_termination / norm.turns_to_termination_scale
            ),
        }


def _pad_plane(plane: Array, pad: int) -> Array:
    H, W = plane.shape
    out = np.zeros((pad, pad), dtype=plane.dtype)
    out[:H, :W] = plane
    return out


def hidden_state_targets_from_engine(
    *,
    ownership: Array,
    armies: Array,
    generals: Array,
    castles: Array,
    seat: int,
    army_bin_edges: Sequence[float] | None = None,
) -> dict[str, Array]:
    """Perspective-relative hidden-state labels from engine truth."""
    _, pad = _action_consts()
    _, n_bins = _schema()
    seat = int(seat)
    enemy = 1 - seat
    own_mask = np.asarray(ownership[seat], dtype=bool)
    enemy_mask = np.asarray(ownership[enemy], dtype=bool)
    armies_a = np.asarray(armies, dtype=np.float64)
    generals_a = np.asarray(generals, dtype=bool)
    castles_a = np.asarray(castles, dtype=bool)
    H, W = enemy_mask.shape

    hidden_owner = _pad_plane(enemy_mask.astype(np.float32), pad)
    enemy_general = _pad_plane(
        (generals_a & enemy_mask).astype(np.float32), pad
    )
    hidden_castle = _pad_plane(
        (castles_a & enemy_mask).astype(np.float32), pad
    )
    bins = np.full((H, W), -1, dtype=np.int16)
    for r, c in zip(*np.where(enemy_mask)):
        bins[r, c] = army_bin_index(float(armies_a[r, c]), army_bin_edges)
    enemy_army_bin = _pad_plane(bins, pad)
    board_mask = np.zeros((pad, pad), dtype=np.float32)
    board_mask[:H, :W] = 1.0
    _ = own_mask  # seat ownership is not an aux head; kept for clarity
    _ = n_bins
    return {
        "hidden_owner": hidden_owner,
        "enemy_army_bin": enemy_army_bin,
        "enemy_general": enemy_general,
        "hidden_castle": hidden_castle,
        "board_mask": board_mask,
    }


def final_margin_targets(
    *,
    land: Sequence[int],
    army: Sequence[int],
    castles: Sequence[int],
    seat: int,
) -> dict[str, float]:
    """Final land/army/castle margins from the sample seat. Labels only."""
    seat = int(seat)
    enemy = 1 - seat
    return {
        "land_margin": normalize_margin(land[seat], land[enemy]),
        "army_margin": normalize_margin(army[seat], army[enemy]),
        "castle_margin": normalize_margin(castles[seat], castles[enemy]),
    }


def turns_to_termination_target(
    *,
    current_turn: int,
    terminal_turn: int,
) -> float:
    """Non-negative turns remaining until termination (engine truth)."""
    remaining = int(terminal_turn) - int(current_turn)
    if remaining < 0:
        raise ValueError(
            f"terminal_turn {terminal_turn} before current_turn {current_turn}"
        )
    return float(remaining)


def build_seat_targets(
    *,
    winner: str,
    seat: int,
    policy: SparsePolicy | Mapping[str, Any] | None,
    ownership: Array,
    armies: Array,
    generals: Array,
    castles: Array,
    final_land: Sequence[int],
    final_army: Sequence[int],
    final_castles: Sequence[int],
    current_turn: int,
    terminal_turn: int,
    army_bin_edges: Sequence[float] | None = None,
) -> SeatTargets:
    """Assemble every Part 12 target for one seat at one state."""
    hidden = hidden_state_targets_from_engine(
        ownership=ownership,
        armies=armies,
        generals=generals,
        castles=castles,
        seat=seat,
        army_bin_edges=army_bin_edges,
    )
    margins = final_margin_targets(
        land=final_land,
        army=final_army,
        castles=final_castles,
        seat=seat,
    )
    wdl = np.asarray(wdl_one_hot(winner, seat=seat), dtype=np.float64)
    value = float(terminal_reward(winner, seat=seat))
    return SeatTargets(
        policy=sparse_policy_to_dense(policy),
        wdl=wdl,
        value=value,
        hidden_owner=hidden["hidden_owner"],
        enemy_army_bin=hidden["enemy_army_bin"],
        enemy_general=hidden["enemy_general"],
        hidden_castle=hidden["hidden_castle"],
        land_margin=margins["land_margin"],
        army_margin=margins["army_margin"],
        castle_margin=margins["castle_margin"],
        turns_to_termination=turns_to_termination_target(
            current_turn=current_turn,
            terminal_turn=terminal_turn,
        ),
        board_mask=hidden["board_mask"],
    )


def apply_root_noise(
    prior: Array,
    *,
    epsilon: float,
    alpha: float,
    rng: np.random.Generator,
) -> Array:
    """Mix Dirichlet noise into a root prior. No-op when epsilon is 0."""
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    eps = float(epsilon)
    if eps <= 0.0:
        s = float(prior.sum())
        return prior / s if s > 0 else prior
    if not (0.0 < eps <= 1.0):
        raise ValueError(f"root_noise_epsilon must be in (0, 1], got {eps}")
    a = float(alpha)
    if a <= 0.0:
        raise ValueError(f"root_noise_alpha must be > 0, got {a}")
    noise = rng.dirichlet(np.full(len(prior), a, dtype=np.float64))
    mixed = (1.0 - eps) * prior + eps * noise
    s = float(mixed.sum())
    return mixed / s if s > 0 else mixed


def sample_action_from_strategy(
    strategy: Array,
    *,
    temperature: float,
    turn: int,
    deterministic_turn: int,
    rng: np.random.Generator,
) -> int:
    """Sample from S_A with temperature; argmax after deterministic_turn."""
    probs = np.asarray(strategy, dtype=np.float64).reshape(-1)
    if int(turn) >= int(deterministic_turn):
        return int(np.argmax(probs))
    t = float(temperature)
    if t <= 0.0:
        raise ValueError(f"action_temperature must be > 0, got {t}")
    if abs(t - 1.0) < 1e-12:
        p = np.maximum(probs, 0.0)
    else:
        # Softmax temperature on log-space of the strategy mass.
        logp = np.log(np.maximum(probs, 1e-12)) / t
        logp = logp - logp.max()
        p = np.exp(logp)
    total = float(p.sum())
    if total <= 0.0:
        return int(rng.integers(0, len(p)))
    return int(rng.choice(len(p), p=p / total))
