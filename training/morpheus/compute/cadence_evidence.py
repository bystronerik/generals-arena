"""Part 13 cadence-pilot evidence: class WDL, belief calibration, pairwise verdict."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
import torch

from arena.matches.loop import make_transition
from arena.records.trajectories import read_trajectory
from training.morpheus.curriculum.reconstruct import (
    reconstruct_prefix,
    replay_prefix_states,
)
from training.morpheus.curriculum.schema import CurriculumItem
from training.morpheus.objective.config import load_objective_config
from training.morpheus.objective.losses import compute_objective_losses
from training.morpheus.trainer.sample import build_train_sample_from_recon

REPO = Path(__file__).resolve().parents[3]
MORPHEUS_BOT = REPO / "bots" / "morpheus"


def _ensure_bot_path(bot_dir: Path | None = None) -> Path:
    root = Path(bot_dir) if bot_dir is not None else MORPHEUS_BOT
    for entry in (REPO, REPO / "bots", root):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def _item_hash(item_id: str) -> int:
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_manifest_items(manifest_path: Path) -> list[CurriculumItem]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [CurriculumItem.from_dict(row) for row in payload["items"]]


def held_out_class_items(
    items: Sequence[CurriculumItem],
    *,
    class_id: int = 1,
    holdout_mod: int = 10,
    holdout_residue: int = 0,
    one_per_game: bool = True,
) -> list[CurriculumItem]:
    """Deterministic held-out class items (default: 10% by item_id hash)."""
    selected: list[CurriculumItem] = []
    best_by_game: dict[str, CurriculumItem] = {}
    for item in items:
        if int(item.class_id) != int(class_id):
            continue
        if _item_hash(item.item_id) % int(holdout_mod) != int(holdout_residue):
            continue
        if not one_per_game:
            selected.append(item)
            continue
        key = str(item.game_id or item.item_id)
        prev = best_by_game.get(key)
        # Prefer the latest class-1 prefix (closest to terminal).
        if prev is None or int(item.prefix_len) > int(prev.prefix_len):
            best_by_game[key] = item
    if one_per_game:
        selected = sorted(best_by_game.values(), key=lambda it: it.item_id)
    else:
        selected = sorted(selected, key=lambda it: it.item_id)
    return selected


@dataclass
class ClassWdlResult:
    active_classes: list[int]
    class_wdl: dict[str, dict[str, int]]
    games: int
    insufficient: bool
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_classes": list(self.active_classes),
            "class_wdl": {k: dict(v) for k, v in self.class_wdl.items()},
            "games": int(self.games),
            "insufficient": bool(self.insufficient),
            "details": list(self.details),
        }


def _smoke_agent(seat: int, H: int, W: int):
    smoke_dir = REPO / "bots" / "smoke"
    if str(smoke_dir) not in sys.path:
        sys.path.insert(0, str(smoke_dir))
    from agent import Agent  # type: ignore

    return Agent(player_id=int(seat), H=int(H), W=int(W))


def _morpheus_controller(
    *,
    seat: int,
    H: int,
    W: int,
    artifact_dir: Path,
    rng_seed: int,
    runtime_kwargs: Mapping[str, Any] | None = None,
):
    from training.morpheus.self_play.seats import (
        training_runtime_config,
    )

    _ensure_bot_path(artifact_dir.parent if (artifact_dir.parent / "agent.py").is_file() else None)
    # Prefer the checkpoint bot package so deployment mirrors the installed bot.
    bot_root = artifact_dir.parent
    _ensure_bot_path(bot_root)
    from evaluator import NetworkEvaluator  # type: ignore
    from inference import load_session  # type: ignore
    from runtime import RuntimeController  # type: ignore

    session = load_session(Path(artifact_dir))
    evaluator = NetworkEvaluator(session)
    kwargs = {
        "n_particles": 2,
        "target_simulations": 2,
        "min_simulations": 1,
        "search_depth": 1,
        "pending_leaf_batch": 1,
        "deadline_ms": 200.0,
    }
    if runtime_kwargs:
        kwargs.update(dict(runtime_kwargs))
    cfg = training_runtime_config(**kwargs)
    return RuntimeController(
        seat=int(seat),
        H=int(H),
        W=int(W),
        evaluator=evaluator,
        config=cfg,
        rng=np.random.default_rng(int(rng_seed)),
    )


def continue_prefix_vs_smoke(
    item: CurriculumItem,
    *,
    artifact_dir: Path,
    max_turns: int = 64,
    n_particles: int = 4,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay a curriculum prefix, then play Morpheus vs smoke to terminal."""
    if item.sample_seat is None:
        raise ValueError(f"{item.item_id}: sample_seat required")
    seat = int(item.sample_seat)
    traj = read_trajectory(Path(item.trajectory_relpath))
    # Engine replay only — belief reconstruct is not required for WDL play.
    _turn, state = replay_prefix_states(
        traj=traj,
        map_seed=int(item.map_seed),
        prefix_len=int(item.prefix_len),
    )
    H = int(np.asarray(state.armies).shape[0])
    W = int(np.asarray(state.armies).shape[1])

    from generals import GeneralsEnv
    from observe import emit_observation
    from state import from_engine

    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    morpheus = _morpheus_controller(
        seat=seat,
        H=H,
        W=W,
        artifact_dir=Path(artifact_dir),
        rng_seed=_item_hash(item.item_id) ^ (0x9E3779B97F4A7C15 + seat),
        runtime_kwargs=runtime_kwargs,
    )
    smoke = _smoke_agent(1 - seat, H, W)

    winner_player = -1
    info = None
    steps = 0
    for _ in range(int(max_turns)):
        m_state = from_engine(state)
        obs0 = emit_observation(m_state, 0)
        obs1 = emit_observation(m_state, 1)
        if seat == 0:
            a0 = morpheus.decide(obs0)
            a1 = smoke.act(obs1)
        else:
            a0 = smoke.act(obs0)
            a1 = morpheus.decide(obs1)
        actions = jnp.stack(
            [
                jnp.asarray(a0, dtype=jnp.int32),
                jnp.asarray(a1, dtype=jnp.int32),
            ]
        )
        state, info = transition(state, actions)
        steps += 1
        if bool(getattr(info, "is_done", False)):
            winner_player = int(info.winner)
            break

    if winner_player == seat:
        result = "win"
    elif winner_player < 0:
        result = "draw"
    else:
        result = "loss"
    return {
        "item_id": item.item_id,
        "game_id": item.game_id,
        "class_id": int(item.class_id),
        "sample_seat": seat,
        "prefix_len": int(item.prefix_len),
        "steps": steps,
        "result": result,
        "winner_player": winner_player,
    }


def measure_class_wdl(
    items: Sequence[CurriculumItem],
    *,
    artifact_dir: Path,
    class_id: int = 1,
    min_wins: int = 32,
    min_losses: int = 32,
    max_games: int = 512,
    n_particles: int = 4,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> ClassWdlResult:
    """Play held-out class prefixes until win/loss floors or exhaustion."""
    wins = losses = draws = 0
    details: list[dict[str, Any]] = []
    for item in items:
        if len(details) >= int(max_games):
            break
        if int(item.class_id) != int(class_id):
            continue
        row = continue_prefix_vs_smoke(
            item,
            artifact_dir=artifact_dir,
            n_particles=n_particles,
            runtime_kwargs=runtime_kwargs,
        )
        details.append(row)
        if row["result"] == "win":
            wins += 1
        elif row["result"] == "loss":
            losses += 1
        else:
            draws += 1
        print(
            f"[class-wdl] game={len(details)} "
            f"W={wins} L={losses} D={draws} "
            f"result={row['result']} steps={row['steps']} "
            f"item={row['item_id']}",
            flush=True,
        )
        if wins >= int(min_wins) and losses >= int(min_losses):
            break
    insufficient = wins < int(min_wins) or losses < int(min_losses)
    return ClassWdlResult(
        active_classes=[int(class_id)],
        class_wdl={
            str(int(class_id)): {
                "wins": int(wins),
                "losses": int(losses),
                "draws": int(draws),
            }
        },
        games=len(details),
        insufficient=insufficient,
        details=details,
    )


def _load_float_model(checkpoint_dir: Path):
    bot = _ensure_bot_path()
    sys.path.insert(0, str(bot))
    from export import build_model_from_checkpoint, load_checkpoint_state  # type: ignore

    ckpt = load_checkpoint_state(Path(checkpoint_dir))
    model = build_model_from_checkpoint(ckpt)
    model.eval()
    return model


def _belief_primary_loss(model, sample, objective) -> dict[str, float]:
    """Belief-head losses on one train sample; lower is better."""
    from network import flatten_policy_logits  # type: ignore

    x = torch.as_tensor(sample.tensor, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        policy_logits = flatten_policy_logits(out.policy, out.pass_logit)
    t = sample.targets
    scaled = t.scaled_scalars(objective.scalar_normalization)
    _, terms = compute_objective_losses(
        config=objective,
        policy_logits=policy_logits,
        wdl_logits=out.wdl_logits,
        hidden_owner_logits=out.hidden_owner,
        enemy_army_logits=out.enemy_army_bins,
        enemy_general_logits=out.enemy_general,
        hidden_castle_logits=out.hidden_castle,
        land_margin_pred=out.land_margin.view(-1),
        army_margin_pred=out.army_margin.view(-1),
        castle_margin_pred=out.castle_margin.view(-1),
        turns_pred=out.turns_to_termination.view(-1),
        policy_target=torch.as_tensor(t.policy, dtype=torch.float32).unsqueeze(0),
        wdl_target=torch.as_tensor(t.wdl, dtype=torch.float32).unsqueeze(0),
        hidden_owner_target=torch.as_tensor(t.hidden_owner, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0),
        enemy_army_bin_target=torch.as_tensor(t.enemy_army_bin, dtype=torch.int64).unsqueeze(
            0
        ),
        enemy_general_target=torch.as_tensor(t.enemy_general, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0),
        hidden_castle_target=torch.as_tensor(t.hidden_castle, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0),
        land_margin_target=torch.as_tensor([scaled["land_margin"]], dtype=torch.float32),
        army_margin_target=torch.as_tensor([scaled["army_margin"]], dtype=torch.float32),
        castle_margin_target=torch.as_tensor(
            [scaled["castle_margin"]], dtype=torch.float32
        ),
        turns_target=torch.as_tensor(
            [scaled["turns_to_termination"]], dtype=torch.float32
        ),
        board_mask=torch.as_tensor(t.board_mask, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0),
    )
    belief_keys = (
        "hidden_owner",
        "enemy_army_bins",
        "enemy_general",
        "hidden_castle",
    )
    parts = {k: float(getattr(terms, k)) for k in belief_keys}
    parts["primary"] = float(sum(parts.values()) / len(parts))
    return parts


def measure_belief_calibration(
    items: Sequence[CurriculumItem],
    *,
    prior_checkpoint: Path,
    later_checkpoint: Path,
    objective_path: Path | None = None,
    max_items: int = 64,
    n_particles: int = 4,
) -> dict[str, Any]:
    """Compare float checkpoints on held-out belief-head loss (lower better)."""
    obj_path = Path(
        objective_path or (REPO / "training/morpheus/configs/pilot-objective.json")
    )
    objective = load_objective_config(obj_path)
    prior_model = _load_float_model(prior_checkpoint)
    later_model = _load_float_model(later_checkpoint)
    prior_rows: list[dict[str, float]] = []
    later_rows: list[dict[str, float]] = []
    used: list[str] = []
    terminal_cache: dict[str, dict[str, Any]] = {}

    for item in items:
        if len(used) >= int(max_items):
            break
        if item.sample_seat is None or not item.trajectory_relpath:
            continue
        traj = read_trajectory(Path(item.trajectory_relpath))
        recon = reconstruct_prefix(item, traj=traj, n_particles=int(n_particles))
        sample = build_train_sample_from_recon(
            item, recon, traj=traj, terminal_cache=terminal_cache
        )
        prior_rows.append(_belief_primary_loss(prior_model, sample, objective))
        later_rows.append(_belief_primary_loss(later_model, sample, objective))
        used.append(item.item_id)
        print(
            f"[belief] scored {len(used)}/{int(max_items)} item={item.item_id}",
            flush=True,
        )

    def _mean(rows: Sequence[Mapping[str, float]], key: str) -> float:
        if not rows:
            return float("nan")
        return float(sum(float(r[key]) for r in rows) / len(rows))

    prior_primary = _mean(prior_rows, "primary")
    later_primary = _mean(later_rows, "primary")
    regressed = bool(later_primary > prior_primary) if used else True
    return {
        "measurement": "held_out_belief_head_soft_ce",
        "primary_metric": "mean_belief_head_loss",
        "n_items": len(used),
        "item_ids": used,
        "prior": {
            "checkpoint": str(prior_checkpoint),
            "primary": prior_primary,
            "hidden_owner": _mean(prior_rows, "hidden_owner"),
            "enemy_army_bins": _mean(prior_rows, "enemy_army_bins"),
            "enemy_general": _mean(prior_rows, "enemy_general"),
            "hidden_castle": _mean(prior_rows, "hidden_castle"),
        },
        "later": {
            "checkpoint": str(later_checkpoint),
            "primary": later_primary,
            "hidden_owner": _mean(later_rows, "hidden_owner"),
            "enemy_army_bins": _mean(later_rows, "enemy_army_bins"),
            "enemy_general": _mean(later_rows, "enemy_general"),
            "hidden_castle": _mean(later_rows, "hidden_castle"),
        },
        "belief_calibration_regressed": regressed,
        "delta_later_minus_prior": float(later_primary - prior_primary)
        if used
        else None,
    }


def decision_rule_verdict(delta) -> str:
    """Map fit.delta(...) to the docs/arena/decision-rule.md verdict."""
    if not getattr(delta, "comparable", True):
        return "unproven"
    p = float(delta.p_stronger)
    lo, hi = float(delta.ci[0]), float(delta.ci[1])
    if p >= 0.95 and lo > 10.0:
        return "improvement"
    if p <= 0.05 and hi < -10.0:
        return "regression"
    if lo >= -25.0 and hi <= 25.0:
        return "no change"
    return "unproven"


def pairwise_star_specs(
    arms: Sequence[Path],
    panel: Sequence[Path],
    *,
    games_per_pair: int,
    round_seed: int,
    seat_policy: str = "alternate",
) -> list[tuple[Path, Path, int]]:
    """Only arm-vs-panel pairs (no panel-vs-panel), matched seeds per arm."""
    from arena.tournaments.competition import expand_pair_seeds

    specs: list[tuple[Path, Path, int]] = []
    for arm in arms:
        pairs = [(Path(arm), Path(opp)) for opp in panel]
        specs.extend(
            expand_pair_seeds(
                pairs,
                games_per_pair=int(games_per_pair),
                round_seed=int(round_seed),
                seat_policy=seat_policy,
            )
        )
    return specs
