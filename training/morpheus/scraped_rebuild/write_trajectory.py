"""Write an arena trajectory from fully-resolved inferred actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from arena.matches.loop import make_board, make_transition, winner_seat
from arena.records.store import engine_version as current_engine_version
from arena.records.trajectories import (
    TrajectoryRecorder,
    read_trajectory,
    trajectory_path,
    verify_trajectory,
)
from training.morpheus.scraped_rebuild.encode import tick_to_joint_action5
from training.morpheus.scraped_rebuild.infer import InferenceResult


@dataclass(frozen=True)
class WriteResult:
    ok: bool
    game_id: str
    path: Path | None
    reason: str
    ambiguous_ticks: int
    turns: int = 0
    verify_summary: str = ""


def game_id_for(player: str, match_id: str) -> str:
    safe_player = player.replace("/", "_").replace(" ", "_")
    return f"scraped_{safe_player}_{match_id}"


def write_trajectory_from_inference(
    inference: InferenceResult,
    *,
    player: str,
    directory: Path,
    engine: str | None = None,
    force: bool = False,
) -> WriteResult:
    """
    Step the real competition engine with inferred Action5 pairs and write
    a trajectory. Returns ``ok=False`` without keeping a file on failure.
    """
    game_id = game_id_for(player, inference.match_id)
    out_path = trajectory_path(game_id, directory)
    if out_path.is_file() and not force:
        return WriteResult(
            ok=True,
            game_id=game_id,
            path=out_path,
            reason="exists",
            ambiguous_ticks=len(inference.ambiguous),
            turns=inference.total_ticks,
            verify_summary="skipped_existing",
        )

    if inference.seed is None:
        return WriteResult(
            ok=False,
            game_id=game_id,
            path=None,
            reason="missing_seed",
            ambiguous_ticks=len(inference.ambiguous),
        )
    if not inference.ok:
        return WriteResult(
            ok=False,
            game_id=game_id,
            path=None,
            reason="unresolved",
            ambiguous_ticks=len(inference.ambiguous),
        )

    from generals import GeneralsEnv

    engine_ver = engine or current_engine_version()
    env = GeneralsEnv(mode="competition")
    state = make_board(env, int(inference.seed))
    transition = make_transition(env)

    bot_a, bot_b = inference.players[0], inference.players[1]
    recorder = TrajectoryRecorder(
        game_id=game_id,
        seed=int(inference.seed),
        mode="competition",
        round_name=directory.name,
        engine_version=engine_ver,
        bot_a=bot_a,
        bot_b=bot_b,
        directory=directory,
    )
    H = int(state.armies.shape[0])
    W = int(state.armies.shape[1])
    recorder.set_dims(H, W)

    winner_player = -1
    last_turn = 0
    info = None
    try:
        for tick in inference.ticks:
            turn = int(tick["t"])
            action_a, action_b = tick_to_joint_action5(tick)
            actions = jnp.stack(
                [
                    jnp.array(action_a, dtype=jnp.int32),
                    jnp.array(action_b, dtype=jnp.int32),
                ]
            )
            state, info = transition(state, actions)
            land = (int(info.land[0]), int(info.land[1]))
            army = (int(info.army[0]), int(info.army[1]))
            recorder.record_turn(
                turn, action_a, action_b, land, army, state=state
            )
            last_turn = turn
            if bool(info.is_done):
                winner_player = int(info.winner)
                break
    except Exception as exc:  # noqa: BLE001 — report and skip
        return WriteResult(
            ok=False,
            game_id=game_id,
            path=None,
            reason=f"engine_step_failed:{exc}",
            ambiguous_ticks=len(inference.ambiguous),
        )

    truncated = winner_player < 0
    # Prefer scraped terminal turn when engine did not end early.
    if last_turn == 0:
        return WriteResult(
            ok=False,
            game_id=game_id,
            path=None,
            reason="empty_actions",
            ambiguous_ticks=len(inference.ambiguous),
        )

    winner = winner_seat(winner_player, truncated=truncated)
    recorder.finish(
        winner=winner,
        turns=last_turn,
        terminated=winner_player >= 0,
        truncated=truncated,
        state=state,
    )
    path = recorder.write()
    traj = read_trajectory(path)
    report = verify_trajectory(traj, engine=engine_ver)
    if not report.ok:
        path.unlink(missing_ok=True)
        return WriteResult(
            ok=False,
            game_id=game_id,
            path=None,
            reason="verify_failed",
            ambiguous_ticks=len(inference.ambiguous),
            turns=last_turn,
            verify_summary=report.summary(),
        )

    return WriteResult(
        ok=True,
        game_id=game_id,
        path=path,
        reason="written",
        ambiguous_ticks=len(inference.ambiguous),
        turns=last_turn,
        verify_summary=report.summary(),
    )


def corpus_index_payload(
    *,
    player: str,
    source_label: str,
    round_name: str,
    engine_version: str,
    games: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "round": round_name,
        "player": player,
        "source_label": source_label,
        "engine_version": engine_version,
        "game_count": len(games),
        "games": games,
    }
