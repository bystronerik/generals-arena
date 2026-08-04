"""Replay-verify self-play shards (Part 11)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from arena.matches.loop import make_board, make_transition, winner_seat
from arena.records.store import engine_version as current_engine_version
from arena.records.trajectories import state_digest
from training.morpheus.self_play.schema import (
    SelfPlayShard,
    iter_shards,
    read_shard,
    value_targets_from_winner,
)
from training.morpheus.self_play.seats import truth_from_engine_state


@dataclass
class VerifyIssue:
    game_id: str
    path: str
    reason: str


@dataclass
class VerifyReport:
    ok: bool
    checked: int
    issues: list[VerifyIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "issues": [
                {"game_id": i.game_id, "path": i.path, "reason": i.reason}
                for i in self.issues
            ],
        }


def verify_shard(
    shard: SelfPlayShard,
    *,
    path: Path | None = None,
    engine: str | None = None,
) -> list[VerifyIssue]:
    """Replay joint actions; require matching digests, outcome, and value targets."""
    issues: list[VerifyIssue] = []
    label = str(path) if path is not None else shard.game_id
    engine_ver = engine or current_engine_version()
    if shard.engine_version != engine_ver:
        issues.append(
            VerifyIssue(
                game_id=shard.game_id,
                path=label,
                reason=(
                    f"engine era mismatch: shard={shard.engine_version} "
                    f"checkout={engine_ver}"
                ),
            )
        )
        return issues

    if shard.recursive_opponent_particles:
        issues.append(
            VerifyIssue(
                game_id=shard.game_id,
                path=label,
                reason="recursive_opponent_particles must be false (Part 05 default)",
            )
        )

    expected_values = value_targets_from_winner(shard.winner)
    if tuple(shard.value_targets) != expected_values:
        issues.append(
            VerifyIssue(
                game_id=shard.game_id,
                path=label,
                reason=(
                    f"value_targets {shard.value_targets} != "
                    f"winner-derived {expected_values}"
                ),
            )
        )

    from generals import GeneralsEnv

    env = GeneralsEnv(mode="competition")
    state = make_board(env, int(shard.seed))
    transition = make_transition(env)
    winner_player = -1
    info = None

    for frame in shard.turns:
        actions = jnp.stack(
            [
                jnp.array(frame.action_a, dtype=jnp.int32),
                jnp.array(frame.action_b, dtype=jnp.int32),
            ]
        )
        state, info = transition(state, actions)
        digests = truth_from_engine_state(state)
        if digests["state_digest"] != frame.state_digest:
            issues.append(
                VerifyIssue(
                    game_id=shard.game_id,
                    path=label,
                    reason=(
                        f"turn {frame.turn} state_digest mismatch: "
                        f"replay={digests['state_digest']} shard={frame.state_digest}"
                    ),
                )
            )
            break
        if digests["ownership_digest"] != frame.truth.ownership_digest:
            issues.append(
                VerifyIssue(
                    game_id=shard.game_id,
                    path=label,
                    reason=f"turn {frame.turn} ownership_digest mismatch",
                )
            )
            break
        if digests["armies_digest"] != frame.truth.armies_digest:
            issues.append(
                VerifyIssue(
                    game_id=shard.game_id,
                    path=label,
                    reason=f"turn {frame.turn} armies_digest mismatch",
                )
            )
            break
        if (
            int(info.land[0]),
            int(info.land[1]),
        ) != frame.truth.land or (
            int(info.army[0]),
            int(info.army[1]),
        ) != frame.truth.army:
            issues.append(
                VerifyIssue(
                    game_id=shard.game_id,
                    path=label,
                    reason=f"turn {frame.turn} land/army truth mismatch",
                )
            )
            break
        if bool(info.is_done):
            winner_player = int(info.winner)
            break

    truncated = winner_player < 0
    winner = winner_seat(winner_player, truncated=truncated)
    if winner != shard.winner:
        issues.append(
            VerifyIssue(
                game_id=shard.game_id,
                path=label,
                reason=f"outcome mismatch: replay={winner} shard={shard.winner}",
            )
        )

    # League shards must record a policy on both seats.
    if shard.source == "league":
        for frame in shard.turns:
            if frame.policy_a is None or frame.policy_b is None:
                issues.append(
                    VerifyIssue(
                        game_id=shard.game_id,
                        path=label,
                        reason="league shard missing policy on a seat (one-seat shortcut)",
                    )
                )
                break

    # Sanity: final digest exists when turns were played.
    if shard.turns:
        _ = state_digest(state)

    return issues


def verify_directory(directory: Path, *, engine: str | None = None) -> VerifyReport:
    directory = Path(directory)
    issues: list[VerifyIssue] = []
    checked = 0
    for path in iter_shards(directory):
        checked += 1
        shard = read_shard(path)
        issues.extend(verify_shard(shard, path=path, engine=engine))
    return VerifyReport(ok=len(issues) == 0 and checked > 0, checked=checked, issues=issues)
