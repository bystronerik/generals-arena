"""In-process self-play / panel game driver (Part 11)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import jax.numpy as jnp
import numpy as np

from arena.matches.loop import make_board, make_transition, winner_seat
from arena.records.store import engine_version as current_engine_version
from training.morpheus.self_play.league import League, smoke_league
from training.morpheus.self_play.sampler import (
    Matchup,
    MixtureConfig,
    PanelMember,
    mixture_proportions,
    resolve_mixture,
    sample_matchup,
)
from training.morpheus.self_play.schema import (
    SelfPlayShard,
    ShardTurn,
    TruthTargets,
    value_targets_from_winner,
    write_shard,
)
from training.morpheus.self_play.seats import (
    _ensure_morpheus_path,
    build_seat_player,
    truth_from_engine_state,
)

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DriverConfig:
    games: int = 2
    seed: int = 0
    max_turns: int = 16
    league_weight: float = 0.7
    panel_weight: float = 0.3
    panel_path: str | None = None
    panel_members: tuple[dict[str, Any], ...] = ()
    n_particles: int = 4
    target_simulations: int = 8
    min_simulations: int = 4
    search_depth: int = 4
    pending_leaf_batch: int = 2
    deadline_ms: float = 60_000.0
    output: str = "data/morpheus/self_play"
    league_path: str | None = None
    recursive_opponent_particles: bool = False
    # Part 13: "sequential" = one core, seat A then B; "parallel" = one seat per core.
    seat_search: Literal["sequential", "parallel"] = "sequential"

    def runtime_kwargs(self) -> dict[str, Any]:
        return {
            "n_particles": self.n_particles,
            "target_simulations": self.target_simulations,
            "min_simulations": self.min_simulations,
            "search_depth": self.search_depth,
            "pending_leaf_batch": self.pending_leaf_batch,
            "deadline_ms": self.deadline_ms,
        }

    def mixture(self) -> MixtureConfig:
        members = tuple(PanelMember.from_dict(m) for m in self.panel_members)
        return MixtureConfig(
            league_weight=self.league_weight,
            panel_weight=self.panel_weight,
            panel_path=self.panel_path,
            panel_members=members,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DriverConfig:
        return cls(
            games=int(data.get("games", 2)),
            seed=int(data.get("seed", 0)),
            max_turns=int(data.get("max_turns", 16)),
            league_weight=float(data.get("league_weight", 0.7)),
            panel_weight=float(data.get("panel_weight", 0.3)),
            panel_path=(
                str(data["panel_path"]) if data.get("panel_path") is not None else None
            ),
            panel_members=tuple(data.get("panel_members") or ()),
            n_particles=int(data.get("n_particles", 4)),
            target_simulations=int(data.get("target_simulations", 8)),
            min_simulations=int(data.get("min_simulations", 4)),
            search_depth=int(data.get("search_depth", 4)),
            pending_leaf_batch=int(data.get("pending_leaf_batch", 2)),
            deadline_ms=float(data.get("deadline_ms", 60_000.0)),
            output=str(data.get("output", "data/morpheus/self_play")),
            league_path=(
                str(data["league_path"]) if data.get("league_path") is not None else None
            ),
            recursive_opponent_particles=bool(
                data.get("recursive_opponent_particles", False)
            ),
            seat_search=(
                "parallel"
                if str(data.get("seat_search", "sequential")).lower() == "parallel"
                else "sequential"
            ),
        )

    @classmethod
    def load(cls, path: Path) -> DriverConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class BatchResult:
    shards: list[Path]
    matchups: list[Matchup]
    proportions: dict[str, float]
    output: Path


def _load_or_smoke_league(path: str | None) -> League:
    if path:
        return League.load(Path(path))
    return smoke_league()


def _act_both_seats(
    players: list[Any],
    obs0: Any,
    obs1: Any,
    *,
    seat_search: Literal["sequential", "parallel"],
) -> tuple[Any, Any, Any, Any]:
    """Return (action_a, policy_a, action_b, policy_b)."""
    if seat_search == "parallel":
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(players[0].act, obs0)
            fut_b = pool.submit(players[1].act, obs1)
            action_a, policy_a = fut_a.result()
            action_b, policy_b = fut_b.result()
        return action_a, policy_a, action_b, policy_b
    action_a, policy_a = players[0].act(obs0)
    action_b, policy_b = players[1].act(obs1)
    return action_a, policy_a, action_b, policy_b


def play_matchup(
    matchup: Matchup,
    *,
    game_id: str,
    runtime_kwargs: dict[str, Any],
    max_turns: int,
    engine: str | None = None,
    recursive_opponent_particles: bool = False,
    seat_rng_salt: int = 0,
    seat_search: Literal["sequential", "parallel"] = "sequential",
) -> SelfPlayShard:
    """Run one competition game in-process and return a replayable shard."""
    if recursive_opponent_particles:
        raise ValueError(
            "Part 05/11 default forbids recursive opponent particles; "
            "set recursive_opponent_particles=false"
        )
    if seat_search not in ("sequential", "parallel"):
        raise ValueError(f"unknown seat_search={seat_search!r}")

    _ensure_morpheus_path()
    from generals import GeneralsEnv
    from observe import emit_observation
    from state import from_engine

    engine_ver = engine or current_engine_version()
    env = GeneralsEnv(mode="competition")
    state = make_board(env, int(matchup.map_seed))
    transition = make_transition(env)
    H = int(state.armies.shape[0])
    W = int(state.armies.shape[1])

    players = [
        build_seat_player(
            matchup.seats[seat],
            H=H,
            W=W,
            rng_seed=10_000 * int(seat_rng_salt) + 17 * seat + int(matchup.map_seed),
            runtime_kwargs=runtime_kwargs,
        )
        for seat in (0, 1)
    ]
    if matchup.source == "league":
        if not all(p.uses_full_stack for p in players):
            raise RuntimeError("league matchup requires full stack on both seats")

    turns: list[ShardTurn] = []
    winner_player = -1
    info = None
    last_turn = 0

    for step in range(int(max_turns)):
        m_state = from_engine(state)
        obs0 = emit_observation(m_state, 0)
        obs1 = emit_observation(m_state, 1)
        action_a, policy_a, action_b, policy_b = _act_both_seats(
            players, obs0, obs1, seat_search=seat_search
        )
        actions = jnp.stack(
            [
                jnp.array(action_a, dtype=jnp.int32),
                jnp.array(action_b, dtype=jnp.int32),
            ]
        )
        state, info = transition(state, actions)
        digests = truth_from_engine_state(state)
        truth = TruthTargets(
            ownership_digest=digests["ownership_digest"],
            armies_digest=digests["armies_digest"],
            generals_digest=digests["generals_digest"],
            castles_digest=digests["castles_digest"],
            land=(int(info.land[0]), int(info.land[1])),
            army=(int(info.army[0]), int(info.army[1])),
        )
        turn = int(info.time) if hasattr(info, "time") else step + 1
        # Prefer engine timestep when present.
        try:
            turn = int(np.asarray(state.time).reshape(-1)[0])
        except Exception:  # noqa: BLE001
            turn = step + 1
        turns.append(
            ShardTurn(
                turn=turn,
                action_a=tuple(int(x) for x in action_a),
                action_b=tuple(int(x) for x in action_b),
                policy_a=policy_a,
                policy_b=policy_b,
                truth=truth,
                state_digest=digests["state_digest"],
            )
        )
        last_turn = turn
        if bool(info.is_done):
            winner_player = int(info.winner)
            break

    truncated = winner_player < 0
    winner = winner_seat(winner_player, truncated=truncated)
    values = value_targets_from_winner(winner)
    return SelfPlayShard(
        game_id=game_id,
        seed=int(matchup.map_seed),
        engine_version=engine_ver,
        source=matchup.source,
        learner_seat=int(matchup.learner_seat),
        seats=(matchup.seats[0].to_dict(), matchup.seats[1].to_dict()),
        H=H,
        W=W,
        turns=turns,
        winner=winner,
        terminated=winner_player >= 0,
        truncated=truncated,
        value_targets=values,
        recursive_opponent_particles=False,
        notes=[
            f"max_turns={max_turns}",
            f"last_turn={last_turn}",
            f"seat_search={seat_search}",
            "part=11",
        ],
    )


def run_batch(
    config: DriverConfig,
    *,
    output: Path | None = None,
    repo_root: Path | None = None,
    engine: str | None = None,
) -> BatchResult:
    """Sample matchups, play games, write shards. Never writes data/games/.

    Pass ``engine`` when the process cannot read competition-module HEAD
    (Modal images copy the submodule without ``.git``).
    """
    root = Path(repo_root) if repo_root is not None else REPO
    out = Path(output) if output is not None else Path(config.output)
    if not out.is_absolute():
        out = root / out
    out_posix = out.resolve().as_posix()
    games_root = (root / "data" / "games").resolve().as_posix()
    if out_posix == games_root or out_posix.startswith(games_root + "/"):
        raise ValueError("refusing to write self-play shards into data/games/")

    league = _load_or_smoke_league(
        str(root / config.league_path) if config.league_path else None
    )
    if not league.frozen:
        league.freeze_epoch()

    mixture = resolve_mixture(config.mixture(), repo_root=root)
    rng = np.random.default_rng(int(config.seed))
    matchups: list[Matchup] = []
    paths: list[Path] = []

    for i in range(int(config.games)):
        matchup = sample_matchup(rng, league, mixture)
        matchups.append(matchup)
        game_id = f"sp_{config.seed}_{i:04d}_{matchup.source}"
        shard = play_matchup(
            matchup,
            game_id=game_id,
            runtime_kwargs=config.runtime_kwargs(),
            max_turns=config.max_turns,
            engine=engine,
            seat_rng_salt=i,
            recursive_opponent_particles=config.recursive_opponent_particles,
            seat_search=config.seat_search,
        )
        paths.append(write_shard(shard, out))

    return BatchResult(
        shards=paths,
        matchups=matchups,
        proportions=mixture_proportions(matchups),
        output=out,
    )
