"""End-to-end self-play throughput measurement for Part 13."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from training.morpheus.self_play.driver import DriverConfig, play_matchup
from training.morpheus.self_play.league import smoke_league
from training.morpheus.self_play.sampler import resolve_mixture, sample_matchup
from training.morpheus.self_play.schema import write_shard

REPO = Path(__file__).resolve().parents[3]


def measure_layout(
    layout: Mapping[str, Any],
    *,
    base_config: DriverConfig,
    output: Path,
    repo_root: Path | None = None,
    engine_version: str | None = None,
) -> dict[str, Any]:
    """Time completed games for one CPU/GPU layout. Never writes data/games/."""
    root = Path(repo_root) if repo_root is not None else REPO
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    games = int(layout.get("games") or base_config.games)
    concurrent = max(1, int(layout.get("concurrent_games") or 1))
    seat_search = (
        "parallel"
        if str(layout.get("seat_search", "sequential")).lower() == "parallel"
        else "sequential"
    )
    physical_cores = int(
        layout.get("physical_cores_per_game")
        or (2 if seat_search == "parallel" else 1)
    )
    backend = str(layout.get("backend") or "cpu")
    name = str(layout.get("name") or f"{backend}-{seat_search}-c{physical_cores}")
    engine = engine_version or layout.get("engine_version")

    cfg = replace(
        base_config,
        games=games,
        seat_search=seat_search,
        output=str(out),
    )

    game_latencies: list[float] = []
    completed = 0
    total_positions = 0
    t0 = time.perf_counter()

    if concurrent <= 1:
        # Single-threaded batch with per-game timing.
        league = smoke_league() if not cfg.league_path else None
        if cfg.league_path:
            from training.morpheus.self_play.league import League

            league = League.load(root / cfg.league_path)
        assert league is not None
        if not league.frozen:
            league.freeze_epoch()
        mixture = resolve_mixture(cfg.mixture(), repo_root=root)
        import numpy as np

        rng = np.random.default_rng(int(cfg.seed))
        for i in range(games):
            matchup = sample_matchup(rng, league, mixture)
            game_id = f"gate_{name}_{cfg.seed}_{i:04d}"
            g0 = time.perf_counter()
            shard = play_matchup(
                matchup,
                game_id=game_id,
                runtime_kwargs=cfg.runtime_kwargs(),
                max_turns=cfg.max_turns,
                seat_rng_salt=i,
                recursive_opponent_particles=cfg.recursive_opponent_particles,
                seat_search=seat_search,
                engine=str(engine) if engine else None,
            )
            write_shard(shard, out)
            game_latencies.append(time.perf_counter() - g0)
            total_positions += 2 * len(shard.turns)
            completed += 1
    else:
        # Concurrent games inside one container (worker maps).
        def _one(i: int) -> tuple[float, int]:
            local_cfg = replace(cfg, games=1, seed=int(cfg.seed) + i)
            g0 = time.perf_counter()
            # Inline one-game path so engine pin applies without git on Modal.
            league = smoke_league() if not local_cfg.league_path else None
            if local_cfg.league_path:
                from training.morpheus.self_play.league import League

                league = League.load(root / local_cfg.league_path)
            assert league is not None
            if not league.frozen:
                league.freeze_epoch()
            mixture = resolve_mixture(local_cfg.mixture(), repo_root=root)
            import numpy as np

            rng = np.random.default_rng(int(local_cfg.seed))
            matchup = sample_matchup(rng, league, mixture)
            shard = play_matchup(
                matchup,
                game_id=f"gate_{name}_{local_cfg.seed}_c{i}",
                runtime_kwargs=local_cfg.runtime_kwargs(),
                max_turns=local_cfg.max_turns,
                seat_rng_salt=i,
                recursive_opponent_particles=local_cfg.recursive_opponent_particles,
                seat_search=seat_search,
                engine=str(engine) if engine else None,
            )
            write_shard(shard, out / f"c{i}")
            return time.perf_counter() - g0, 2 * len(shard.turns)

        with ThreadPoolExecutor(max_workers=concurrent) as pool:
            futs = [pool.submit(_one, i) for i in range(games)]
            for fut in as_completed(futs):
                lat, pos = fut.result()
                game_latencies.append(lat)
                total_positions += pos
                completed += 1

    wall_s = time.perf_counter() - t0
    cpu_hours = (wall_s / 3600.0) * float(physical_cores) * float(concurrent)
    # Worker-hours for one container: wall hours (container occupancy).
    worker_hours = wall_s / 3600.0
    mean_latency = (
        sum(game_latencies) / len(game_latencies) if game_latencies else wall_s
    )
    positions_per_game = (
        float(total_positions) / float(completed) if completed else 0.0
    )
    games_per_worker_hour = (
        float(completed) / worker_hours if worker_hours > 0 else 0.0
    )

    return {
        "name": name,
        "backend": backend,
        "physical_cores_per_game": physical_cores,
        "seat_search": seat_search,
        "concurrent_games": concurrent,
        "completed_games": completed,
        "wall_s": wall_s,
        "cpu_hours": cpu_hours,
        "worker_hours": worker_hours,
        "mean_game_latency_s": mean_latency,
        "game_latencies_s": game_latencies,
        "positions_total": total_positions,
        "positions_per_game": positions_per_game,
        "games_per_worker_hour": games_per_worker_hour,
        "engine_version": engine,
        "output": str(out),
        "notes": list(layout.get("notes") or []),
    }
