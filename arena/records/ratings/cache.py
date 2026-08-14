"""
Per-round count caches, so a refit does not re-read every game ever played.

A cached round is not an approximation of the work, it *is* the work, already
done: the count table is the round's sufficient statistic. Refitting after one
new round then costs O(#rounds) instead of O(#games).

The round is also the unit ratings are fitted over (`rounds.py`), so this module
hands out per-round tables and `cached_count_table` is the one that composes
them — not the other way round.

Two things invalidate a cached round:

1. **The round's files changed** — keyed on a digest of `(name, size, mtime_ns)`
   over that directory, so an added, removed, or rewritten game busts it.
2. **The rules changed** — keyed on the policy and on which hashes are
   registered. A cache that survived a policy edit would silently answer the
   old question.

At the current scale this is optional; it is what keeps the refit under a
second at 100x the games.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from arena.records.ratings.counts import CountTable, count_table, merge
from arena.records.ratings.policy import Policy
from arena.records.registry import Registry
from arena.records.store import GAMES_DIR, list_game_paths, load_game

CACHE_DIRNAME = "cache"
# v2 added `engine_versions`: the distinct eras among the round's *stored*
# games, which a per-round report names so an era-spanning round is visible
# rather than inferred.
CACHE_FORMAT_VERSION = 2

# Games sitting directly in data/games/ rather than in a round folder.
ROOT_ROUND = "_root"


@dataclass(frozen=True)
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def rounds(self) -> int:
        return self.hits + self.misses


@dataclass(frozen=True)
class RoundCounts:
    """One round's sufficient statistics, plus the provenance a report needs."""

    round: str
    table: CountTable
    # Distinct `engine_version` values over the round's stored games, before the
    # era filter. More than one means the round spans an engine bump, which the
    # era filter then silently halves — so it is reported, not inferred.
    engine_versions: tuple[str, ...] = ()

    @property
    def stored_games(self) -> int:
        """Records in the round: the rated ones plus every rejection."""
        return self.table.games + sum(self.table.excluded.values())

    @property
    def era_split(self) -> bool:
        return len(self.engine_versions) > 1


def round_directories(
    games_dir: Path, *, include_root: bool = True
) -> list[tuple[str, Path]]:
    """
    `(round name, directory)` for every round.

    `include_root=False` drops the loose `data/games/*.json` bucket. Those files
    carry a `round` field that no directory backs, so they are not a round and
    cannot be fitted as one — see `rounds.fit_rounds`, the one caller that
    passes it.
    """
    rounds: list[tuple[str, Path]] = []
    if not games_dir.exists():
        return rounds
    if include_root and any(
        p.is_file() and p.name != "manifest.json" for p in games_dir.glob("*.json")
    ):
        rounds.append((ROOT_ROUND, games_dir))
    for child in sorted(games_dir.iterdir()):
        if child.is_dir():
            rounds.append((child.name, child))
    return rounds


def round_signature(directory: Path) -> str:
    """A digest of the round's files: name, size, mtime. No game is read."""
    sha = hashlib.sha256()
    for path in sorted(list_game_paths(directory)):
        stat = path.stat()
        sha.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return f"sha256:{sha.hexdigest()}"


def rules_key(policy: Policy, registry: Registry | None) -> str:
    """
    A digest of everything except the games that could change the answer.

    The registry is in here because `require_registered` reads it: registering
    a hash makes games that were excluded eligible, with no file touched.
    """
    sha = hashlib.sha256()
    sha.update(json.dumps(policy.to_dict(), sort_keys=True).encode("utf-8"))
    if registry is not None and policy.require_registered:
        for bot_id in registry.bot_ids():
            entry = registry.load(bot_id)
            if entry is None:
                continue
            for version in sorted(v.content_hash for v in entry.versions):
                sha.update(f"{bot_id}@{version}\n".encode("utf-8"))
    return f"sha256:{sha.hexdigest()}"


def round_file_stem(round_name: str) -> str:
    """A round name as a filename component. Shared with `data/ratings/fits/`."""
    return round_name.replace("/", "_")


def _cache_path(cache_dir: Path, round_name: str) -> Path:
    return cache_dir / f"{round_file_stem(round_name)}.counts.json"


def prune_cache(cache_dir: Path, keep: Iterable[str]) -> list[Path]:
    """
    Delete cached tables for rounds that no longer exist under `data/games/`.

    A stale cache is harmless — its signature no longer matches anything that is
    read — but `data/ratings/cache/` had accumulated ~13 tables for deleted
    rounds, which reads as a round set that is not the round set.
    """
    if not cache_dir.exists():
        return []
    wanted = {f"{round_file_stem(name)}.counts.json" for name in keep}
    removed = []
    for path in sorted(cache_dir.glob("*.counts.json")):
        if path.name not in wanted:
            path.unlink()
            removed.append(path)
    return removed


def _read_cached(
    path: Path, *, round_name: str, signature: str, key: str
) -> RoundCounts | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        data.get("version") != CACHE_FORMAT_VERSION
        or data.get("signature") != signature
        or data.get("rules_key") != key
    ):
        return None
    return RoundCounts(
        round=round_name,
        table=CountTable.from_dict(data["table"]),
        engine_versions=tuple(data.get("engine_versions", ())),
    )


def _write_cached(
    path: Path, *, signature: str, key: str, counts: RoundCounts
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "round": counts.round,
        "signature": signature,
        "rules_key": key,
        "engine_versions": list(counts.engine_versions),
        "table": counts.table.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def round_count_tables(
    *,
    games_dir: Path | None = None,
    cache_dir: Path,
    policy: Policy,
    registry: Registry | None = None,
    use_cache: bool = True,
    include_root: bool = True,
) -> tuple[list[RoundCounts], CacheStats]:
    """
    One count table per round, in the order `round_directories` walks them.

    This is the whole read of `data/games/`. Both callers are built on it: the
    per-round fitter takes the list as it is, and `cached_count_table` merges
    it, so the pooled table is provably the same work.
    """
    directory = games_dir or GAMES_DIR
    key = rules_key(policy, registry)
    hits = misses = 0
    rounds: list[RoundCounts] = []

    for round_name, round_dir in round_directories(directory, include_root=include_root):
        signature = round_signature(round_dir)
        path = _cache_path(cache_dir, round_name)
        counts = (
            _read_cached(path, round_name=round_name, signature=signature, key=key)
            if use_cache
            else None
        )
        if counts is None:
            misses += 1
            games = [load_game(p) for p in list_game_paths(round_dir)]
            counts = RoundCounts(
                round=round_name,
                table=count_table(games, policy=policy, registry=registry),
                engine_versions=tuple(sorted({g.engine_version for g in games})),
            )
            if use_cache:
                _write_cached(path, signature=signature, key=key, counts=counts)
        else:
            hits += 1
        rounds.append(counts)

    return rounds, CacheStats(hits=hits, misses=misses)


def cached_count_table(
    *,
    games_dir: Path | None = None,
    cache_dir: Path,
    policy: Policy,
    registry: Registry | None = None,
    extra_entities: Iterable[str] = (),
    use_cache: bool = True,
) -> tuple[CountTable, CacheStats]:
    """
    Aggregate every round into one count table, reusing unchanged rounds.

    The result is identical to `count_table` over all games — the cache changes
    only how much work it takes, never the answer. `tests/test_ratings_cache.py`
    asserts exactly that.

    Nothing published reads this any more: ratings are fitted per round, and
    pooling rounds is what per-round fitting exists to stop. It stays because it
    is the definition the per-round split is checked against.
    """
    rounds, stats = round_count_tables(
        games_dir=games_dir,
        cache_dir=cache_dir,
        policy=policy,
        registry=registry,
        use_cache=use_cache,
    )
    tables = [counts.table for counts in rounds]
    merged = merge(tables) if tables else count_table([], policy=policy, registry=registry)
    if extra_entities:
        merged = _with_entities(merged, extra_entities)
    return merged, stats


def _with_entities(table: CountTable, extra: Iterable[str]) -> CountTable:
    from arena.records.ratings.counts import build

    return build(
        {(c.seat_a, c.seat_b): (c.wins_a, c.wins_b, c.draws) for c in table.cells},
        entities=[*table.entities, *extra],
        excluded=table.excluded,
    )


__all__ = [
    "CACHE_DIRNAME",
    "CacheStats",
    "ROOT_ROUND",
    "RoundCounts",
    "cached_count_table",
    "prune_cache",
    "round_count_tables",
    "round_directories",
    "round_file_stem",
    "round_signature",
    "rules_key",
]
