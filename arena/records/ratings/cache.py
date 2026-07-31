"""
Per-round count caches, so a refit does not re-read every game ever played.

Count tables are exactly additive and their aggregation is order-independent,
which is what makes this safe: a cached round is not an approximation of the
work, it *is* the work, already done. Refitting after one new round then costs
O(#rounds) instead of O(#games).

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
CACHE_FORMAT_VERSION = 1

# Games sitting directly in data/games/ rather than in a round folder.
ROOT_ROUND = "_root"


@dataclass(frozen=True)
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def rounds(self) -> int:
        return self.hits + self.misses


def round_directories(games_dir: Path) -> list[tuple[str, Path]]:
    """`(round name, directory)` for every round, root files included."""
    rounds: list[tuple[str, Path]] = []
    if not games_dir.exists():
        return rounds
    if any(p.is_file() and p.name != "manifest.json" for p in games_dir.glob("*.json")):
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


def _cache_path(cache_dir: Path, round_name: str) -> Path:
    safe = round_name.replace("/", "_")
    return cache_dir / f"{safe}.counts.json"


def _read_cached(path: Path, *, signature: str, key: str) -> CountTable | None:
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
    return CountTable.from_dict(data["table"])


def _write_cached(
    path: Path, *, round_name: str, signature: str, key: str, table: CountTable
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "round": round_name,
        "signature": signature,
        "rules_key": key,
        "table": table.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    """
    directory = games_dir or GAMES_DIR
    key = rules_key(policy, registry)
    hits = misses = 0
    tables: list[CountTable] = []

    for round_name, round_dir in round_directories(directory):
        signature = round_signature(round_dir)
        path = _cache_path(cache_dir, round_name)
        table = _read_cached(path, signature=signature, key=key) if use_cache else None
        if table is None:
            misses += 1
            games = [load_game(p) for p in list_game_paths(round_dir)]
            table = count_table(games, policy=policy, registry=registry)
            if use_cache:
                _write_cached(
                    path, round_name=round_name, signature=signature, key=key, table=table
                )
        else:
            hits += 1
        tables.append(table)

    merged = merge(tables) if tables else count_table([], policy=policy, registry=registry)
    if extra_entities:
        merged = _with_entities(merged, extra_entities)
    return merged, CacheStats(hits=hits, misses=misses)


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
    "cached_count_table",
    "round_directories",
    "round_signature",
    "rules_key",
]
