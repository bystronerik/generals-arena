"""
One-shot projection of stored game records from schema v4 to v5.

v5 makes the required fields exactly the rating identity and the outcome and
moves every observation into `metrics` (docs/arena/game-record-schema.md).
There is deliberately no dual-version reader — `GameRecord.from_dict` rejects
v4 as loudly as v4 rejected v3 — so the stored pool is projected once, in
place, by this script.

The projection is pure field shuffling: it drops the fields nothing read,
drops the one field that was fully derivable, and moves the observational
fields into `metrics` under the same names. It never invents, reinterprets, or
deletes a *value*, so the rating fit over the migrated pool is identical (the
fit reads only identity and `winner`).

Committed even though it runs once: it is the documentation of what the
projection did to 10k stored games.

    python -m scripts.migrate_games_v5 --dry-run
    python -m scripts.migrate_games_v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.records.store import (  # noqa: E402
    CURRENT_SCHEMA_VERSION,
    GAMES_DIR,
    write_record_json,
)

# Zero readers across arena/ and scripts/ (recorder plan §1 A12). `game_id`
# embeds a UTC minute stamp, so dropping these loses only sub-minute
# provenance.
DROPPED = ("started_at", "finished_at", "duration_seconds")

# `terminated` is exactly `winner != "draw"` under the match loop's mapping, so
# a stored copy could only ever disagree with the outcome it restates. v5
# derives it (`GameRecord.terminated`).
DERIVED = ("terminated",)

# Observations, not identity or outcome: same names, now under `metrics`.
MOVED_TO_METRICS = (
    "castles_built_a",
    "castles_built_b",
    "final_land_a",
    "final_land_b",
    "final_army_a",
    "final_army_b",
)

# One v4 name meant two different measurements. The top-level
# `castles_built_a/_b` was the engine's tally of castle births; the *same* key
# inside `metrics` was metro's own `castles_built` counter, flattened out of its
# stderr telemetry. They disagree on 16 of the 1000 games that carry both, so
# neither may shadow the other. v5 gives the engine tally the plain name (it is
# ground truth) and the bot's belief an explicit one — matching the
# `castles_built_probe` key metro's probe now emits.
PROBE_SHADOWED = {
    "castles_built_a": "castles_built_probe_a",
    "castles_built_b": "castles_built_probe_b",
}


class MigrationError(RuntimeError):
    """A record could not be projected onto v5."""


def project_v5(record: dict[str, Any]) -> dict[str, Any]:
    """Return the v5 form of one v4 record dict. Pure; does not mutate input."""
    version = int(record.get("schema_version", 1))
    if version >= CURRENT_SCHEMA_VERSION:
        return dict(record)
    if version != 4:
        raise MigrationError(
            f"only v4 records can be projected onto v{CURRENT_SCHEMA_VERSION} "
            f"(got v{version}); no pre-v4 record survives"
        )

    out = dict(record)
    metrics = dict(out.get("metrics") or {})

    for old_key, new_key in PROBE_SHADOWED.items():
        if old_key in metrics:
            if new_key in metrics:
                raise MigrationError(
                    f"cannot rename {old_key} to {new_key}: both are present"
                )
            metrics[new_key] = metrics.pop(old_key)

    for key in MOVED_TO_METRICS:
        value = out.pop(key, None)
        if value is not None:
            if key in metrics:
                raise MigrationError(
                    f"metrics already carries {key}; moving the top-level field "
                    f"would silently overwrite a different measurement"
                )
            metrics[key] = value

    for key in DROPPED + DERIVED:
        out.pop(key, None)

    out["metrics"] = metrics
    out["schema_version"] = CURRENT_SCHEMA_VERSION
    return out


def migrate_file(path: Path, *, dry_run: bool) -> bool:
    """Project one record file in place. Returns True when it changed."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise MigrationError(f"record must be an object: {path}")

    projected = project_v5(record)
    if projected == record:
        return False
    if not dry_run:
        write_record_json(projected, path)
    return True


def game_record_paths(games_root: Path) -> list[Path]:
    """Every stored game JSON under `games_root`, manifests excluded."""
    return sorted(
        p
        for p in games_root.rglob("*.json")
        if p.is_file() and p.name != "manifest.json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project stored game records from schema v4 to v5, in place."
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=GAMES_DIR,
        help=f"root to walk recursively (default: {GAMES_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing",
    )
    args = parser.parse_args(argv)

    paths = game_record_paths(args.games_dir)
    if not paths:
        print(f"[migrate_v5] no records under {args.games_dir}")
        return 0

    changed = 0
    for path in paths:
        if migrate_file(path, dry_run=args.dry_run):
            changed += 1

    verb = "would migrate" if args.dry_run else "migrated"
    print(
        f"[migrate_v5] {verb} {changed} of {len(paths)} record(s) under "
        f"{args.games_dir} (already-v5 records untouched)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
