"""Load and validate the Morpheus bootstrap panel config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.paths import REPO_ROOT
from arena.records.fingerprint import bot_content_hash
from arena.records.store import engine_version

DEFAULT_PANEL_PATH = (
    REPO_ROOT / "scripts" / "configs" / "morpheus" / "bootstrap-panel.json"
)

REQUIRED_ROLES = frozenset({"anchor", "heuristic", "research"})
EXCLUDED_BOT_IDS = frozenset({"classic_duel"})
BOARD_SIDE_MIN = 18
BOARD_SIDE_MAX = 21


class PanelError(ValueError):
    """The panel config is incomplete or disagrees with the checkout."""


def load_panel(path: Path | None = None) -> dict[str, Any]:
    """Read a panel JSON and enforce the required schema fields."""
    panel_path = (path or DEFAULT_PANEL_PATH).resolve()
    if not panel_path.is_file():
        raise PanelError(f"panel config missing: {panel_path}")
    data = json.loads(panel_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PanelError(f"panel root must be an object: {panel_path}")
    _validate_schema(data, panel_path)
    return data


def _validate_schema(data: dict[str, Any], panel_path: Path) -> None:
    for key in (
        "name",
        "selection_date",
        "rating_era",
        "round_seed",
        "seat_policy",
        "games_per_pair",
        "source_label",
        "members",
    ):
        if key not in data:
            raise PanelError(f"{panel_path}: missing field {key!r}")

    members = data["members"]
    if not isinstance(members, list) or len(members) < 5:
        raise PanelError(
            f"{panel_path}: members must list at least 5 bots (got {len(members) if isinstance(members, list) else type(members)})"
        )

    roles: set[str] = set()
    seen_ids: set[str] = set()
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise PanelError(f"{panel_path}: members[{index}] must be an object")
        for key in ("bot_id", "content_hash", "role", "rating", "decisive_games"):
            if key not in member:
                raise PanelError(
                    f"{panel_path}: members[{index}] missing field {key!r}"
                )
        bot_id = str(member["bot_id"])
        if bot_id in EXCLUDED_BOT_IDS:
            raise PanelError(f"{panel_path}: excluded bot {bot_id!r} cannot join the panel")
        if bot_id in seen_ids:
            raise PanelError(f"{panel_path}: duplicate bot_id {bot_id!r}")
        seen_ids.add(bot_id)
        role = str(member["role"])
        if role not in REQUIRED_ROLES:
            raise PanelError(
                f"{panel_path}: members[{index}] role {role!r} not in {sorted(REQUIRED_ROLES)}"
            )
        roles.add(role)
        if int(member["decisive_games"]) < 0:
            raise PanelError(f"{panel_path}: members[{index}] decisive_games must be >= 0")

    missing_roles = REQUIRED_ROLES - roles
    if missing_roles:
        raise PanelError(
            f"{panel_path}: panel missing required roles {sorted(missing_roles)}"
        )

    if str(data["seat_policy"]) != "alternate":
        raise PanelError(
            f"{panel_path}: seat_policy must be 'alternate' for a measurement panel"
        )
    if int(data["games_per_pair"]) < 1:
        raise PanelError(f"{panel_path}: games_per_pair must be >= 1")


def panel_run_scripts(panel: dict[str, Any], *, bots_dir: Path | None = None) -> list[Path]:
    """Resolve each member to `bots/<id>/run.sh` in panel order."""
    root = bots_dir or (REPO_ROOT / "bots")
    scripts: list[Path] = []
    for member in panel["members"]:
        bot_id = str(member["bot_id"])
        run_sh = root / bot_id / "run.sh"
        if not run_sh.is_file():
            raise PanelError(f"panel member {bot_id!r} has no run.sh at {run_sh}")
        scripts.append(run_sh.resolve())
    return scripts


def validate_panel_against_checkout(
    panel: dict[str, Any],
    *,
    bots_dir: Path | None = None,
    engine: str | None = None,
) -> list[str]:
    """
    Return human-readable problems when hashes or era disagree with this tree.

    An empty list means the panel is ready to record.
    """
    problems: list[str] = []
    current_engine = engine or engine_version()
    if str(panel["rating_era"]) != current_engine:
        problems.append(
            f"rating_era {panel['rating_era']!r} != checkout engine {current_engine!r}"
        )

    root = bots_dir or (REPO_ROOT / "bots")
    for member in panel["members"]:
        bot_id = str(member["bot_id"])
        expected = str(member["content_hash"])
        run_sh = root / bot_id / "run.sh"
        if not run_sh.is_file():
            problems.append(f"{bot_id}: missing {run_sh}")
            continue
        got = bot_content_hash(run_sh)
        if got != expected:
            problems.append(
                f"{bot_id}: panel content_hash {expected} != checkout {got}"
            )
    return problems


def member_by_id(panel: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(m["bot_id"]): m for m in panel["members"]}
