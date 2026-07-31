"""
Repo-side remote play session with honest result logging.

Uses ``generals_client`` (EIO v4, no bot_key) instead of upstream
``GeneralsIOClient``. Disconnect and malformed frames never count as wins.

See ``docs/research/strategies/human-95-plan.md`` §5.4.
"""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from generals_client.bot import BotError, GameResult
from generals_client.transport import DEFAULT_SERVER

from arena.remote_bridge import (
    ArenaGameClient,
    UnifiedBot,
    ensure_bot_username,
    opponent_stars,
    opponent_username,
    register_username_safe,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

DECIDED_REASONS = frozenset({"game_won", "game_lost"})


def git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def normalize_bot_endpoint_username(username: str) -> str:
    """Backward-compatible alias: ensure ``[Bot]`` prefix for generals_client."""
    return ensure_bot_username(username)


def opponent_is_bot(username: str | None) -> bool | None:
    if username is None:
        return None
    return username.startswith("[Bot]")


def result_from_reason(result_reason: str) -> str:
    if result_reason == "game_won":
        return "win"
    if result_reason == "game_lost":
        return "loss"
    if result_reason == "disconnect":
        return "disconnect"
    if result_reason == "stall":
        return "error"
    return "error"


def _replay_id_from_url(replay_url: str) -> str | None:
    if not replay_url:
        return None
    path = urlparse(replay_url).path.rstrip("/")
    if not path:
        return None
    return path.rsplit("/", 1)[-1] or None


class FidelityRemoteSession:
    """One remote bot session: register, play games, log JSON under ``log_dir``."""

    def __init__(
        self,
        bot: UnifiedBot,
        user_id: str,
        *,
        bot_name: str,
        room_mode: str,
        log_dir: Path,
        server_url: str = DEFAULT_SERVER,
    ):
        self.bot = bot
        self.bot_name = bot_name
        self.room_mode = room_mode
        self.log_dir = log_dir
        self.server_url = server_url
        self.client = ArenaGameClient(bot, user_id, server_url=server_url)
        self._score_wins = 0
        self._score_losses = 0
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def endpoint(self) -> str:
        host = urlparse(self.server_url).netloc or self.server_url
        return host.replace("https://", "").replace("http://", "")

    def register(self, username: str) -> None:
        register_username_safe(self.client, username)

    def play_private(self, game_id: str) -> None:
        self._run_game(lambda: self.client.play_private(game_id))

    def play_1v1(self) -> None:
        self._run_game(self.client.play_1v1)

    def _run_game(self, play_fn) -> None:
        try:
            result = play_fn()
            reason = "game_won" if result.won else "game_lost"
            self._finish_with_reason(reason, result=result)
        except BotError as exc:
            detail = str(exc)
            reason = "disconnect" if "disconnect" in detail.lower() else "receive_error"
            self._finish_with_reason(reason, detail=detail)

    def _finish_with_reason(
        self,
        result_reason: str,
        *,
        result: GameResult | None = None,
        detail: str | None = None,
    ) -> None:
        outcome = result_from_reason(result_reason)
        counts_toward_block = result_reason in DECIDED_REASONS
        is_winner = result_reason == "game_won"

        replay_id = _replay_id_from_url(result.replay_url) if result else None
        replay_url = result.replay_url if result else None

        self._write_game_log(
            result=outcome,
            result_reason=result_reason,
            counts_toward_block=counts_toward_block,
            replay_id=replay_id,
            detail=detail,
        )

        if counts_toward_block:
            self._score_wins += int(is_winner)
            self._score_losses += int(not is_winner)

        if counts_toward_block:
            status = "Won!" if is_winner else "Lost."
        elif result_reason == "disconnect":
            status = "Disconnected (not counted)."
        else:
            status = f"Ended ({result_reason}, not counted)."

        link = replay_url or "unknown"
        print(
            f"You {status} Score {self._score_wins}:{self._score_losses}. "
            f"Replay link: {link}"
        )

    def _write_game_log(
        self,
        *,
        result: str,
        result_reason: str,
        counts_toward_block: bool,
        replay_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        stats = self.bot.session.session_stats()
        state = self.bot.last_state
        opp_name = opponent_username(state)
        stars = opponent_stars(state)

        record: dict[str, Any] = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "session_id": uuid.uuid4().hex[:12],
            "replay_id": replay_id,
            "bot_id": self.bot_name,
            "bot_commit": git_head(),
            "room_mode": self.room_mode,
            "endpoint": self.endpoint,
            "opponent_username": opp_name,
            "opponent_is_bot": opponent_is_bot(opp_name),
            "opponent_stars": stars,
            "result": result,
            "result_reason": result_reason,
            "counts_toward_block": counts_toward_block,
            "server_turns": stats["server_turns"],
            "peak_land": stats["peak_land"],
            "peak_army": stats["peak_army"],
            "final_land": stats["final_land"],
            "final_army": stats["final_army"],
            "saw_enemy_general_at": stats["saw_enemy_general_at"],
            "builds_dropped": stats["builds_dropped"],
            "faults": stats["faults"],
            "timeouts": stats["timeouts"],
        }
        if detail:
            record["finish_detail"] = detail

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.log_dir / f"{ts}_{self.bot_name}_{result_reason}.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        counted = "counted" if counts_toward_block else "discarded"
        try:
            display_path = path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = path
        print(f"Logged game ({counted}) to {display_path}")
