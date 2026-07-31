"""
Repo-side remote play session with honest result logging.

Uses ``generals_client`` (EIO v4, no bot_key) instead of upstream
``GeneralsIOClient``. Disconnect and malformed frames never count as wins.

See ``docs/research/strategies/human-95-plan.md`` §5.4.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from generals_client.bot import BotError, GAME_TIMEOUT_SECONDS, GameResult
from generals_client.transport import DEFAULT_SERVER

from arena.remote_bridge import (
    ArenaGameClient,
    UnifiedBot,
    opponent_stars,
    opponent_username,
    register_username_safe,
)
from arena.paths import REPO_ROOT
from arena.store import git_head_sha, utc_now_iso

DECIDED_REASONS = frozenset({"game_won", "game_lost"})
QUEUE_TIMEOUT_DETAIL = "did not finish within"
DEFAULT_REQUEUE_BACKOFF_S = 30.0
MAX_REQUEUE_BACKOFF_S = 300.0


@dataclass
class SessionSummary:
    """Aggregate stats for one remote_play session."""

    bot_id: str
    room_mode: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    queue_timeouts: int = 0
    not_counted: int = 0
    duration_s: float = 0.0
    endpoint: str = ""
    stop_reason: str = "completed"
    queue_attempts: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "room_mode": self.room_mode,
            "endpoint": self.endpoint,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "queue_timeouts": self.queue_timeouts,
            "not_counted": self.not_counted,
            "duration_s": round(self.duration_s, 2),
            "stop_reason": self.stop_reason,
            "queue_attempts": self.queue_attempts,
        }


def is_queue_timeout(detail: str | None) -> bool:
    """Return True when a BotError detail indicates queue/game timeout."""
    return bool(detail and QUEUE_TIMEOUT_DETAIL in detail.lower())


def git_head() -> str | None:
    """Full HEAD SHA for remote game logs (None when git is unavailable)."""
    return git_head_sha(short=False, repo_root=REPO_ROOT)


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
        game_timeout: float | None = None,
    ):
        self.bot = bot
        self.bot_name = bot_name
        self.room_mode = room_mode
        self.log_dir = log_dir
        self.server_url = server_url
        self.client = ArenaGameClient(bot, user_id, server_url=server_url)
        self.client.game_timeout = (
            game_timeout if game_timeout is not None else GAME_TIMEOUT_SECONDS
        )
        self._score_wins = 0
        self._score_losses = 0
        self._last_result_reason: str | None = None
        self._last_finish_detail: str | None = None
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def wins(self) -> int:
        """Games won this session (decided games only)."""
        return self._score_wins

    @property
    def losses(self) -> int:
        """Games lost this session (decided games only)."""
        return self._score_losses

    @property
    def endpoint(self) -> str:
        host = urlparse(self.server_url).netloc or self.server_url
        return host.replace("https://", "").replace("http://", "")

    def register(self, username: str) -> None:
        register_username_safe(self.client, username)

    def play_private(self, game_id: str) -> str:
        return self._run_game(lambda: self.client.play_private(game_id))

    def play_1v1(self) -> str:
        return self._run_game(self.client.play_1v1)

    @property
    def last_finish_detail(self) -> str | None:
        return self._last_finish_detail

    def _run_game(self, play_fn) -> str:
        try:
            result = play_fn()
            reason = "game_won" if result.won else "game_lost"
            self._finish_with_reason(reason, result=result)
            return reason
        except BotError as exc:
            detail = str(exc)
            reason = "disconnect" if "disconnect" in detail.lower() else "receive_error"
            self._finish_with_reason(reason, detail=detail)
            return reason

    def _finish_with_reason(
        self,
        result_reason: str,
        *,
        result: GameResult | None = None,
        detail: str | None = None,
    ) -> None:
        self._last_result_reason = result_reason
        self._last_finish_detail = detail
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
            "recorded_at": utc_now_iso(),
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


def run_lobby_session(
    session: FidelityRemoteSession,
    lobby_id: str,
    username: str,
    max_games: int | None,
) -> SessionSummary:
    """Play private lobby games until max_games or unlimited."""
    summary = SessionSummary(
        bot_id=session.bot_name,
        room_mode=session.room_mode,
        endpoint=session.endpoint,
    )
    started = time.time()
    session.register(username)
    games_played = 0
    while max_games is None or games_played < max_games:
        reason = session.play_private(lobby_id)
        games_played += 1
        summary.games_played += 1
        _tally_session_reason(summary, reason, session.last_finish_detail)
        if max_games is not None and games_played >= max_games:
            break
    summary.duration_s = time.time() - started
    summary.wins = session.wins
    summary.losses = session.losses
    return summary


def run_1v1_session(
    session: FidelityRemoteSession,
    username: str,
    max_games: int,
    *,
    queue_timeout_seconds: float,
    session_deadline: float | None = None,
    requeue_backoff_s: float = DEFAULT_REQUEUE_BACKOFF_S,
) -> SessionSummary:
    """Play 1v1 queue games with requeue backoff on queue timeout."""
    summary = SessionSummary(
        bot_id=session.bot_name,
        room_mode=session.room_mode,
        endpoint=session.endpoint,
    )
    started = time.time()
    session.register(username)
    session.client.game_timeout = queue_timeout_seconds
    backoff = requeue_backoff_s

    while summary.games_played < max_games:
        if session_deadline is not None and time.time() >= session_deadline:
            summary.stop_reason = "session_minutes"
            break

        summary.queue_attempts += 1
        reason = session.play_1v1()
        detail = session.last_finish_detail

        if is_queue_timeout(detail):
            summary.queue_timeouts += 1
            if session_deadline is not None and time.time() + backoff >= session_deadline:
                summary.stop_reason = "session_minutes"
                break
            print(
                f"Queue timeout after {queue_timeout_seconds:.0f}s; "
                f"requeue in {backoff:.0f}s...",
                flush=True,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2.0, MAX_REQUEUE_BACKOFF_S)
            continue

        backoff = requeue_backoff_s
        summary.games_played += 1
        _tally_session_reason(summary, reason, detail)

    summary.duration_s = time.time() - started
    summary.wins = session.wins
    summary.losses = session.losses
    return summary


def _tally_session_reason(
    summary: SessionSummary,
    reason: str,
    detail: str | None,
) -> None:
    if reason in DECIDED_REASONS:
        return
    if is_queue_timeout(detail):
        summary.queue_timeouts += 1
    else:
        summary.not_counted += 1


def print_session_summary(summary: SessionSummary) -> None:
    """Print a one-screen session summary on exit."""
    print("\n--- Session summary ---")
    print(f"Bot: {summary.bot_id}  Mode: {summary.room_mode}  Endpoint: {summary.endpoint}")
    print(
        f"Games: {summary.games_played}  W-L: {summary.wins}-{summary.losses}  "
        f"Queue timeouts: {summary.queue_timeouts}  Not counted: {summary.not_counted}"
    )
    if summary.room_mode == "1v1":
        print(f"Queue attempts: {summary.queue_attempts}")
    print(f"Duration: {summary.duration_s:.0f}s  Stop: {summary.stop_reason}")
