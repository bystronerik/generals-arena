"""
Repo-side GeneralsIO client with honest result logging for remote evaluation.

The upstream ``GeneralsIOClient._play_game`` treats ``ValueError`` from
``receive()`` as a win. This module subclasses without editing the submodule.
See ``docs/research/strategies/human-95-plan.md`` §5.4.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generals.remote.generalsio_client import BOT_ENDPOINT, GeneralsIOClient

from arena.remote_adapter import StdioStrategyAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reasons that map to a decided game and may count toward the 95/100 block.
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
    """
    botws.generals.io rejects usernames starting with ``[Bot]``.

    Strip that prefix for registration; keep the rest unchanged.
    """
    name = username.strip()
    if name.startswith("[Bot]"):
        name = name[len("[Bot]") :].strip()
    return name or username.strip()


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


class FidelityGeneralsIOClient(GeneralsIOClient):
    """GeneralsIOClient that never records disconnects or malformed frames as wins."""

    def __init__(
        self,
        agent: StdioStrategyAdapter,
        user_id: str,
        *,
        bot_name: str,
        room_mode: str,
        log_dir: Path,
        public_server: bool = False,
    ):
        super().__init__(agent, user_id, public_server=public_server)
        bot_key = os.environ.get("GENERALS_BOT_KEY", "").strip()
        if bot_key:
            self.bot_key = bot_key
        self._arena_agent = agent
        self.bot_name = bot_name
        self.room_mode = room_mode
        self.log_dir = log_dir
        self._opponent_username: str | None = None
        self._last_finish_detail: str | None = None
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def endpoint(self) -> str:
        return "ws.generals.io" if self.public_server else "botws.generals.io"

    def register_agent(self, username: str) -> None:
        normalized = normalize_bot_endpoint_username(username)
        if normalized != username:
            print(
                "Note: bot endpoint registration strips the [Bot] prefix "
                f"(registering as {normalized!r}).",
            )
        try:
            super().register_agent(normalized)
        except ValueError as exc:
            message = str(exc)
            if "already have a username" in message.lower():
                print(
                    "Note: user id already has a bound username on generals.io; "
                    "continuing with existing registration.",
                )
                return
            raise

    def _initialize_game(self, data: dict) -> None:
        super()._initialize_game(data)
        self._arena_agent.reset()
        idx = self.game_state.opponent_index
        self._opponent_username = self.game_state.usernames[idx]

    def _play_game(self) -> None:
        """Game loop matching upstream terminal events; ignore benign server noise."""
        while True:
            try:
                received = self.receive()
            except ValueError as exc:
                self._finish_with_reason("disconnect", detail=str(exc))
                return

            if not isinstance(received, (list, tuple)) or len(received) == 0:
                self._finish_with_reason("receive_error", detail="empty receive payload")
                return

            event = received[0]
            match event:
                case "game_update":
                    if len(received) != 3:
                        self._finish_with_reason(
                            "receive_error",
                            detail=(
                                f"expected 3-tuple for game_update, got {len(received)}"
                            ),
                        )
                        return
                    _, data, _ = received
                    self.game_state.update(data)
                    obs = self.game_state.get_observation()
                    action = self._generate_action(obs)
                    if action:
                        self.emit("attack", action)
                case "game_won":
                    self._finish_with_reason("game_won")
                    return
                case "game_lost":
                    self._finish_with_reason("game_lost")
                    return
                case _:
                    # Upstream has no default case — chat_message and similar are skipped.
                    continue

    def _finish_with_reason(self, result_reason: str, *, detail: str | None = None) -> None:
        self._last_finish_detail = detail
        result = result_from_reason(result_reason)
        counts_toward_block = result_reason in DECIDED_REASONS
        is_winner = result_reason == "game_won"

        self._write_game_log(
            result=result,
            result_reason=result_reason,
            counts_toward_block=counts_toward_block,
            detail=detail,
        )

        self._status = "off"
        if counts_toward_block:
            self._score_wins += int(is_winner)
            self._score_losses += int(not is_winner)

        if counts_toward_block:
            status = "Won!" if is_winner else "Lost."
        elif result_reason == "disconnect":
            status = "Disconnected (not counted)."
        else:
            status = f"Ended ({result_reason}, not counted)."

        prefix = "bot." if not self.public_server else ""
        replay = self._replay_id or "unknown"
        print(
            f"You {status} Score {self._score_wins}:{self._score_losses}. "
            f"Replay link: https://{prefix}generals.io/replays/{replay}"
        )
        self.emit("leave_game")

    def _finish_game(self, is_winner: bool) -> None:
        """Upstream hook; route through fidelity finish."""
        self._finish_with_reason("game_won" if is_winner else "game_lost")

    def _write_game_log(
        self,
        *,
        result: str,
        result_reason: str,
        counts_toward_block: bool,
        detail: str | None = None,
    ) -> None:
        stats = self._arena_agent.session_stats()
        opponent_stars: int | None = None
        if hasattr(self, "game_state") and self.game_state is not None:
            idx = self.game_state.opponent_index
            stars = getattr(self.game_state, "stars", None)
            if stars is not None and 0 <= idx < len(stars):
                opponent_stars = int(stars[idx])

        record: dict[str, Any] = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "session_id": uuid.uuid4().hex[:12],
            "replay_id": self._replay_id or None,
            "bot_id": self.bot_name,
            "bot_commit": git_head(),
            "room_mode": self.room_mode,
            "endpoint": self.endpoint,
            "opponent_username": self._opponent_username,
            "opponent_is_bot": opponent_is_bot(self._opponent_username),
            "opponent_stars": opponent_stars,
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
