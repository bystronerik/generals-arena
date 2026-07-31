#!/usr/bin/env python3
"""
Watch .env.agent for GENERALS_LOBBY_ID and run lobby games when it appears.

Polls the env file every 15 s. Starts lobby mode when a lobby id is set.
Exits when the target counted-game count is reached or the lobby id is cleared.

Never prints secret values (user id, bot key, lobby id).
See docs/engine/remote-play-setup.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from remote_play import (  # noqa: E402
    REMOTE_GAMES_DIR,
    _default_username,
    _load_dotenv_files,
    _require_user_id,
    run_lobby_session,
)
from arena.remote_adapter import verify_adapter_offline  # noqa: E402
from arena.remote_block import count_human_block_games  # noqa: E402
from arena.remote_bridge import make_unified_bot  # noqa: E402
from arena.remote_client import FidelityRemoteSession  # noqa: E402

ENV_AGENT = REPO_ROOT / ".env.agent"
SECRET_KEYS = frozenset(
    {"GENERALS_USER_ID", "GENERALS_BOT_KEY", "GENERALS_LOBBY_ID", "GENERALS_USERNAME"}
)


def _parse_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE pairs from one env file; do not mutate os.environ."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            out[key] = value
    return out


def _lobby_id_from_agent_file() -> str | None:
    raw = _parse_env_file(ENV_AGENT).get("GENERALS_LOBBY_ID", "").strip()
    return raw or None


def _apply_env_file(path: Path) -> None:
    """Load env file without overwriting variables already in the shell."""
    for key, value in _parse_env_file(path).items():
        if key not in os.environ:
            os.environ[key] = value


def _counted_human_games_since(since: float) -> int:
    return count_human_block_games(REMOTE_GAMES_DIR, since_mtime=since)


def _lobby_cleared_while_running(started_at: float, initial_lobby_id: str) -> bool:
    current = _lobby_id_from_agent_file()
    return current is None or current != initial_lobby_id


class LobbyWatchSession:
    """Run lobby games until counted target or lobby id cleared."""

    def __init__(
        self,
        *,
        bot: str,
        max_games: int,
        lobby_id: str,
        poll_interval: float,
    ) -> None:
        self.bot = bot
        self.max_games = max_games
        self.lobby_id = lobby_id
        self.poll_interval = poll_interval
        self.started_at = time.time()
        self.counted_at_start = _counted_human_games_since(self.started_at)

    def counted_this_session(self) -> int:
        return _counted_human_games_since(self.started_at)

    def remaining(self) -> int:
        return max(0, self.max_games - self.counted_this_session())

    def run(self) -> int:
        user_id = _require_user_id()
        username = _default_username(self.bot)
        session = FidelityRemoteSession(
            make_unified_bot(self.bot),
            user_id,
            bot_name=self.bot,
            room_mode="lobby",
            log_dir=REMOTE_GAMES_DIR,
        )

        print(
            f"Lobby watch: bot={self.bot!r}, target={self.max_games} counted human games.",
            flush=True,
        )
        print("Lobby id detected in .env.agent (value not printed).", flush=True)

        games_joined = 0
        while self.remaining() > 0:
            if _lobby_cleared_while_running(self.started_at, self.lobby_id):
                print("GENERALS_LOBBY_ID cleared or changed; stopping.", flush=True)
                break

            before = self.counted_this_session()
            try:
                run_lobby_session(
                    session,
                    user_id,
                    self.lobby_id,
                    username,
                    max_games=1,
                )
            except KeyboardInterrupt:
                print("\nStopped by user.", flush=True)
                return 130

            games_joined += 1
            after = self.counted_this_session()
            if after > before:
                print(
                    f"Counted human games this session: {after}/{self.max_games}",
                    flush=True,
                )
            else:
                print(
                    f"Game finished (not counted toward block). "
                    f"Counted this session: {after}/{self.max_games}",
                    flush=True,
                )

            if _lobby_cleared_while_running(self.started_at, self.lobby_id):
                print("GENERALS_LOBBY_ID cleared or changed; stopping.", flush=True)
                break

            if self.remaining() > 0:
                time.sleep(self.poll_interval)

        counted = self.counted_this_session()
        print(
            f"Lobby watch done: {counted} counted human games "
            f"(joined {games_joined} lobby game(s)).",
            flush=True,
        )
        return 0


def watch_for_lobby(
    *,
    bot: str,
    max_games: int,
    poll_interval: float,
    max_watch_s: float | None,
) -> int:
    watch_started = time.time()
    print(
        f"Watching {ENV_AGENT.relative_to(REPO_ROOT)} every {poll_interval:.0f}s "
        f"for GENERALS_LOBBY_ID (secret values are not printed).",
        flush=True,
    )

    while True:
        lobby_id = _lobby_id_from_agent_file()
        if lobby_id:
            _apply_env_file(ENV_AGENT)
            _load_dotenv_files()
            return LobbyWatchSession(
                bot=bot,
                max_games=max_games,
                lobby_id=lobby_id,
                poll_interval=poll_interval,
            ).run()

        elapsed = time.time() - watch_started
        if max_watch_s is not None and elapsed >= max_watch_s:
            print(
                f"No GENERALS_LOBBY_ID after {elapsed:.0f}s; exiting cleanly.",
                flush=True,
            )
            return 0

        remaining = ""
        if max_watch_s is not None:
            remaining = f" ({max(0.0, max_watch_s - elapsed):.0f}s left)"
        print(
            f"{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z — "
            f"waiting for GENERALS_LOBBY_ID{remaining}",
            flush=True,
        )
        time.sleep(poll_interval)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_files()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bot",
        default="classic_duel",
        help="Bot to play in lobby mode (default: classic_duel)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=20,
        help="Stop after this many counted human games (default: 20)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=15.0,
        help="Seconds between .env.agent polls (default: 15)",
    )
    parser.add_argument(
        "--max-watch-minutes",
        type=float,
        default=None,
        help="Exit if no lobby id appears within this many minutes",
    )
    parser.add_argument(
        "--verify-offline",
        action="store_true",
        help="Run offline adapter checks before watching",
    )
    args = parser.parse_args(argv)

    if args.verify_offline:
        errors = verify_adapter_offline()
        if errors:
            for err in errors:
                print(err, file=sys.stderr)
            return 1

    max_watch_s = None
    if args.max_watch_minutes is not None:
        max_watch_s = args.max_watch_minutes * 60.0

    return watch_for_lobby(
        bot=args.bot,
        max_games=args.max_games,
        poll_interval=args.poll_interval,
        max_watch_s=max_watch_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
