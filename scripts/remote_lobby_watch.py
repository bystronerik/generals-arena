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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.remote.adapter import verify_adapter_offline
from arena.remote.block import count_human_block_games
from arena.remote.bridge import make_unified_bot
from arena.remote.client import FidelityRemoteSession, run_lobby_session
from arena.remote.env import (
    REMOTE_GAMES_DIR,
    REPO_ROOT,
    apply_env_file,
    default_username,
    load_dotenv_files,
    parse_env_file,
    require_user_id,
    resolve_server_url,
)

ENV_AGENT = REPO_ROOT / ".env.agent"
SECRET_KEYS = frozenset(
    {"GENERALS_USER_ID", "GENERALS_BOT_KEY", "GENERALS_LOBBY_ID", "GENERALS_USERNAME"}
)


def _lobby_id_from_agent_file() -> str | None:
    raw = parse_env_file(ENV_AGENT).get("GENERALS_LOBBY_ID", "").strip()
    return raw or None


def _counted_human_games_since(since: float) -> int:
    return count_human_block_games(REMOTE_GAMES_DIR, since_mtime=since)


def _lobby_cleared_while_running(initial_lobby_id: str) -> bool:
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
        server_url: str,
    ) -> None:
        self.bot = bot
        self.max_games = max_games
        self.lobby_id = lobby_id
        self.poll_interval = poll_interval
        self.server_url = server_url
        self.started_at = time.time()

    def counted_this_session(self) -> int:
        return _counted_human_games_since(self.started_at)

    def remaining(self) -> int:
        return max(0, self.max_games - self.counted_this_session())

    def run(self) -> int:
        user_id = require_user_id()
        username = default_username(self.bot)
        session = FidelityRemoteSession(
            make_unified_bot(self.bot),
            user_id,
            bot_name=self.bot,
            room_mode="lobby",
            log_dir=REMOTE_GAMES_DIR,
            server_url=self.server_url,
        )

        print(
            f"Lobby watch: bot={self.bot!r}, target={self.max_games} counted human games.",
            flush=True,
        )
        print("Lobby id detected in .env.agent (value not printed).", flush=True)

        games_joined = 0
        while self.remaining() > 0:
            if _lobby_cleared_while_running(self.lobby_id):
                print("GENERALS_LOBBY_ID cleared or changed; stopping.", flush=True)
                break

            before = self.counted_this_session()
            try:
                run_lobby_session(
                    session,
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

            if _lobby_cleared_while_running(self.lobby_id):
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
    server_url: str,
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
            apply_env_file(ENV_AGENT)
            load_dotenv_files()
            return LobbyWatchSession(
                bot=bot,
                max_games=max_games,
                lobby_id=lobby_id,
                poll_interval=poll_interval,
                server_url=server_url,
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
    load_dotenv_files()

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
        "--server-url",
        default=None,
        help="Bot server URL (default: https://botws.generals.io)",
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

    server_url = resolve_server_url(server_url=args.server_url)

    return watch_for_lobby(
        bot=args.bot,
        max_games=args.max_games,
        poll_interval=args.poll_interval,
        max_watch_s=max_watch_s,
        server_url=server_url,
    )


if __name__ == "__main__":
    raise SystemExit(main())
