#!/usr/bin/env python3
"""
Play a local heuristic bot on live generals.io (classic rules, not competition).

Requires GENERALS_USER_ID in the environment. See docs/engine/remote-play-setup.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.remote_adapter import (  # noqa: E402
    REMOTE_RECOMMENDED_BOTS,
    StdioStrategyAdapter,
    list_remote_bots,
    make_remote_agent,
    verify_adapter_offline,
)
from arena.remote_client import (  # noqa: E402
    FidelityGeneralsIOClient,
    git_head,
    normalize_bot_endpoint_username,
)

REMOTE_GAMES_DIR = REPO_ROOT / "data" / "remote_games"
SETUP_DOC = "docs/engine/remote-play-setup.md"


def _load_dotenv_files() -> None:
    """Load KEY=VALUE pairs from .env and .env.agent without overwriting existing env."""
    for name in (".env", ".env.agent"):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def _require_user_id() -> str:
    user_id = os.environ.get("GENERALS_USER_ID", "").strip()
    if user_id:
        return user_id
    msg = (
        "GENERALS_USER_ID is not set.\n\n"
        "Live generals.io play needs a secret user id you invent (any long random string).\n"
        "It is not issued by generals.io; register_agent binds a username to it once.\n\n"
        "Setup:\n"
        "  export GENERALS_USER_ID='<your long random secret>'\n"
        "  export GENERALS_USERNAME='[Bot] arena_army_convey'   # optional\n"
        "  export GENERALS_LOBBY_ID='arena-test'                # for --mode lobby\n\n"
        "Or put the same keys in .env.agent (gitignored).\n"
        f"Full guide: {SETUP_DOC}\n\n"
        "Run with --dry-run to offline-verify the adapter without credentials."
    )
    print(msg, file=sys.stderr)
    sys.exit(2)


def _default_username(bot: str) -> str:
    return os.environ.get("GENERALS_USERNAME", f"[Bot] arena_{bot}").strip()


def _default_lobby_id() -> str:
    return os.environ.get("GENERALS_LOBBY_ID", "arena-test").strip()


def run_lobby_session(
    agent: StdioStrategyAdapter,
    user_id: str,
    lobby_id: str,
    username: str,
    max_games: int | None,
) -> None:
    games_played = 0
    with FidelityGeneralsIOClient(
        agent,
        user_id,
        bot_name=agent.bot_name,
        room_mode="lobby",
        log_dir=REMOTE_GAMES_DIR,
    ) as client:
        client.register_agent(username)
        while max_games is None or games_played < max_games:
            if client.status == "off":
                client.join_private_lobby(lobby_id)
            if client.status == "lobby":
                client.join_game()
                games_played += 1
                if max_games is not None and games_played >= max_games:
                    break


def run_1v1_session(
    agent: StdioStrategyAdapter,
    user_id: str,
    username: str,
    max_games: int,
    *,
    public_server: bool = False,
) -> None:
    with FidelityGeneralsIOClient(
        agent,
        user_id,
        bot_name=agent.bot_name,
        room_mode="1v1",
        log_dir=REMOTE_GAMES_DIR,
        public_server=public_server,
    ) as client:
        client.register_agent(username)
        for _ in range(max_games):
            client.join_1v1_queue()


def run_dry_run(bot: str) -> int:
    print("Dry-run: offline adapter verification (no network).")
    errors = verify_adapter_offline()
    if errors:
        print("Offline verification FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("Offline verification passed.")
    try:
        agent = make_remote_agent(bot)
        obs_errors = verify_adapter_offline()
        if obs_errors:
            raise RuntimeError("; ".join(obs_errors))
        print(f"Bot {bot!r} loads and acts on synthetic observation.")
    except Exception as exc:
        print(f"Bot load failed: {exc}", file=sys.stderr)
        return 1

    blockers = []
    if not os.environ.get("GENERALS_USER_ID", "").strip():
        blockers.append("GENERALS_USER_ID is not set (required for live play)")
    if blockers:
        print("\nLive play blockers:")
        for item in blockers:
            print(f"  - {item}")
        print(f"\nSee {SETUP_DOC} for setup steps.")
    else:
        print("\nGENERALS_USER_ID is set; live play can proceed.")
    print(
        "\nNote: 95% win rate vs humans requires 100+ logged human games in "
        f"{REMOTE_GAMES_DIR.relative_to(REPO_ROOT)}/ — not claimed here."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_files()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bot",
        default="army_convey",
        choices=list_remote_bots(),
        help="bots/<name>/ to play (default: army_convey)",
    )
    parser.add_argument(
        "--mode",
        choices=("lobby", "1v1", "dry-run"),
        default="dry-run",
        help="lobby = private lobby; 1v1 = public queue; dry-run = offline verify only",
    )
    parser.add_argument(
        "--lobby-id",
        default=None,
        help="Private lobby id (default: GENERALS_LOBBY_ID or arena-test)",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Bot username on generals.io (default: GENERALS_USERNAME or [Bot] arena_<bot>)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Stop after N games (default: unlimited for lobby, 1 for 1v1)",
    )
    parser.add_argument(
        "--verify-offline",
        action="store_true",
        help="Run offline adapter checks before connecting",
    )
    parser.add_argument(
        "--public-server",
        action="store_true",
        help="Use ws.generals.io instead of botws.generals.io (default: bot endpoint)",
    )
    args = parser.parse_args(argv)

    if args.bot not in REMOTE_RECOMMENDED_BOTS:
        print(
            f"Note: {args.bot!r} is not in the recommended remote set "
            f"{REMOTE_RECOMMENDED_BOTS}.",
            file=sys.stderr,
        )

    if args.mode == "dry-run":
        return run_dry_run(args.bot)

    if args.verify_offline:
        errors = verify_adapter_offline()
        if errors:
            for err in errors:
                print(err, file=sys.stderr)
            return 1

    user_id = _require_user_id()
    username = args.username or _default_username(args.bot)
    registered_as = normalize_bot_endpoint_username(username)
    if registered_as != username:
        print(
            "Note: bot endpoint strips [Bot] prefix at registration "
            f"(will register as {registered_as!r}).",
            file=sys.stderr,
        )
    elif not username.startswith("[Bot]"):
        print(
            "Warning: generals.io bot convention uses a [Bot] username prefix in env.",
            file=sys.stderr,
        )

    agent = make_remote_agent(args.bot)
    started = time.time()

    try:
        if args.mode == "lobby":
            lobby_id = args.lobby_id or _default_lobby_id()
            print(f"Starting {agent.id} in private lobby {lobby_id!r} as {username!r}...")
            run_lobby_session(agent, user_id, lobby_id, username, args.max_games)
        else:
            max_games = args.max_games if args.max_games is not None else 1
            endpoint = "ws.generals.io" if args.public_server else "botws.generals.io"
            print(
                f"Starting {agent.id} in 1v1 queue on {endpoint} "
                f"as {username!r} ({max_games} game(s))..."
            )
            run_1v1_session(
                agent,
                user_id,
                username,
                max_games,
                public_server=args.public_server,
            )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except Exception:
        session_log = REMOTE_GAMES_DIR / f"session_error_{int(time.time())}.json"
        REMOTE_GAMES_DIR.mkdir(parents=True, exist_ok=True)
        session_log.write_text(
            json.dumps(
                {
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "bot_id": args.bot,
                    "bot_commit": git_head(),
                    "error": traceback.format_exc(),
                    "duration_s": round(time.time() - started, 2),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Session error logged to {session_log.relative_to(REPO_ROOT)}", file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
