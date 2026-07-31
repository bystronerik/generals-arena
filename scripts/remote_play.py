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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.remote_adapter import (
    REMOTE_RECOMMENDED_BOTS,
    list_remote_bots,
    verify_adapter_offline,
)
from arena.remote_bridge import ensure_bot_username, make_unified_bot
from arena.remote_client import (
    FidelityRemoteSession,
    SessionSummary,
    git_head,
    print_session_summary,
    run_1v1_session,
    run_lobby_session,
)
from arena.remote_env import (
    REMOTE_GAMES_DIR,
    REPO_ROOT,
    SETUP_DOC,
    default_lobby_id,
    default_username,
    load_dotenv_files,
    require_user_id,
    resolve_server_url,
)

DEFAULT_QUEUE_TIMEOUT_S = 600.0


def _make_session(
    bot_name: str,
    user_id: str,
    room_mode: str,
    server_url: str,
    queue_timeout_seconds: float | None,
) -> FidelityRemoteSession:
    bot = make_unified_bot(bot_name)
    return FidelityRemoteSession(
        bot,
        user_id,
        bot_name=bot_name,
        room_mode=room_mode,
        log_dir=REMOTE_GAMES_DIR,
        server_url=server_url,
        game_timeout=queue_timeout_seconds,
    )


def run_dry_run(
    bot: str,
    *,
    username: str | None = None,
    lobby_id: str | None = None,
) -> int:
    print("Dry-run: offline adapter verification (no network).")
    registered_as = ensure_bot_username(username or _default_username(bot))
    lobby = lobby_id or _default_lobby_id()
    print(f"Live username would register as: {registered_as!r}")
    print(f"Lobby mode would use lobby_id: {lobby!r}")
    errors = verify_adapter_offline()
    if errors:
        print("Offline verification FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("Offline verification passed.")
    try:
        from arena.bot_api import StrategySession, from_game_state
        from generals_client.state import GameState

        session = StrategySession(bot)
        start = {
            "playerIndex": 0,
            "replay_id": "dryrun",
            "usernames": ["[Bot] test", "human"],
        }
        state = GameState(start)
        state.apply_update(
            {
                "turn": 1,
                "map_diff": [0, 9, 3, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "scores": [
                    {"i": 0, "tiles": 1, "total": 1},
                    {"i": 1, "tiles": 1, "total": 1},
                ],
                "generals": [4, -1],
            }
        )
        obs = from_game_state(state)
        action = session.act(obs)
        if len(action) != 5:
            raise RuntimeError(f"expected 5-int action, got {action!r}")
        print(f"Bot {bot!r} loads and acts on synthetic generals_client state.")
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


def _write_session_log(summary: SessionSummary, *, bot: str, started: float) -> None:
    REMOTE_GAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = REMOTE_GAMES_DIR / f"session_summary_{int(time.time())}.json"
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "bot_commit": git_head(),
        **summary.as_dict(),
    }
    if summary.duration_s == 0.0:
        payload["duration_s"] = round(time.time() - started, 2)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    load_dotenv_files()

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
        "--queue-timeout-seconds",
        type=float,
        default=DEFAULT_QUEUE_TIMEOUT_S,
        help=(
            "Per-attempt timeout for queue + game (default: 600). "
            "1v1 mode requeues with backoff when this expires."
        ),
    )
    parser.add_argument(
        "--session-minutes",
        type=float,
        default=None,
        help="Stop the whole session after this many minutes (1v1 requeue loop)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Bot server URL (default: https://botws.generals.io)",
    )
    parser.add_argument(
        "--public-server",
        action="store_true",
        help="Alias for the public bot server (https://botws.generals.io)",
    )
    parser.add_argument(
        "--verify-offline",
        action="store_true",
        help="Run offline adapter checks before connecting",
    )
    args = parser.parse_args(argv)

    server_url = resolve_server_url(
        server_url=args.server_url,
        public_server=args.public_server,
    )

    if args.bot not in REMOTE_RECOMMENDED_BOTS:
        print(
            f"Note: {args.bot!r} is not in the recommended remote set "
            f"{REMOTE_RECOMMENDED_BOTS}.",
            file=sys.stderr,
        )

    if args.mode == "dry-run":
        return run_dry_run(
            args.bot,
            username=args.username,
            lobby_id=args.lobby_id,
        )

    if args.verify_offline:
        errors = verify_adapter_offline()
        if errors:
            for err in errors:
                print(err, file=sys.stderr)
            return 1

    user_id = require_user_id()
    username = args.username or default_username(args.bot)
    registered_as = ensure_bot_username(username)
    if registered_as != username.strip():
        print(
            f"Note: will register as {registered_as!r} (generals_client requires [Bot] prefix).",
            file=sys.stderr,
        )
    elif not username.startswith("[Bot]"):
        print(
            "Warning: generals.io bot convention uses a [Bot] username prefix in env.",
            file=sys.stderr,
        )

    session = _make_session(
        args.bot,
        user_id,
        args.mode,
        server_url,
        args.queue_timeout_seconds,
    )
    started = time.time()
    session_deadline = None
    if args.session_minutes is not None:
        session_deadline = started + args.session_minutes * 60.0

    summary: SessionSummary | None = None
    exit_code = 0

    try:
        if args.mode == "lobby":
            lobby_id = args.lobby_id or default_lobby_id()
            print(
                f"Starting {args.bot} in private lobby {lobby_id!r} "
                f"as {username!r} on {server_url}..."
            )
            summary = run_lobby_session(
                session,
                lobby_id,
                username,
                args.max_games,
            )
        else:
            max_games = args.max_games if args.max_games is not None else 1
            deadline_note = ""
            if args.session_minutes is not None:
                deadline_note = f", session cap {args.session_minutes:.0f} min"
            print(
                f"Starting {args.bot} in 1v1 queue on {server_url} "
                f"as {username!r} ({max_games} game(s), "
                f"queue timeout {args.queue_timeout_seconds:.0f}s{deadline_note})..."
            )
            summary = run_1v1_session(
                session,
                username,
                max_games,
                queue_timeout_seconds=args.queue_timeout_seconds,
                session_deadline=session_deadline,
            )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        exit_code = 130
        if summary is None:
            summary = SessionSummary(
                bot_name=args.bot,
                room_mode=args.mode,
                endpoint=session.endpoint,
                wins=session._score_wins,
                losses=session._score_losses,
                stop_reason="interrupt",
            )
        summary.stop_reason = "interrupt"
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
    finally:
        if summary is not None:
            if summary.duration_s == 0.0:
                summary.duration_s = time.time() - started
            print_session_summary(summary)
            _write_session_log(summary, bot=args.bot, started=started)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
