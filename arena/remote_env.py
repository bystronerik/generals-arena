"""Remote play environment: repo paths, dotenv load, credential checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from generals_client.transport import DEFAULT_SERVER

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_GAMES_DIR = REPO_ROOT / "data" / "remote_games"
SETUP_DOC = "docs/engine/remote-play-setup.md"
PUBLIC_SERVER_URL = DEFAULT_SERVER


def ensure_repo_on_path() -> None:
    """Insert repo root on ``sys.path`` when the script is run directly."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_dotenv_files() -> None:
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


def require_user_id() -> str:
    """Return GENERALS_USER_ID or exit with setup instructions."""
    user_id = os.environ.get("GENERALS_USER_ID", "").strip()
    if user_id:
        return user_id
    msg = (
        "GENERALS_USER_ID is not set.\n\n"
        "Live generals.io play needs a secret user id you invent (any long random string).\n"
        "It is not issued by generals.io; register_username binds a display name to it once.\n\n"
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


def default_username(bot: str) -> str:
    """Default generals.io username for a bot."""
    return os.environ.get("GENERALS_USERNAME", f"[Bot] arena_{bot}").strip()


def default_lobby_id() -> str:
    """Default private lobby id."""
    return os.environ.get("GENERALS_LOBBY_ID", "arena-test").strip()


def resolve_server_url(*, server_url: str | None = None) -> str:
    """Resolve the bot server URL, defaulting to the public bot server."""
    if server_url:
        return server_url.strip()
    return PUBLIC_SERVER_URL
