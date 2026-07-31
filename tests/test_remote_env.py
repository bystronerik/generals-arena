"""Tests for arena.remote_env helpers."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from arena.remote_env import (
    PUBLIC_SERVER_URL,
    default_username,
    load_dotenv_files,
    require_user_id,
    resolve_server_url,
)


def test_resolve_server_url_default():
    assert resolve_server_url() == PUBLIC_SERVER_URL


def test_resolve_server_url_explicit():
    assert resolve_server_url(server_url="https://example.test") == "https://example.test"


def test_resolve_server_url_public_alias():
    assert resolve_server_url(public_server=True) == PUBLIC_SERVER_URL


def test_default_username_from_env(monkeypatch):
    monkeypatch.setenv("GENERALS_USERNAME", "[Bot] custom")
    assert default_username("smoke") == "[Bot] custom"


def test_default_username_fallback(monkeypatch):
    monkeypatch.delenv("GENERALS_USERNAME", raising=False)
    assert default_username("army_convey") == "[Bot] arena_army_convey"


def test_load_dotenv_files_does_not_overwrite(monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("GENERALS_USER_ID=from_file\n", encoding="utf-8")
    monkeypatch.setenv("GENERALS_USER_ID", "from_shell")
    monkeypatch.setattr("arena.remote_env.REPO_ROOT", tmp_path)
    load_dotenv_files()
    assert os.environ["GENERALS_USER_ID"] == "from_shell"


def test_require_user_id_exits_when_missing(monkeypatch):
    monkeypatch.delenv("GENERALS_USER_ID", raising=False)
    with pytest.raises(SystemExit) as exc:
        require_user_id()
    assert exc.value.code == 2


def test_require_user_id_returns_value(monkeypatch):
    monkeypatch.setenv("GENERALS_USER_ID", "secret-id")
    assert require_user_id() == "secret-id"
