"""Tests for arena.remote_env helpers."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from arena.remote_env import (
    PUBLIC_SERVER_URL,
    apply_env_file,
    default_username,
    load_dotenv_files,
    parse_env_file,
    require_user_id,
    resolve_server_url,
)


def test_resolve_server_url_default():
    assert resolve_server_url() == PUBLIC_SERVER_URL


def test_resolve_server_url_explicit():
    assert resolve_server_url(server_url="https://example.test") == "https://example.test"


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


def test_parse_env_file_reads_pairs_without_mutating_environ(monkeypatch, tmp_path: Path):
    """The lobby watcher needs "parse without applying"; D18 shared this."""
    env_file = tmp_path / ".env.agent"
    env_file.write_text(
        "# comment\n\nGENERALS_LOBBY_ID='lobby-1'\nQUOTED=\"v\"\nnot_a_pair\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GENERALS_LOBBY_ID", raising=False)
    assert parse_env_file(env_file) == {"GENERALS_LOBBY_ID": "lobby-1", "QUOTED": "v"}
    assert "GENERALS_LOBBY_ID" not in os.environ


def test_parse_env_file_missing_returns_empty(tmp_path: Path):
    assert parse_env_file(tmp_path / "nope.env") == {}


def test_apply_env_file_does_not_overwrite(monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("A_KEY=from_file\nB_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("A_KEY", "from_shell")
    monkeypatch.delenv("B_KEY", raising=False)
    apply_env_file(env_file)
    assert os.environ["A_KEY"] == "from_shell"
    assert os.environ["B_KEY"] == "from_file"
