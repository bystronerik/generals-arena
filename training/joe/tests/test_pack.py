"""Code-tarball packing for vast.ai delivery (cheap default suite).

Builds a fake checkout tree — not this repo — so the test stays stdlib-only
and does not add measurable time to the 15 s budget.
"""

from __future__ import annotations

import tarfile

from training.joe.pack import (
    excluded,
    extract_checkout,
    iter_pack_files,
    pack_checkout,
)


def _fake_repo(tmp_path):
    (tmp_path / "competition-module" / "generals").mkdir(parents=True)
    (tmp_path / "competition-module" / "generals" / "__init__.py").write_text(
        "x = 1\n")
    (tmp_path / "competition-module" / ".git").mkdir()
    (tmp_path / "competition-module" / ".git" / "config").write_text("secret")
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "__init__.py").write_text("# pkg\n")
    joe = tmp_path / "training" / "joe"
    joe.mkdir()
    (joe / "vast_boot.py").write_text("print('boot')\n")
    (joe / "__pycache__").mkdir()
    (joe / "__pycache__" / "vast_boot.cpython-312.pyc").write_bytes(b"\0")
    (joe / "tests").mkdir()
    (joe / "tests" / "test_secret.py").write_text("assert False\n")
    return tmp_path


def test_excluded_drops_git_pycache_and_joe_tests():
    assert excluded("competition-module/.git/config")
    assert excluded("training/joe/__pycache__/x.pyc")
    assert excluded("training/joe/tests/test_secret.py")
    assert not excluded("training/joe/vast_boot.py")
    assert not excluded("competition-module/generals/__init__.py")


def test_pack_and_extract_round_trip(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    tar_path = tmp_path / "code.tar.gz"
    meta = pack_checkout(repo, tar_path, git_sha="abc123")
    assert tar_path.is_file()
    assert meta["git_sha"] == "abc123"
    assert meta["n_files"] >= 3
    assert meta["sha256"]
    names = tarfile.open(tar_path, "r:gz").getnames()
    assert "training/joe/vast_boot.py" in names
    assert "training/__init__.py" in names
    assert "competition-module/generals/__init__.py" in names
    assert not any(".git" in n.split("/") for n in names)
    assert not any(n.startswith("training/joe/tests") for n in names)
    assert not any("__pycache__" in n.split("/") for n in names)

    dest = tmp_path / "out"
    extract_checkout(tar_path, dest)
    assert (dest / "training" / "joe" / "vast_boot.py").read_text() == \
        "print('boot')\n"
    assert not (dest / "training" / "joe" / "tests").exists()
    assert not (dest / "competition-module" / ".git").exists()


def test_iter_pack_files_skips_missing_tests_dir(tmp_path):
    repo = _fake_repo(tmp_path)
    rels = {rel for _abs, rel in iter_pack_files(repo)}
    assert "training/joe/tests/test_secret.py" not in rels
    assert "training/joe/vast_boot.py" in rels
