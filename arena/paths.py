"""Repo root for arena modules.

Defined once here so it stays correct regardless of how deep a module sits in
the package. A module computing `Path(__file__).parent.parent` itself is right
only at the top level of `arena/`, and moving that file into a subpackage
silently retargets every derived data path at `arena/` instead of the repo.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
