"""
The shell half of the submission minifier: does it only remove comments?

`tools/package_submission.py` strips whole-line `#` comments from the zip's
`run.sh`, `build.sh` and `.cargo/config.toml`, for the same reason the `.rs`
members go through `tools/minify`: a submission carries logic, not the
reasoning behind it. The risk is entirely on one side — a rule that removes too
much ships a launcher that does not launch, and the judge forfeits a game on a
bot that fails to start. So the cases below are mostly about what must survive.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "package_submission.py"

# Loaded by path because `tools/` is not a package — it is excluded from the
# bot's content hash, so it has no `__init__.py` and never will. The
# sys.modules registration is not optional: `@dataclass` resolves its own
# module by name while the class body executes, and a module that is not
# registered makes that lookup return None.
_spec = importlib.util.spec_from_file_location("morpheus_rs_package_submission", TOOL)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
strip_line_comments = _module.strip_line_comments


def test_the_shebang_survives():
    """Without it the judge runs the script under whatever it feels like."""
    out = strip_line_comments("#!/usr/bin/env bash\n# a note\nexec ./bot\n")
    assert out.splitlines()[0] == "#!/usr/bin/env bash"
    assert "a note" not in out


def test_whole_line_comments_go_indented_or_not():
    out = strip_line_comments("#!/bin/sh\n# top\n    # indented\ncd \"$DIR\"\n")
    assert out == '#!/bin/sh\ncd "$DIR"\n'


def test_a_hash_that_is_not_a_comment_is_left_alone():
    """
    The reason this is line-based. `${var#prefix}` is a parameter expansion and
    `"#"` is a string; a rule that hunted trailing comments would eat both, and
    the failure would be a bot that does not start.
    """
    source = '#!/bin/sh\nX="${DIR#/}"   # trailing note\necho "#1"\n'
    out = strip_line_comments(source)
    assert '${DIR#/}' in out
    assert 'echo "#1"' in out


def test_blank_runs_left_by_removed_blocks_collapse_to_one():
    """Removing a comment block should not leave the hole it came out of."""
    out = strip_line_comments("#!/bin/sh\n\n# one\n# two\n\n\nexec ./bot\n")
    assert out == "#!/bin/sh\n\nexec ./bot\n"


def test_the_shipped_launchers_still_do_what_they_did():
    """
    Not a syntax check — the packaging smoke runs these for real. This is the
    cheap half: every line that makes the launcher a launcher is still there.
    """
    for text in (_module.RUN_SH_VENDORED, _module.RUN_SH_STATIC):
        out = strip_line_comments(text)
        assert out.startswith("#!/usr/bin/env bash\n")
        assert "set -euo pipefail" in out
        assert out.count("export ") == 6
        assert "exec " in out
        assert "#" not in out.split("\n", 1)[1]
