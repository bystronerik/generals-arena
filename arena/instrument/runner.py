"""
Instrumented stdio entry point for one seat: `python -m arena.instrument.runner`.

Stands in for `bash bots/<name>/run.sh` on recorded matches. It speaks the same
protocol, from the same parsing code in `bots/_common/wire.py`, driving the same
`Agent.act` — and additionally samples `bots/<name>/probe.py` after each move,
appending one jsonl line per turn to `--trace`.

Why a separate entry point rather than a flag inside `wire.py`: every file in a
bot's directory, plus `_common/wire.py`, is in the bot's source closure — which
is both its rating identity and its submission bundle. Instrumentation living
there would ship to the judge and would fork every bot's hash on every edit.
Here it ships nowhere and forks nothing.

The equivalence that matters is behavioural, not textual: driven over the same
observations, this runner must emit a byte-identical action stream to the clean
entry point. Nothing here touches the agent — the probe only reads.

    python -m arena.instrument.runner bots/metro --trace /tmp/seat-a.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from arena.instrument.probes import load_probe, probe_extras
from arena.paths import REPO_ROOT
from arena.records.telemetry_schema import validate_extras

BOTS_DIR = REPO_ROOT / "bots"


def _put_bot_on_path(bot_dir: Path) -> None:
    """
    Resolve imports exactly as the bot's own `main.py` does.

    `bots/` goes on `sys.path` so `_common.wire` resolves, then the bot's own
    directory ahead of it, so a bare `agent` means *this* bot's agent.
    """
    for path in (str(BOTS_DIR), str(bot_dir)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _load_agent_class(bot_dir: Path):
    """
    The bot's `Agent`, imported exactly as `bots/<name>/main.py` imports it.

    A plain `import agent` against the path set above, not a by-path load: this
    process drives one seat, so the bare name is unambiguous, and importing the
    way the bot itself does is what makes the two entry points equivalent.
    """
    return importlib.import_module("agent").Agent


def run_instrumented(bot_dir: Path, trace: Path | None) -> None:
    """Drive one seat over stdio, tracing the agent's internals per turn."""
    _put_bot_on_path(bot_dir)
    from _common.wire import _read_observation  # same parser the clean loop uses

    agent_class = _load_agent_class(bot_dir)
    probe = load_probe(bot_dir)

    stdin, stdout = sys.stdin, sys.stdout
    handshake = stdin.readline()
    if not handshake:
        return
    player_id, H, W = (int(x) for x in handshake.split())

    agent = agent_class(player_id=player_id, H=H, W=W)
    lines: list[str] = []
    turn = 0

    try:
        while True:
            first = stdin.readline()
            if not first:
                return

            obs = _read_observation(stdin, H, W, first)
            p, r, c, d, s = agent.act(obs)
            stdout.write(f"{p} {r} {c} {d} {s}\n")
            stdout.flush()
            turn += 1

            if probe is not None:
                # `turn` is the trajectory's 1-based step index, not
                # `obs.turn`: the observation is numbered before the step, so
                # using it would offset every trace by one against the engine
                # trajectory the reducers align it with.
                #
                # Sampled after the move, so the trace shows what the decision
                # left behind. Validated here as well as at record build, so an
                # undeclared key fails the match that emitted it.
                extras = validate_extras(probe_extras(probe, agent))
                lines.append(json.dumps({"t": turn, **extras}, separators=(",", ":")))
    finally:
        # Buffered to the end: a per-turn flush would put file io on the move
        # path, which is exactly what must not perturb the game.
        if trace is not None and lines:
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one seat's bot over stdio with per-turn probe tracing."
    )
    parser.add_argument("bot_dir", type=Path, help="path to bots/<name>/")
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="write one jsonl line per turn here (omit to run uninstrumented)",
    )
    args = parser.parse_args(argv)

    bot_dir = args.bot_dir.resolve()
    if not (bot_dir / "agent.py").is_file():
        parser.error(f"no agent.py under {bot_dir}")
    run_instrumented(bot_dir, args.trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
