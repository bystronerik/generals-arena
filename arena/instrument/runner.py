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

It can additionally run a **capture module** (`arena.instrument.capture`), which
writes whole arrays per turn rather than declared scalars — the corpus the
morpheus-rs parity harness replays against. Capture is armed either per-run
with `--capture`, or across a whole recorded match by exporting
`ARENA_CAPTURE_MODULE`, in which case each seat's output path is derived from
its own trace path.

The equivalence that matters is behavioural, not textual: driven over the same
observations, this runner must emit a byte-identical action stream to the clean
entry point. The probe only reads. A capture module may wrap agent internals —
the RNG proxy does — but only to observe them: a wrapper that changed a drawn
value would invalidate the very corpus it exists to produce.

    python -m arena.instrument.runner bots/metro --trace /tmp/seat-a.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from arena.instrument.capture import (
    CAPTURE_MODULE_ENV,
    CAPTURE_SUFFIX,
    CaptureSink,
    capture_close,
    capture_frame,
    capture_install,
    capture_module_from_env,
    load_capture,
)
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


def _capture_target(
    capture_module: Path | None, trace: Path | None, capture_out: Path | None
) -> tuple[Path, Path] | None:
    """
    Resolve `(module_path, out_path)` for capture, or None when it is off.

    The output path is derived from this seat's trace path unless given
    explicitly, which is what lets one env var arm both seats of a recorded
    match without them writing over each other.
    """
    if capture_module is None:
        return None
    out = capture_out
    if out is None:
        if trace is None:
            return None
        out = trace.with_name(trace.name + CAPTURE_SUFFIX)
    return capture_module, out


def run_instrumented(
    bot_dir: Path,
    trace: Path | None,
    *,
    capture_module: Path | None = None,
    capture_out: Path | None = None,
) -> None:
    """Drive one seat over stdio, tracing the agent's internals per turn."""
    _put_bot_on_path(bot_dir)
    from _common.wire import _read_observation  # same parser the clean loop uses

    agent_class = _load_agent_class(bot_dir)
    probe = load_probe(bot_dir)

    target = _capture_target(capture_module, trace, capture_out)
    capture = load_capture(target[0]) if target is not None else None
    sink = CaptureSink(target[1]) if target is not None else None

    stdin, stdout = sys.stdin, sys.stdout
    handshake = stdin.readline()
    if not handshake:
        return
    player_id, H, W = (int(x) for x in handshake.split())

    agent = agent_class(player_id=player_id, H=H, W=W)
    # Before the first move: a capture that wraps agent internals (the RNG
    # proxy) has to be in place for turn 1, or the recorded draw stream starts
    # one turn late and every replay against it is off by a turn.
    capture_install(capture, agent)
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

            captured = capture_frame(capture, agent, obs, (p, r, c, d, s), turn)
            if captured is not None and sink is not None:
                # Written here, after the reply is flushed and outside every
                # clock the agent reports. See arena.instrument.capture on why
                # captures stream while traces buffer.
                sink.write(captured)
    finally:
        # Buffered to the end: a per-turn flush would put file io on the move
        # path, which is exactly what must not perturb the game.
        if trace is not None and lines:
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if sink is not None:
            sink.close()
            capture_close(capture)


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
    parser.add_argument(
        "--capture",
        type=Path,
        default=None,
        help=(
            "path to a capture module (see arena.instrument.capture); defaults "
            f"to ${{{CAPTURE_MODULE_ENV}}} when that is set"
        ),
    )
    parser.add_argument(
        "--capture-out",
        type=Path,
        default=None,
        help=f"capture output path (default: <trace>{CAPTURE_SUFFIX})",
    )
    args = parser.parse_args(argv)

    bot_dir = args.bot_dir.resolve()
    if not (bot_dir / "agent.py").is_file():
        parser.error(f"no agent.py under {bot_dir}")
    run_instrumented(
        bot_dir,
        args.trace,
        capture_module=args.capture or capture_module_from_env(),
        capture_out=args.capture_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
