"""Capture the joe-rs parity corpus from the Python joe bot (port-plan §6).

Two phases:

1. **Play** (`--play`): run real `--mode competition` matchup games with the
   deployed `bots/joe/run.sh` seat wrapped in a `tee`, recording the exact
   wire text joe saw (`.in.log`) and replied (`.out.log`).
2. **Capture** (`--capture`): replay each recorded game's frames through the
   *imported* `bots/joe/agent.py` / `joe_obs.py` functions — the same code
   the deployed bot runs, under `eqx.filter_jit` like the deployed step —
   and dump per-turn surfaces to a `.npz`. The recomputed greedy replies are
   cross-checked against the recorded `.out.log`; any mismatch means the
   capture path diverged from deployment and the tool aborts.

Corpus layout (`data/joe/joe-rs-parity/`, gitignored):
    games/<name>.in.log     wire text joe received (handshake + frames)
    games/<name>.out.log    wire text joe sent (one action line per turn)
    games/<name>.npz        per-turn surfaces (see `capture_game`)
    games/<name>.json       opponent, seed, outcome

Every turn stores the FNV-1a-64 hash of the augmented tensor's f32 bits (the
sequence surface's currency) plus the action; a stratified sample of turns
stores every full surface, including the input obs-state, for localization
when a sequence check fails.

Usage:
    .venv/bin/python bots/joe-rs/tools/capture_fixtures.py --play --capture
    .venv/bin/python bots/joe-rs/tools/capture_fixtures.py --capture   # logs exist
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
JOE_DIR = REPO / "bots" / "joe"
CORPUS = REPO / "data" / "joe" / "joe-rs-parity"
PAD = 21

# Mixed opponents, mixed seeds (port-plan §6). The joe mirror is the long
# game / truncation candidate: two identical nets stall each other, so the
# corpus carries three of them.
#
# This list is the whole documented corpus (`docs/bots/joe-rs/parity.md`:
# 14 games). Keep it that way. The last four used to be passed by hand as
# `--game`, which made the documented re-export sequence — a bare
# `--play --capture` — rebuild only the first ten and silently drop two of
# the three joe mirrors. A corpus member that lives in a shell history is
# not a corpus member.
DEFAULT_GAMES = [
    ("aegis", 0),
    ("macaria", 1),
    ("boom", 2),
    ("castle_rush", 3),
    ("cm_hunter", 4),
    ("general_hunter", 5),
    ("expand_plus", 6),
    ("blitz", 7),
    ("metro", 8),
    ("joe", 9),
    ("joe", 10),
    ("joe", 11),
    ("garrison", 12),
    ("metro", 42),
]

sys.path.insert(0, str(JOE_DIR))
sys.path.insert(0, str(REPO / "bots"))


def aug_hash(arr: np.ndarray) -> np.int64:
    """CRC-32 (zlib) over the little-endian f32 bytes — mirror of
    `parity.rs::crc32`. Any single-bit difference in any turn's tensor flips
    it, which is all the sequence check needs; localization then runs the
    per-frame surfaces."""
    import zlib

    return np.int64(zlib.crc32(np.ascontiguousarray(arr, dtype="<f4").tobytes()))


def play_games(games, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    python = REPO / ".venv" / "bin" / "python"
    wrapper = out_dir / "_tee_joe.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'tee "$JOE_TEE_IN" | bash "$JOE_RUN_SH" | tee "$JOE_TEE_OUT"\n'
    )
    wrapper.chmod(0o755)

    for opponent, seed in games:
        name = f"{opponent}-seed{seed}"
        in_log = out_dir / f"{name}.in.log"
        if in_log.exists():
            print(f"[capture] {name}: logs exist, skipping play")
            continue
        env = dict(os.environ)
        env["PYTHON"] = str(python)
        env["JOE_TEE_IN"] = str(in_log)
        env["JOE_TEE_OUT"] = str(out_dir / f"{name}.out.log")
        env["JOE_RUN_SH"] = str(JOE_DIR / "run.sh")
        opp_run = REPO / "bots" / opponent / "run.sh"
        print(f"[capture] playing joe vs {opponent} (seed {seed})")
        proc = subprocess.run(
            [str(python), str(REPO / "competition-module" / "competition" / "matchup.py"),
             str(wrapper), str(opp_run), "--mode", "competition", "--seed", str(seed)],
            env=env, capture_output=True, text=True)
        tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
        if proc.returncode != 0:
            raise SystemExit(
                f"matchup failed for {name}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        outcome = "unknown"
        for line in proc.stdout.splitlines():
            if "captured the enemy general" in line or "truncated" in line:
                outcome = line.strip()
        (out_dir / f"{name}.json").write_text(json.dumps(
            {"opponent": opponent, "seed": seed, "outcome": outcome}, indent=2) + "\n")
        print(f"[capture]   {outcome}")


def parse_in_log(path: Path):
    """Handshake + per-turn frames, exactly as `_common/wire.py` reads them."""
    lines = path.read_text().splitlines()
    player_id, H, W = (int(x) for x in lines[0].split())
    frames = []
    pos = 1
    frame_len = 1 + 3 * H
    while pos < len(lines):
        chunk = lines[pos:pos + frame_len]
        if len(chunk) < frame_len:
            break  # engine closed mid-frame — normal game end
        scalars = [int(x) for x in chunk[0].split()]
        grids = np.array(
            [[int(x) for x in row.split()] for row in chunk[1:]],
            dtype=np.int32).reshape(3, H, W)
        frames.append((scalars, grids))
        pos += frame_len
    return player_id, H, W, frames


def capture_game(name: str, out_dir: Path, stride_target: int = 40) -> None:
    npz_path = out_dir / f"{name}.npz"
    if npz_path.exists():
        print(f"[capture] {name}: npz exists, skipping")
        return

    import equinox as eqx
    import jax.numpy as jnp
    import jax.random as jrandom

    from _common.wire import Observation
    # The two penalty constants are imported, never copied: this file has to
    # reproduce agent.py's emitted move exactly, and a second literal here
    # would drift the moment either is retuned.
    from agent import REPEAT_DECAY, REPEAT_PENALTY, frame_to_raw
    from joe_net import HistoryTransformer
    from joe_obs import (
        N_ACTION_CHANNELS,
        augment_obs,
        build_cost_from_raw,
        compute_build_mask_from_raw,
        compute_valid_move_mask,
        decode_action,
        init_obs_state,
    )

    with open(JOE_DIR / "artifact" / "manifest.json") as f:
        manifest = json.load(f)
    arch = manifest["network"]
    template = HistoryTransformer(
        grid_size=int(arch["pad_to"]), pad_to=int(arch["pad_to"]),
        history_size=int(arch["history_size"]), patch_size=int(arch["patch_size"]),
        depth=int(arch["depth"]), embed_dim=int(arch["embed_dim"]),
        n_head=int(arch["n_head"]), ff_factor=int(arch["ff_factor"]),
        use_bf16=False, value_loss=arch["value_loss"], num_bins=int(arch["num_bins"]),
        v_min=float(arch["v_min"]), v_max=float(arch["v_max"]),
        key=jrandom.PRNGKey(0))
    net = eqx.tree_deserialise_leaves(str(JOE_DIR / "artifact" / manifest["weights"]), template)

    pad_to = int(arch["pad_to"])

    @eqx.filter_jit
    def capture_step(net, raw, obs_state, visits):
        """agent.py::step with every intermediate returned.

        Every surface here is the **unpenalised** network path, because that
        is what joe-rs ports. joe's repetition penalty (agent.py, added
        2026-08-16) is a research layer on top of the net, deliberately left
        out of the port, so grading the Rust binary against a penalised
        oracle would fail it for a difference it is supposed to have.

        `penalised_action` is the one exception. It is not stored in the
        .npz and joe-rs never sees it; it exists only so the recorded
        .out.log still cross-checks this capture against the deployed path.
        Without it the assertion below would compare two different programs.
        """
        cost = build_cost_from_raw(raw)
        aug, new_state = augment_obs(raw, cost, obs_state)
        move = compute_valid_move_mask(raw[0], raw[5] > 0, raw[3] > 0)
        build = compute_build_mask_from_raw(raw, cost)
        temporal = jnp.stack(
            [new_state.opponent_army_history, new_state.opponent_land_history])
        logits, value, value_bins = net._forward(aug, move, build, temporal)
        idx = jnp.argmax(logits)
        action = decode_action(idx, pad_to)

        # agent.py::step, from the penalty to the decayed count, mirrored.
        penalised = (logits.reshape(N_ACTION_CHANNELS, pad_to, pad_to)
                     - REPEAT_PENALTY * visits[None, :, :]).reshape(-1)
        p_idx = jnp.argmax(penalised)
        cells = pad_to * pad_to
        channel, position = p_idx // cells, p_idx % cells
        row, col = position // pad_to, position % pad_to
        new_visits = (visits * REPEAT_DECAY).at[row, col].add(
            jnp.where(channel < 8, 1.0, 0.0))

        return dict(cost=cost, aug=aug, move=move, build=build, temporal=temporal,
                    logits=logits, value=value, value_bins=value_bins,
                    idx=idx, action=action,
                    penalised_action=decode_action(p_idx, pad_to)), new_state, new_visits

    player_id, H, W, frames = parse_in_log(out_dir / f"{name}.in.log")
    out_log = out_dir / f"{name}.out.log"
    if out_log.exists():
        replies = out_log.read_text().splitlines()
        # A reply can be missing for the very last frame if the seat was
        # closed mid-turn; compare only paired turns.
        T = min(len(frames), len(replies))
    else:
        # Synthetic frame streams (e.g. the stitched long sequence) have no
        # recorded replies; the oracle's own outputs are the fixture.
        replies = None
        T = len(frames)
    frames = frames[:T]

    # Stratified sample for the per-frame surfaces: every ~stride turns plus
    # the first and last three (early state growth and late accumulation).
    stride = max(1, T // stride_target)
    sampled = sorted(set(range(0, T, stride)) | {0, 1, 2} | {T - 3, T - 2, T - 1})
    sampled = [t for t in sampled if 0 <= t < T]

    state = init_obs_state(int(arch["pad_to"]))
    state_fields = state._fields
    # agent.py drops the warm-up compile's counts, so a real game starts from
    # zeros here too — which is why joe's first move is the plain argmax.
    visits = jnp.zeros((pad_to, pad_to), dtype=jnp.float32)

    all_hash = np.zeros(T, dtype=np.int64)
    all_action = np.zeros((T, 5), dtype=np.int32)
    all_idx = np.zeros(T, dtype=np.int64)
    all_value = np.zeros(T, dtype=np.float32)
    per_frame = {k: [] for k in
                 ("raw", "cost", "aug", "move", "build", "temporal",
                  "logits", "value", "value_bins", "idx", "action")}
    state_in_frames = {f"state_{k}": [] for k in state_fields}

    for t, (scalars, grids) in enumerate(frames):
        obs = Observation(
            H=H, W=W, turn=scalars[0], my_land=scalars[1], my_army=scalars[2],
            opp_land=scalars[3], opp_army=scalars[4],
            type_grid=grids[0].tolist(), owner_grid=grids[1].tolist(),
            army_grid=grids[2].tolist())
        raw = jnp.asarray(frame_to_raw(obs))
        if t in sampled:
            for k, v in zip(state_fields, state):
                state_in_frames[f"state_{k}"].append(np.asarray(v))
        outs, state, visits = capture_step(net, raw, state, visits)

        aug_np = np.asarray(outs["aug"], dtype=np.float32)
        all_hash[t] = aug_hash(aug_np)
        action = np.asarray(outs["action"], dtype=np.int32)
        all_action[t] = action
        all_idx[t] = int(outs["idx"])
        all_value[t] = float(outs["value"])

        # The deployed reply (after agent.py's pass clamp) must match what
        # the live game recorded, or this capture is not the deployed path.
        # This is the *penalised* action: the .out.log was recorded by the
        # deployed joe, which applies the penalty. The surfaces above stay
        # unpenalised — joe-rs is graded on the net, not on joe's research
        # layer — so these two deliberately differ on repeated cells.
        if replies is not None:
            p, r, c, d, s = (int(x) for x in
                             np.asarray(outs["penalised_action"], dtype=np.int32))
            reply = (1, 0, 0, 0, 0) if p == 1 else (p, r, c, d, s)
            recorded = tuple(int(x) for x in replies[t].split())
            assert reply == recorded, \
                f"{name} turn {t}: recomputed {reply} != recorded {recorded}"

        if t in sampled:
            per_frame["raw"].append(np.asarray(raw, dtype=np.float32))
            per_frame["cost"].append(np.asarray(outs["cost"], dtype=np.int32))
            per_frame["aug"].append(aug_np)
            per_frame["move"].append(np.asarray(outs["move"], dtype=bool))
            per_frame["build"].append(np.asarray(outs["build"], dtype=bool))
            per_frame["temporal"].append(np.asarray(outs["temporal"], dtype=np.float32))
            per_frame["logits"].append(np.asarray(outs["logits"], dtype=np.float32))
            per_frame["value"].append(np.float32(outs["value"]))
            per_frame["value_bins"].append(np.asarray(outs["value_bins"], dtype=np.float32))
            per_frame["idx"].append(np.int64(outs["idx"]))
            per_frame["action"].append(action)

    payload = {
        "player_id": np.int32(player_id), "H": np.int32(H), "W": np.int32(W),
        "turns": np.int64(T), "sampled_turns": np.asarray(sampled, dtype=np.int64),
        "all_aug_hash": all_hash, "all_action": all_action,
        "all_idx": all_idx, "all_value": all_value,
    }
    for k, v in per_frame.items():
        payload[k] = np.stack(v)
    for k, v in state_in_frames.items():
        payload[k] = np.stack(v)
    for k, v in zip(state_fields, state):
        payload[f"final_state_{k}"] = np.asarray(v)
    np.savez_compressed(npz_path, **payload)
    print(f"[capture] {name}: {T} turns, {len(sampled)} sampled frames "
          f"-> {npz_path.name} ({npz_path.stat().st_size // 1024} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play", action="store_true", help="play the corpus games")
    parser.add_argument("--capture", action="store_true", help="replay logs into npz")
    parser.add_argument("--out", type=Path, default=CORPUS / "games")
    parser.add_argument("--game", nargs=2, action="append", metavar=("OPP", "SEED"),
                        help="extra (opponent, seed) game; replaces the default list")
    args = parser.parse_args()
    if not (args.play or args.capture):
        parser.error("pass --play and/or --capture")

    if args.play:
        games = ([(o, int(s)) for o, s in args.game] if args.game else DEFAULT_GAMES)
        play_games(games, args.out)
    if args.capture:
        for path in sorted(args.out.glob("*.in.log")):
            capture_game(path.name.removesuffix(".in.log"), args.out)


if __name__ == "__main__":
    main()
