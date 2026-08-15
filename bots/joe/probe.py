"""
Per-turn probe for `joe`: what the network actually returned, before the argmax.

Loaded only by `arena.instrument.runner` on recorded matches. Passive: it reads
`agent.logits`, `agent.value`, `agent.move_mask` and `agent.build_mask`, which
`agent.act` stashes as device arrays, and never writes to the agent. **Not** part
of the bot's source closure, hash, or submission bundle, and never imported from
`agent.py` — `arena.records.fingerprint` raises if it ever is.

Joe is one forward pass and an argmax, so a diagnosis has to separate three
different faults that all present as "joe cycles between two tiles":

- **near-tie**: top-1 and top-2 are within noise and the argmax alternates
  between them. `joe_margin_milli` is ~0 and `joe_entropy_milli` is high.
- **confident cycle**: the policy genuinely prefers the loop.
  `joe_margin_milli` is large and stays large.
- **stale input**: `AugmentedObsState` persists for the whole game and is never
  reset, so accumulated `seen`/`enemy_seen`/history channels can make two
  distinct board states produce the same tensor. The net is deterministic, so
  identical `joe_logits_hash` on two turns means identical input — that is the
  direct test, and it needs no copy of the observation.

`joe_top5` packs the decoded top five so the read side does not need the net.
Keys are declared in `arena.records.telemetry_schema`; an undeclared one fails
the recorded match rather than landing untyped.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Action channel layout (bots/joe/joe_obs.py decode_action): 0-3 full move,
# 4-7 half move, 8 pass, 9 build.
N_CHANNELS = 10
_WINDOW = 12  # how far back "recently occupied" and cycle detection look

# Probe-local history. Not agent state: the probe may not keep anything on the
# agent, and these series are about the trace, not about the decision.
_recent_cells: list[tuple[int, int]] = []
_recent_actions: list[tuple[int, int, int]] = []


def _decode(idx: int, pad_to: int) -> tuple[int, int, int]:
    """Flat logit index -> (channel, row, col); the pure-numpy decode_action."""
    gc = pad_to * pad_to
    d, pos = divmod(int(idx), gc)
    return d, pos // pad_to, pos % pad_to


def _label(d: int, r: int, c: int) -> str:
    """A whitespace-free name for one action: tokens may not contain spaces."""
    if d == 8:
        return "pass"
    if d == 9:
        return f"build@{r}.{c}"
    return f"{'half' if d >= 4 else 'move'}@{r}.{c}d{d % 4}"


def _cycle_period(seq: list[tuple[int, int, int]]) -> int:
    """
    Smallest p in 1..4 such that the last 3p entries repeat with period p.

    Three full repetitions, not two: two consecutive equal actions is ordinary
    play (a stack walking one direction), while three is a loop. Returns 0 when
    the tail does not repeat.
    """
    for p in range(1, 5):
        if len(seq) < 3 * p:
            continue
        tail = seq[-3 * p:]
        if all(tail[i] == tail[i % p] for i in range(len(tail))):
            return p
    return 0


def extras(agent) -> dict:
    logits = np.asarray(agent.logits, dtype=np.float32)
    pad_to = int(agent.pad_to)

    order = np.argsort(logits)[::-1][:5]
    top = [(int(i), float(logits[i])) for i in order]
    top1, top2 = top[0][1], top[1][1]

    # Softmax over the masked logits. prepare_action_mask adds -1e9 to illegal
    # entries, so they leave the distribution on their own; the max subtraction
    # is what keeps that from overflowing.
    shifted = np.exp(logits - top1)
    probs = shifted / shifted.sum()
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum())

    # Cycle and revisit series track the action joe **played**, which is the
    # penalised argmax, not the raw one the top-5 describes. Reading them off
    # `joe_top5` would report whether the *network* cycles while the repetition
    # penalty was busy making sure the *bot* does not — the opposite of the
    # question these two keys exist to answer.
    d1, r1, c1 = _decode(int(np.asarray(agent.action_idx)), pad_to)
    _recent_cells.append((r1, c1))
    _recent_actions.append((d1, r1, c1))
    del _recent_cells[:-_WINDOW], _recent_actions[:-_WINDOW]

    move_mask = np.asarray(agent.move_mask)
    build_mask = np.asarray(agent.build_mask)

    return {
        # Packed top five: "label:logit_milli" entries, highest first.
        "joe_top5": "|".join(
            f"{_label(*_decode(i, pad_to))}:{round(v * 1000)}" for i, v in top),
        "joe_margin_milli": round((top1 - top2) * 1000),
        "joe_entropy_milli": round(entropy * 1000),
        # Training rollouts sampled this distribution; deployment takes its
        # argmax. This is the mass argmax throws away, so 1/(1 - p) is how many
        # turns the training-time sampler needed to leave a state the deployed
        # policy never leaves.
        "joe_top1_prob_milli": round(float(probs.max()) * 1000),
        "joe_value_milli": round(float(np.asarray(agent.value)) * 1000),
        # Identical logits <=> identical net input, the net being deterministic.
        "joe_logits_hash": hashlib.blake2b(
            logits.tobytes(), digest_size=8).hexdigest(),
        "joe_move_cells": int(move_mask.sum()),
        "joe_build_cells": int(build_mask.sum()),
        # The repetition penalty: whether it moved the decision this turn, and
        # how much penalty has piled up on the worst cell. Reported as a pair
        # for the same reason macaria reports (searched, overrode) — "never
        # fired" and "fired and agreed" are different faults with one symptom.
        "joe_reppen_overrode": 1 if bool(agent.overrode) else 0,
        "joe_visit_peak_milli": round(
            float(np.asarray(agent.visits).max()) * 1000),
        # How many of the last _WINDOW turns already acted from this same cell.
        "joe_cell_revisits": _recent_cells[:-1].count((r1, c1)),
        "joe_cycle_period": _cycle_period(_recent_actions),
    }
