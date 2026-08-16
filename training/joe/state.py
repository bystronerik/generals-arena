"""Run state (``state.json``) for resumable training.

Schema v2 (docs/research/strategies/joe-vast-training-plan.md, section 1):
one file in the checkpoint dir that records the global step, the curriculum
position, and the exact checkpoint files that form a consistent set — a
reader never has to glob and guess. Writes are atomic (tmp + ``os.replace``)
so a kill mid-write cannot corrupt the file.

``prune_checkpoints`` is the disk-side counterpart: it removes step-named
files only below a step R2 has confirmed, so local retention follows what
the bucket durably holds.

Stdlib only — no JAX — so the round-trip tests run in the cheap default
suite.
"""

import json
import os
import re
import time

STATE_SCHEMA = 2
STATE_FILENAME = "state.json"


def write_state(ckpt_dir, run_name, global_step, curriculum_stage,
                last_eval_wr, engine_sha, files):
    """Atomically write ``state.json`` and return the written dict.

    ``files`` maps role (``"full"``, ``"ema"``) to the checkpoint basename
    the state refers to.
    """
    state = {
        "schema": STATE_SCHEMA,
        "run_name": run_name,
        "global_step": int(global_step),
        "curriculum_stage": int(curriculum_stage),
        "last_eval_wr": float(last_eval_wr),
        "engine_sha": engine_sha,
        "time": time.time(),
        "files": dict(files),
    }
    path = os.path.join(ckpt_dir, STATE_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)
    return state


def read_state(ckpt_dir):
    """Read ``state.json``; None when absent or pre-v2.

    A pre-v2 file (the old ``{"iteration": ...}`` shape) has no resume
    information, so the caller falls back to the legacy manual path
    (``init_checkpoint`` / ``iteration_offset``). A schema newer than this
    code raises: resuming from a misread state would corrupt the run.
    """
    path = os.path.join(ckpt_dir, STATE_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    schema = state.get("schema", 1)
    if schema < STATE_SCHEMA:
        return None
    if schema > STATE_SCHEMA:
        raise ValueError(
            f"{path} has schema {schema}; this code understands "
            f"{STATE_SCHEMA}")
    return state


def prune_checkpoints(ckpt_dir, run_name, keep_through_step):
    """Delete step-named checkpoints the bucket already supersedes.

    ``keep_through_step`` is the newest global step R2 has verified — what
    ``CheckpointUploader.__call__`` returns. A file goes only when its step
    is strictly below that, because the bucket then holds a newer complete
    set and the local copy is redundant. Files at or above it are the only
    copy until an upload confirms them, so they stay: a run whose uploads
    keep failing prunes nothing and loses nothing.

    Unnumbered artifacts (``<run>_ema.eqx``, ``<run>_final.eqx``,
    ``<run>_ema_final.eqx``) never match the pattern, and the basenames
    ``state.json`` names are protected outright. Returns the removed
    basenames.
    """
    if not keep_through_step or keep_through_step <= 0:
        return []
    pattern = re.compile(rf"^{re.escape(run_name)}_(?:ema_)?(\d+)\.eqx$")
    state = read_state(ckpt_dir)
    protected = set((state or {}).get("files", {}).values())
    removed = []
    for name in sorted(os.listdir(ckpt_dir)):
        match = pattern.match(name)
        if match is None or name in protected:
            continue
        if int(match.group(1)) >= keep_through_step:
            continue
        try:
            os.remove(os.path.join(ckpt_dir, name))
        except OSError as exc:
            print(f"could not prune {name}: {exc}", flush=True)
            continue
        removed.append(name)
    return removed


def check_curriculum_stage(stage_idx, num_stages):
    """Fail loudly when the saved stage no longer exists in the config."""
    if not 0 <= stage_idx < num_stages:
        raise ValueError(
            f"Saved curriculum_stage {stage_idx} is out of range for a "
            f"{num_stages}-stage curriculum; the config's curriculum list "
            "changed shape since the checkpoint")
