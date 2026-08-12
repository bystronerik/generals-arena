"""Run state (``state.json``) for resumable training.

Schema v2 (docs/research/strategies/joe-vast-training-plan.md, section 1):
one file in the checkpoint dir that records the global step, the curriculum
position, and the exact checkpoint files that form a consistent set — a
reader never has to glob and guess. Writes are atomic (tmp + ``os.replace``)
so a kill mid-write cannot corrupt the file.

Stdlib only — no JAX — so the round-trip tests run in the cheap default
suite.
"""

import json
import os
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


def check_curriculum_stage(stage_idx, num_stages):
    """Fail loudly when the saved stage no longer exists in the config."""
    if not 0 <= stage_idx < num_stages:
        raise ValueError(
            f"Saved curriculum_stage {stage_idx} is out of range for a "
            f"{num_stages}-stage curriculum; the config's curriculum list "
            "changed shape since the checkpoint")
