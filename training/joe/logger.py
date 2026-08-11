"""Minimal run logger: JSONL metrics + stdout, no external services.

The released code logs to wandb; this repo keeps runs self-contained on the
Modal Volume, so metrics land next to the checkpoints as one JSON line per
logged step. Anything downstream (plots, reports) reads the file.
"""

import json
import os
import time


def _to_plain(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


class Logger:
    def __init__(self, run_dir: str, hparams: dict | None = None):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self._file = open(self.path, "a")
        if hparams is not None:
            with open(os.path.join(run_dir, "hparams.json"), "w") as f:
                json.dump({k: _to_plain(v) if not isinstance(v, (list, dict, str, bool, type(None))) else v
                           for k, v in hparams.items()}, f, indent=2)

    def log(self, step: int, metrics: dict):
        row = {"step": int(step), "time": time.time()}
        row.update({k: _to_plain(v) for k, v in metrics.items()})
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def log_eval(self, step: int, wins: int, losses: int, draws: int, finished: int):
        self.log(step, {
            "eval/wins": wins, "eval/losses": losses, "eval/draws": draws,
            "eval/finished": finished,
            "eval/win_rate": wins / max(finished, 1),
        })

    def finish(self):
        self._file.close()
