"""Resumable Morpheus trainer, immutable checkpoints, and run manifests (Part 14).

Training state stays under ``data/morpheus/`` or a Modal Volume. It never enters
the bot closure until Part 15 freezes one checkpoint.
"""

from training.morpheus.trainer.checkpoint import (
    CheckpointError,
    CheckpointState,
    load_checkpoint,
    write_checkpoint,
)
from training.morpheus.trainer.config import TrainRunConfig, load_train_run_config
from training.morpheus.trainer.loop import TrainResult, resume_training, run_training
from training.morpheus.trainer.manifest import (
    EVENT_ARENA_CHECKPOINT_ACCEPT,
    EVENT_CURRICULUM_CLASS_ADVANCE,
    EVENT_SNAPSHOT_SAVED,
    RunManifest,
    load_run_manifest,
    write_run_manifest,
)

__all__ = [
    "CheckpointError",
    "CheckpointState",
    "EVENT_ARENA_CHECKPOINT_ACCEPT",
    "EVENT_CURRICULUM_CLASS_ADVANCE",
    "EVENT_SNAPSHOT_SAVED",
    "RunManifest",
    "TrainResult",
    "TrainRunConfig",
    "load_checkpoint",
    "load_run_manifest",
    "load_train_run_config",
    "resume_training",
    "run_training",
    "write_checkpoint",
    "write_run_manifest",
]
