"""Atomic immutable training checkpoints (Part 14).

A checkpoint directory is never overwritten. Writes go to a temporary sibling
directory, then ``os.replace`` publishes the final content-addressed name.
Resume rejects schema, engine-era, tensor/action schema, or run-manifest drift.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from training.morpheus.self_play.league import League
from training.morpheus.trainer.buffer import BufferCursor
from training.morpheus.trainer.manifest import load_run_manifest

REPO = Path(__file__).resolve().parents[3]
MORPHEUS_BOT = REPO / "bots" / "morpheus"

CHECKPOINT_META = "meta.json"
MODEL_FILE = "model.pt"
OPTIMIZER_FILE = "optimizer.pt"
SCHEDULER_FILE = "scheduler.pt"
LEAGUE_FILE = "league.json"
CURRICULUM_FILE = "curriculum.json"
RNG_FILE = "rng.pt"
CONSUMED_FILE = "consumed.json"
BUDGET_FILE = "budget.json"
POINTER_NAME = "latest_checkpoint.json"


class CheckpointError(RuntimeError):
    """Checkpoint write/load or resume validation failed."""


def _ensure_bot_path() -> None:
    for entry in (REPO, REPO / "bots", MORPHEUS_BOT):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


@dataclass
class CheckpointState:
    """Full resumable trainer state for one immutable snapshot."""

    global_step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    league: dict[str, Any]
    curriculum: dict[str, Any]
    rng: dict[str, Any]
    consumed: BufferCursor
    budget: dict[str, Any]
    run_compat: dict[str, Any]
    n_blocks: int
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    def content_digest(self) -> str:
        """Stable digest over model weights and step identity."""
        hasher = hashlib.sha256()
        hasher.update(f"step={self.global_step}\n".encode("utf-8"))
        # torch.save to bytes would be platform-sensitive; hash tensor bytes.
        for key in sorted(self.model_state):
            tensor = self.model_state[key]
            hasher.update(key.encode("utf-8"))
            if hasattr(tensor, "detach"):
                arr = tensor.detach().cpu().contiguous().numpy().tobytes()
            else:
                arr = bytes(str(tensor), "utf-8")
            hasher.update(arr)
        return f"sha256:{hasher.hexdigest()}"


def capture_rng_state(np_rng) -> dict[str, Any]:
    """Capture numpy Generator + torch CPU/CUDA RNG for exact resume.

    Stored via ``torch.save`` (not JSON) because bit-generator state contains
    ndarray values.
    """
    torch_state: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
        "cuda_all": None,
    }
    if torch.cuda.is_available():
        torch_state["cuda_all"] = torch.cuda.get_rng_state_all()
    return {
        "numpy": np_rng.bit_generator.state,
        "torch": torch_state,
    }


def restore_rng_state(payload: Mapping[str, Any], np_rng) -> None:
    np_state = payload.get("numpy")
    if np_state is not None:
        np_rng.bit_generator.state = np_state
    torch_state = payload.get("torch") or {}
    cpu = torch_state.get("cpu")
    if cpu is not None:
        torch.set_rng_state(cpu)
    cuda_all = torch_state.get("cuda_all")
    if cuda_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_all)


def checkpoint_dir_name(global_step: int, digest: str) -> str:
    short = digest.split(":", 1)[-1][:12]
    return f"ckpt-{int(global_step):08d}-{short}"


def list_checkpoints(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    paths = [
        p
        for p in run_dir.iterdir()
        if p.is_dir() and p.name.startswith("ckpt-") and (p / CHECKPOINT_META).is_file()
    ]
    return sorted(paths, key=lambda p: p.name)


def latest_checkpoint_path(run_dir: Path) -> Path | None:
    pointer = Path(run_dir) / POINTER_NAME
    if pointer.is_file():
        data = json.loads(pointer.read_text(encoding="utf-8"))
        name = data.get("checkpoint_id")
        if name:
            path = Path(run_dir) / str(name)
            if path.is_dir():
                return path
    paths = list_checkpoints(run_dir)
    return paths[-1] if paths else None


def write_checkpoint(
    run_dir: Path,
    state: CheckpointState,
    *,
    event: str = "snapshot_saved",
) -> Path:
    """Atomically publish an immutable checkpoint directory."""
    run_dir = Path(run_dir)
    if "data/games" in run_dir.as_posix():
        raise CheckpointError(f"refusing to write checkpoints under {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    digest = state.content_digest()
    final_name = checkpoint_dir_name(state.global_step, digest)
    final_path = run_dir / final_name
    if final_path.exists():
        # Identical content-addressed name: treat as already published.
        return final_path

    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{final_name}.tmp-", dir=str(run_dir))
    )
    try:
        meta = {
            "checkpoint_id": final_name,
            "global_step": int(state.global_step),
            "content_digest": digest,
            "n_blocks": int(state.n_blocks),
            "seed": int(state.seed),
            "event": event,
            "run_compat": dict(state.run_compat),
            **dict(state.meta),
        }
        (tmp_dir / CHECKPOINT_META).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        torch.save(state.model_state, tmp_dir / MODEL_FILE)
        # Also write state_dict.pt so bots/morpheus/export can reload weights.
        torch.save(state.model_state, tmp_dir / "state_dict.pt")
        torch.save(state.optimizer_state, tmp_dir / OPTIMIZER_FILE)
        if state.scheduler_state is not None:
            torch.save(state.scheduler_state, tmp_dir / SCHEDULER_FILE)
        (tmp_dir / LEAGUE_FILE).write_text(
            json.dumps(state.league, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (tmp_dir / CURRICULUM_FILE).write_text(
            json.dumps(state.curriculum, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        torch.save(state.rng, tmp_dir / RNG_FILE)
        (tmp_dir / CONSUMED_FILE).write_text(
            json.dumps(state.consumed.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (tmp_dir / BUDGET_FILE).write_text(
            json.dumps(state.budget, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_dir, final_path)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Pointer is the only mutable file; it names an immutable checkpoint.
    pointer = {
        "checkpoint_id": final_name,
        "global_step": int(state.global_step),
        "content_digest": digest,
        "path": final_name,
    }
    pointer_path = run_dir / POINTER_NAME
    tmp_pointer = pointer_path.with_suffix(".tmp")
    tmp_pointer.write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_pointer, pointer_path)
    return final_path


def load_checkpoint(path: Path) -> CheckpointState:
    path = Path(path)
    if not path.is_dir():
        raise CheckpointError(f"checkpoint directory missing: {path}")
    meta_path = path / CHECKPOINT_META
    if not meta_path.is_file():
        raise CheckpointError(f"checkpoint meta missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_file = path / MODEL_FILE
    if not model_file.is_file():
        model_file = path / "state_dict.pt"
    if not model_file.is_file():
        raise CheckpointError(f"model weights missing in {path}")
    optimizer_file = path / OPTIMIZER_FILE
    if not optimizer_file.is_file():
        raise CheckpointError(f"optimizer state missing in {path}")
    scheduler_state = None
    scheduler_file = path / SCHEDULER_FILE
    if scheduler_file.is_file():
        scheduler_state = torch.load(scheduler_file, map_location="cpu", weights_only=False)

    league = json.loads((path / LEAGUE_FILE).read_text(encoding="utf-8"))
    curriculum = json.loads((path / CURRICULUM_FILE).read_text(encoding="utf-8"))
    rng_path = path / RNG_FILE
    if not rng_path.is_file():
        raise CheckpointError(f"rng state missing in {path}")
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    consumed = BufferCursor.from_dict(
        json.loads((path / CONSUMED_FILE).read_text(encoding="utf-8"))
    )
    budget = json.loads((path / BUDGET_FILE).read_text(encoding="utf-8"))
    return CheckpointState(
        global_step=int(meta["global_step"]),
        model_state=torch.load(model_file, map_location="cpu", weights_only=False),
        optimizer_state=torch.load(
            optimizer_file, map_location="cpu", weights_only=False
        ),
        scheduler_state=scheduler_state,
        league=league,
        curriculum=curriculum,
        rng=rng,
        consumed=consumed,
        budget=budget,
        run_compat=dict(meta.get("run_compat") or {}),
        n_blocks=int(meta.get("n_blocks") or 12),
        seed=int(meta.get("seed") or 0),
        meta=meta,
    )


def validate_checkpoint_against_run(path: Path, run_dir: Path) -> CheckpointState:
    """Load a checkpoint and reject resume on schema / run-manifest drift."""
    state = load_checkpoint(path)
    manifest = load_run_manifest(run_dir)
    expected = manifest.compatibility_key()
    got = dict(state.run_compat)
    mismatches = {
        k: (expected[k], got.get(k))
        for k in expected
        if got.get(k) != expected[k]
    }
    if mismatches:
        raise CheckpointError(
            f"resume rejected; checkpoint run_compat mismatch: {mismatches}"
        )
    return state


def build_model_and_optimizer(
    *,
    n_blocks: int,
    seed: int,
    optimizer_cfg,
    scheduler_cfg,
    device: torch.device,
):
    """Construct model + AdamW + optional StepLR for a fresh or resumed run."""
    _ensure_bot_path()
    from network import make_model

    torch.manual_seed(seed)
    model = make_model(seed=seed, n_blocks=n_blocks).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.learning_rate),
        weight_decay=float(optimizer_cfg.weight_decay),
        betas=tuple(optimizer_cfg.betas),
        eps=float(optimizer_cfg.eps),
    )
    if scheduler_cfg.name == "constant":
        # Identity step schedule: gamma=1 keeps LR fixed; step_size unused.
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(scheduler_cfg.step_size), gamma=1.0
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.step_size),
            gamma=float(scheduler_cfg.gamma),
        )
    return model, optimizer, scheduler


def apply_checkpoint_to_modules(
    state: CheckpointState,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> None:
    model.load_state_dict(state.model_state)
    optimizer.load_state_dict(state.optimizer_state)
    if state.scheduler_state is not None and scheduler is not None:
        scheduler.load_state_dict(state.scheduler_state)


def league_from_checkpoint(state: CheckpointState) -> League:
    return League.from_dict(state.league)
