"""Export Morpheus float checkpoints to sandbox static 8-bit TorchScript artifacts."""
from __future__ import annotations

import json
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from network import (
    BOARD,
    IN_CHANNELS,
    MorpheusNet,
    MorpheusOutput,
    architecture_summary,
    make_model,
)
from schema import (
    ARMY_BIN_EDGES,
    ARMY_SCALE_DEFAULT,
    N_ARMY_BINS,
    build_manifest,
    write_manifest,
)

MANIFEST_NAME = "manifest.json"
DEFAULT_ARTIFACT_FILE = "model.pt"

EXPORT_OUTPUT_NAMES = (
    "policy",
    "pass_logit",
    "wdl",
    "hidden_owner",
    "enemy_army_bins",
    "enemy_general",
    "hidden_castle",
    "land_margin",
    "army_margin",
    "castle_margin",
    "turns_to_termination",
)


def expected_export_shapes(batch: int) -> list[list[int]]:
    """Canonical TorchScript output shapes for one batch size."""
    return [
        [batch, 9, BOARD, BOARD],
        [batch, 1],
        [batch, 3],
        [batch, 1, BOARD, BOARD],
        [batch, N_ARMY_BINS, BOARD, BOARD],
        [batch, 1, BOARD, BOARD],
        [batch, 1, BOARD, BOARD],
        [batch, 1],
        [batch, 1],
        [batch, 1],
        [batch, 1],
    ]


def _supported_qengines() -> list[str]:
    try:
        return list(torch.backends.quantized.supported_engines)
    except Exception:  # noqa: BLE001
        return []


def _pick_qengine() -> str:
    supported = _supported_qengines()
    for name in ("qnnpack", "fbgemm", "x86"):
        if name in supported:
            return name
    if supported:
        return supported[0]
    raise RuntimeError("no torch quantized engine available on this host")


def _set_qengine(engine: str) -> None:
    torch.backends.quantized.engine = engine


class MorpheusExportWrapper(nn.Module):
    """Tuple outputs for FX quantization and TorchScript trace."""

    def __init__(self, model: MorpheusNet) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        out = self.model(x)
        return (
            out.policy,
            out.pass_logit,
            out.wdl_logits,
            out.hidden_owner,
            out.enemy_army_bins,
            out.enemy_general,
            out.hidden_castle,
            out.land_margin,
            out.army_margin,
            out.castle_margin,
            out.turns_to_termination,
        )


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    """Load a trainer checkpoint or bare state dict from a file or directory."""
    if path.is_dir():
        candidates = [
            path / "state_dict.pt",
            path / "model.pt",
            path / "checkpoint.pt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"no checkpoint file in {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
        meta = {k: v for k, v in raw.items() if k != "state_dict"}
    elif isinstance(raw, dict) and all(isinstance(k, str) for k in raw):
        state = raw
        meta = {}
    else:
        raise TypeError(f"unrecognized checkpoint format: {path}")
    meta_path = path.parent / "meta.json"
    if meta_path.is_file():
        meta = {**json.loads(meta_path.read_text()), **meta}
    return {"state_dict": state, "meta": meta}


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> MorpheusNet:
    meta = checkpoint.get("meta", {})
    n_blocks = int(meta.get("n_blocks", 12))
    seed = int(meta.get("seed", 0))
    model = make_model(seed=seed, n_blocks=n_blocks)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    manifest_path: Path
    artifact_path: Path
    qengine: str
    float_to_export_mae: dict[str, float]


def _mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def _parity_errors(
    float_out: tuple[torch.Tensor, ...],
    export_out: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    return {
        f"{name}_mae": _mae(f, e)
        for name, f, e in zip(EXPORT_OUTPUT_NAMES, float_out, export_out, strict=True)
    }


def export_static_int8(
    model: MorpheusNet,
    output_dir: Path,
    *,
    qengine: Optional[str] = None,
    training_run: Optional[dict[str, Any]] = None,
    example_batch: int = 1,
    seed: int = 0,
) -> ExportResult:
    """FX static PTQ export to ``output_dir`` with manifest and traced ``model.pt``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = qengine or _pick_qengine()
    supported = _supported_qengines()
    if engine not in supported:
        raise RuntimeError(
            f"quantized engine {engine!r} not in supported_engines={supported}"
        )
    _set_qengine(engine)

    wrapper = MorpheusExportWrapper(model)
    wrapper.eval()
    torch.manual_seed(seed)
    example = torch.randn(example_batch, IN_CHANNELS, BOARD, BOARD)

    with torch.no_grad():
        float_ref = wrapper(example)

    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        qconfig_mapping = get_default_qconfig_mapping(engine)
        prepared = prepare_fx(wrapper, qconfig_mapping, example_inputs=(example,))
        with torch.no_grad():
            for batch in (1, 4, 64):
                prepared(torch.randn(batch, IN_CHANNELS, BOARD, BOARD))
        quantized = convert_fx(prepared)

    artifact_path = output_dir / DEFAULT_ARTIFACT_FILE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        with torch.no_grad():
            traced = torch.jit.trace(quantized, example, check_trace=False, strict=False)
        torch.jit.save(traced, str(artifact_path))

    loaded = torch.jit.load(str(artifact_path))
    loaded.eval()
    with torch.no_grad():
        export_out = loaded(example)
    parity = _parity_errors(float_ref, export_out)

    arch = architecture_summary(model)
    quantization = {
        "format": f"static_ptq_fx_{engine}_int8",
        "engine": engine,
        "runtime": f"torch.jit.trace+fx_static_{engine}",
        "group_norm_float_fallback": True,
        "float_to_export_mae": parity,
        "host": platform.platform(),
    }
    run_meta = training_run or {
        "run_id": "export-local",
        "checkpoint_id": "unknown",
        "note": "Part 04 export adapter",
    }
    manifest = build_manifest(
        weights_path=artifact_path,
        architecture=arch,
        quantization=quantization,
        training_run=run_meta,
        runtime=quantization["runtime"],
        artifact_file=DEFAULT_ARTIFACT_FILE,
        army_scale=ARMY_SCALE_DEFAULT,
        army_bin_edges=ARMY_BIN_EDGES,
    )
    manifest_path = output_dir / MANIFEST_NAME
    write_manifest(manifest_path, manifest)

    return ExportResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        qengine=engine,
        float_to_export_mae=parity,
    )


def export_from_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    qengine: Optional[str] = None,
    training_run: Optional[dict[str, Any]] = None,
) -> ExportResult:
    checkpoint = load_checkpoint_state(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint)
    meta = checkpoint.get("meta", {})
    seed = int(meta.get("seed", 0))
    return export_static_int8(
        model,
        output_dir,
        qengine=qengine,
        training_run=training_run,
        seed=seed,
    )


def forward_exported(
    session: torch.jit.ScriptModule,
    x: torch.Tensor,
) -> MorpheusOutput:
    """Run a traced export and rebuild the named output tuple."""
    outputs = session(x)
    if len(outputs) != len(EXPORT_OUTPUT_NAMES):
        raise ValueError(
            f"export returned {len(outputs)} tensors; expected {len(EXPORT_OUTPUT_NAMES)}"
        )
    return MorpheusOutput(*outputs)
