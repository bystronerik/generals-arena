"""Export Morpheus checkpoints to sandbox TorchScript artifacts.

Default deployment format is float32 TorchScript: at this model size the
static int8 path is both slower (GroupNorm float fallback forces
dequant→quant round trips) and lossy unless calibrated on real observation
tensors. The int8 path remains available via ``fmt="int8"`` for research.
"""
from __future__ import annotations

import copy
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
    sha256_file,
    write_manifest,
)

MANIFEST_NAME = "manifest.json"
DEFAULT_ARTIFACT_FILE = "model.pt"
POLICY_ARTIFACT_FILE = "model_policy.pt"
POLICY_WDL_ARTIFACT_FILE = "model_policy_wdl.pt"

FLOAT32_FORMAT = "float32"
FLOAT32_RUNTIME = "torch.jit.script+float32"

# Soft upper bounds from recorded Part 04 / Part 00b qnnpack MAE (with margin).
ONLINE_PARITY_LIMITS: dict[str, float] = {
    "policy_mae": 2.0,
    "pass_logit_mae": 1.0,
    "wdl_mae": 1.0,
}

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

POLICY_OUTPUT_NAMES = ("policy", "pass_logit")
POLICY_WDL_OUTPUT_NAMES = ("policy", "pass_logit", "wdl")


def expected_export_shapes(batch: int) -> list[list[int]]:
    """Canonical TorchScript output shapes for one batch size (full heads)."""
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


def expected_policy_shapes(batch: int) -> list[list[int]]:
    return [[batch, 9, BOARD, BOARD], [batch, 1]]


def expected_policy_wdl_shapes(batch: int) -> list[list[int]]:
    return [[batch, 9, BOARD, BOARD], [batch, 1], [batch, 3]]


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
    """Full-head tuple outputs for FX quantization and TorchScript trace."""

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


class MorpheusPolicyExportWrapper(nn.Module):
    """Policy + pass only — online belief proposal and enemy priors."""

    def __init__(self, model: MorpheusNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_policy(x)


class MorpheusPolicyWdlExportWrapper(nn.Module):
    """Policy + pass + WDL — online root and leaf evaluation."""

    def __init__(self, model: MorpheusNet) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward_policy_wdl(x)


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
    policy_artifact_path: Path
    policy_wdl_artifact_path: Path
    qengine: str
    float_to_export_mae: dict[str, float]
    online_float_to_export_mae: dict[str, dict[str, float]]


def _mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def _parity_errors(
    float_out: tuple[torch.Tensor, ...],
    export_out: tuple[torch.Tensor, ...],
    names: tuple[str, ...] = EXPORT_OUTPUT_NAMES,
) -> dict[str, float]:
    return {
        f"{name}_mae": _mae(f, e)
        for name, f, e in zip(names, float_out, export_out, strict=True)
    }


def assert_online_parity(
    mae: dict[str, float],
    *,
    limits: dict[str, float] = ONLINE_PARITY_LIMITS,
) -> None:
    """Raise if policy / pass / WDL MAE exceed recorded soft bounds."""
    for key, limit in limits.items():
        if key not in mae:
            continue
        value = float(mae[key])
        if value != value:  # NaN
            raise ValueError(f"online parity {key} is NaN")
        if value > limit:
            raise ValueError(
                f"online parity {key}={value:.6f} exceeds limit {limit:.6f}"
            )


def _quantize_and_trace(
    wrapper: nn.Module,
    example: torch.Tensor,
    artifact_path: Path,
    *,
    engine: str,
) -> torch.jit.ScriptModule:
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    wrapper.eval()
    _set_qengine(engine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        qconfig_mapping = get_default_qconfig_mapping(engine)
        prepared = prepare_fx(wrapper, qconfig_mapping, example_inputs=(example,))
        with torch.no_grad():
            for batch in (1, 4, 64):
                prepared(torch.randn(batch, IN_CHANNELS, BOARD, BOARD))
        quantized = convert_fx(prepared)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        with torch.no_grad():
            traced = torch.jit.trace(quantized, example, check_trace=False, strict=False)
        torch.jit.save(traced, str(artifact_path))

    loaded = torch.jit.load(str(artifact_path))
    loaded.eval()
    return loaded


def _script_and_save(
    wrapper: nn.Module,
    artifact_path: Path,
) -> torch.jit.ScriptModule:
    """Script and save a float wrapper — no quantization, bit-faithful weights."""
    wrapper.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        scripted = torch.jit.script(wrapper)
        torch.jit.save(scripted, str(artifact_path))

    loaded = torch.jit.load(str(artifact_path))
    loaded.eval()
    return loaded


def _export_artifacts(
    model: MorpheusNet,
    output_dir: Path,
    *,
    fmt: str,
    qengine: Optional[str],
    training_run: Optional[dict[str, Any]],
    example_batch: int,
    seed: int,
    require_online_parity: bool,
) -> ExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if fmt == FLOAT32_FORMAT:
        engine = "none"
    else:
        engine = qengine or _pick_qengine()
        supported = _supported_qengines()
        if engine not in supported:
            raise RuntimeError(
                f"quantized engine {engine!r} not in supported_engines={supported}"
            )
        _set_qengine(engine)

    torch.manual_seed(seed)
    example = torch.randn(example_batch, IN_CHANNELS, BOARD, BOARD)

    # Separate float copies so FX prepare/convert cannot mutate a shared trunk.
    full_wrapper = MorpheusExportWrapper(copy.deepcopy(model))
    policy_wrapper = MorpheusPolicyExportWrapper(copy.deepcopy(model))
    policy_wdl_wrapper = MorpheusPolicyWdlExportWrapper(copy.deepcopy(model))
    with torch.no_grad():
        float_full = full_wrapper(example)
        float_policy = policy_wrapper(example)
        float_policy_wdl = policy_wdl_wrapper(example)

    artifact_path = output_dir / DEFAULT_ARTIFACT_FILE
    policy_path = output_dir / POLICY_ARTIFACT_FILE
    policy_wdl_path = output_dir / POLICY_WDL_ARTIFACT_FILE

    if fmt == FLOAT32_FORMAT:
        loaded_full = _script_and_save(full_wrapper, artifact_path)
        loaded_policy = _script_and_save(policy_wrapper, policy_path)
        loaded_policy_wdl = _script_and_save(policy_wdl_wrapper, policy_wdl_path)
    else:
        loaded_full = _quantize_and_trace(
            full_wrapper, example, artifact_path, engine=engine
        )
        loaded_policy = _quantize_and_trace(
            policy_wrapper, example, policy_path, engine=engine
        )
        loaded_policy_wdl = _quantize_and_trace(
            policy_wdl_wrapper, example, policy_wdl_path, engine=engine
        )

    with torch.no_grad():
        export_full = loaded_full(example)
        export_policy = loaded_policy(example)
        export_policy_wdl = loaded_policy_wdl(example)

    parity = _parity_errors(float_full, export_full)
    online_parity = {
        "policy": _parity_errors(
            float_policy, export_policy, names=POLICY_OUTPUT_NAMES
        ),
        "policy_wdl": _parity_errors(
            float_policy_wdl, export_policy_wdl, names=POLICY_WDL_OUTPUT_NAMES
        ),
    }
    if require_online_parity:
        assert_online_parity(online_parity["policy"])
        assert_online_parity(online_parity["policy_wdl"])
    else:
        online_parity["parity_assert_skipped"] = True

    arch = architecture_summary(model)
    if fmt == FLOAT32_FORMAT:
        quantization = {
            "format": FLOAT32_FORMAT,
            "engine": "none",
            "runtime": FLOAT32_RUNTIME,
            "group_norm_float_fallback": False,
            "float_to_export_mae": parity,
            "online_float_to_export_mae": online_parity,
            "host": platform.platform(),
        }
    else:
        quantization = {
            "format": f"static_ptq_fx_{engine}_int8",
            "engine": engine,
            "runtime": f"torch.jit.trace+fx_static_{engine}",
            "group_norm_float_fallback": True,
            "float_to_export_mae": parity,
            "online_float_to_export_mae": online_parity,
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
    manifest["online_entry_points"] = {
        "policy": POLICY_ARTIFACT_FILE,
        "policy_wdl": POLICY_WDL_ARTIFACT_FILE,
    }
    # Digests for online artifacts (full weights_sha256 already in build_manifest).
    manifest["online_weights_sha256"] = {
        "policy": sha256_file(policy_path),
        "policy_wdl": sha256_file(policy_wdl_path),
    }
    manifest_path = output_dir / MANIFEST_NAME
    write_manifest(manifest_path, manifest)

    return ExportResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        policy_artifact_path=policy_path,
        policy_wdl_artifact_path=policy_wdl_path,
        qengine=engine,
        float_to_export_mae=parity,
        online_float_to_export_mae=online_parity,
    )


def export_static_int8(
    model: MorpheusNet,
    output_dir: Path,
    *,
    qengine: Optional[str] = None,
    training_run: Optional[dict[str, Any]] = None,
    example_batch: int = 1,
    seed: int = 0,
    require_online_parity: bool = True,
) -> ExportResult:
    """FX static PTQ export with full + online policy / policy+WDL entry points."""
    return _export_artifacts(
        model,
        output_dir,
        fmt="int8",
        qengine=qengine,
        training_run=training_run,
        example_batch=example_batch,
        seed=seed,
        require_online_parity=require_online_parity,
    )


def export_float32(
    model: MorpheusNet,
    output_dir: Path,
    *,
    training_run: Optional[dict[str, Any]] = None,
    example_batch: int = 1,
    seed: int = 0,
    require_online_parity: bool = True,
) -> ExportResult:
    """Float TorchScript export — bit-faithful weights, no calibration step."""
    return _export_artifacts(
        model,
        output_dir,
        fmt=FLOAT32_FORMAT,
        qengine=None,
        training_run=training_run,
        example_batch=example_batch,
        seed=seed,
        require_online_parity=require_online_parity,
    )


def export_from_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    fmt: str = FLOAT32_FORMAT,
    qengine: Optional[str] = None,
    training_run: Optional[dict[str, Any]] = None,
    require_online_parity: bool = True,
) -> ExportResult:
    checkpoint = load_checkpoint_state(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint)
    meta = checkpoint.get("meta", {})
    seed = int(meta.get("seed", 0))
    return _export_artifacts(
        model,
        output_dir,
        fmt=fmt,
        qengine=qengine,
        training_run=training_run,
        example_batch=1,
        seed=seed,
        require_online_parity=require_online_parity,
    )


def forward_exported(
    session: torch.jit.ScriptModule,
    x: torch.Tensor,
) -> MorpheusOutput:
    """Run a traced full export and rebuild the named output tuple."""
    outputs = session(x)
    if len(outputs) != len(EXPORT_OUTPUT_NAMES):
        raise ValueError(
            f"export returned {len(outputs)} tensors; expected {len(EXPORT_OUTPUT_NAMES)}"
        )
    return MorpheusOutput(*outputs)


def forward_policy_exported(
    session: torch.jit.ScriptModule,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a traced policy-only online export."""
    outputs = session(x)
    if len(outputs) != len(POLICY_OUTPUT_NAMES):
        raise ValueError(
            f"policy export returned {len(outputs)} tensors; "
            f"expected {len(POLICY_OUTPUT_NAMES)}"
        )
    return outputs[0], outputs[1]


def forward_policy_wdl_exported(
    session: torch.jit.ScriptModule,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a traced policy+WDL online export."""
    outputs = session(x)
    if len(outputs) != len(POLICY_WDL_OUTPUT_NAMES):
        raise ValueError(
            f"policy_wdl export returned {len(outputs)} tensors; "
            f"expected {len(POLICY_WDL_OUTPUT_NAMES)}"
        )
    return outputs[0], outputs[1], outputs[2]
