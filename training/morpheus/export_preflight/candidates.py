"""Export candidates limited to competition-sandbox packages."""
from __future__ import annotations

import json
import platform
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from safetensors.torch import save_file

from training.morpheus.export_preflight.fixtures import (
    BATCH_SHAPES,
    all_batch_fixtures,
    make_input,
)
from training.morpheus.export_preflight.model import make_probe


def _make_probe_sized(
    *,
    seed: int,
    n_blocks: int,
    trunk_channels: int | None,
    expansion: int | None,
):
    kwargs = {}
    if trunk_channels is not None:
        kwargs["trunk_channels"] = trunk_channels
    if expansion is not None:
        kwargs["expansion"] = expansion
    return make_probe(seed=seed, n_blocks=n_blocks, **kwargs)
from morpheus.export import (
    EXPORT_OUTPUT_NAMES,
    MorpheusExportWrapper,
    expected_export_shapes,
)
from morpheus.network import MorpheusOutput


def _forward_export_outputs(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    out = model(x)
    if isinstance(out, MorpheusOutput):
        return tuple(out)
    return tuple(out)

# Soft bound matching the competition process memory ceiling.
RSS_BOUND_BYTES = 2 * 1024 * 1024 * 1024


def _supported_qengines() -> list[str]:
    try:
        return list(torch.backends.quantized.supported_engines)
    except Exception:  # noqa: BLE001 — probe only
        return []


def _set_qengine(engine: str) -> None:
    torch.backends.quantized.engine = engine


def _module_type_inventory(module: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mod in module.modules():
        name = f"{type(mod).__module__}.{type(mod).__name__}"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _is_quantized_module(mod: nn.Module) -> bool:
    return type(mod).__module__.startswith("torch.ao.nn.quantized")


def _collect_fallbacks(module: nn.Module) -> list[str]:
    """Operators that remain float inside an otherwise quantized graph."""
    float_ops: list[str] = []
    for name, mod in module.named_modules():
        if isinstance(mod, nn.GroupNorm):
            float_ops.append(f"GroupNorm:{name or '<root>'}")
        elif isinstance(mod, nn.Conv2d) and not _is_quantized_module(mod):
            float_ops.append(f"Conv2d(float):{name or '<root>'}")
        elif isinstance(mod, nn.Linear) and not _is_quantized_module(mod):
            float_ops.append(f"Linear(float):{name or '<root>'}")
    return float_ops


def _has_static_8bit_ops(module: nn.Module) -> bool:
    for mod in module.modules():
        if _is_quantized_module(mod) and type(mod).__name__ in {
            "Conv2d",
            "Conv1d",
            "Conv3d",
            "Linear",
        }:
            return True
        weight = getattr(mod, "weight", None)
        if torch.is_tensor(weight) and weight.dtype in (torch.qint8, torch.quint8):
            return True
    return False


def _outputs_to_cpu(
    outputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    return tuple(o.detach().cpu() for o in outputs)


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


def _peak_rss_bytes() -> int:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    if platform.system() == "Darwin":
        return int(usage)
    return int(usage) * 1024


def _latency_ms(
    run: Callable[[], Any],
    *,
    warmup: int = 1,
    reps: int = 3,
) -> dict[str, float]:
    for _ in range(warmup):
        run()
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "p50_ms": samples[len(samples) // 2],
        "mean_ms": sum(samples) / len(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "reps": float(reps),
    }


def _run_batches(
    forward: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    *,
    seed: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for fixture in all_batch_fixtures(seed=seed):
        out = _outputs_to_cpu(forward(fixture.tensor))
        shapes = [list(o.shape) for o in out]
        lat = _latency_ms(lambda t=fixture.tensor: forward(t))
        results[fixture.name] = {
            "batch": fixture.batch,
            "output_shapes": shapes,
            "latency": lat,
            "ok": shapes == expected_export_shapes(fixture.batch),
        }
    return results


def _serialize_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _candidate_base(
    name: str,
    *,
    runtime: str,
    quantization_format: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "runtime": runtime,
        "quantization_format": quantization_format,
        "sandbox_packages_only": True,
        "exported": False,
        "reloaded": False,
        "executed": False,
        "has_static_8bit_ops": False,
        "unsupported_operators": [],
        "fallback_operators": [],
        "serialized_bytes": None,
        "peak_rss_bytes": None,
        "rss_bound_bytes": RSS_BOUND_BYTES,
        "memory_bounded": False,
        "float_to_export_error": None,
        "batch_results": {},
        "module_inventory": {},
        "error": None,
        "viable": False,
        "notes": [],
    }


def try_torch_fx_static(
    *,
    engine: str,
    seed: int,
    n_blocks: int,
    trunk_channels: int | None = None,
    expansion: int | None = None,
    work_dir: Path,
) -> dict[str, Any]:
    """FX graph-mode static quantization with a sandbox torch build."""
    result = _candidate_base(
        f"torch_fx_static_{engine}",
        runtime=f"torch.ao.quantization.quantize_fx+{engine}",
        quantization_format=f"static_ptq_fx_{engine}_int8",
    )
    supported = _supported_qengines()
    if engine not in supported:
        result["error"] = (
            f"quantized engine {engine!r} is not in supported_engines={supported}"
        )
        result["unsupported_operators"].append(f"qengine:{engine}")
        result["notes"].append(
            "Engine unavailable on this host; the Linux judge image may differ."
        )
        return result

    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    except ImportError as exc:
        result["error"] = f"quantize_fx unavailable: {exc}"
        result["sandbox_packages_only"] = False
        return result

    float_model = _make_probe_sized(
        seed=seed, n_blocks=n_blocks, trunk_channels=trunk_channels, expansion=expansion
    )
    wrapper = MorpheusExportWrapper(float_model)
    example = make_input(1, seed=seed)
    with torch.no_grad():
        float_ref = _outputs_to_cpu(_forward_export_outputs(float_model, example))

    try:
        _set_qengine(engine)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            qconfig_mapping = get_default_qconfig_mapping(engine)
            prepared = prepare_fx(
                wrapper,
                qconfig_mapping,
                example_inputs=(example,),
            )
            with torch.no_grad():
                for batch in BATCH_SHAPES:
                    prepared(make_input(batch, seed=seed + batch))
            quantized = convert_fx(prepared)

        result["exported"] = True
        result["module_inventory"] = _module_type_inventory(quantized)
        result["fallback_operators"] = _collect_fallbacks(quantized)
        result["has_static_8bit_ops"] = _has_static_8bit_ops(quantized)
        if any(
            key.startswith("torch.nn.modules.normalization.GroupNorm")
            for key in result["module_inventory"]
        ):
            result["notes"].append(
                "GroupNorm stays float between quantized Conv layers "
                "(dequant → GroupNorm → quant). Reported, not hidden."
            )

        artifact = work_dir / f"fx_static_{engine}.pt"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            # FX GraphModules often fail torch.jit.script on newer torch
            # (unknown _torch_Tensor_ annotations). Trace preserves the
            # quantized ops and accepts the required batch shapes.
            with torch.no_grad():
                exported = torch.jit.trace(
                    quantized, example, check_trace=False, strict=False
                )
            torch.jit.save(exported, str(artifact))
        result["serialized_bytes"] = _serialize_size(artifact)
        result["notes"].append("Serialized via torch.jit.trace (script rejected FX annotations).")

        loaded = torch.jit.load(str(artifact))
        loaded.eval()
        result["reloaded"] = True

        def forward(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            with torch.no_grad():
                return _outputs_to_cpu(_forward_export_outputs(loaded, x))

        result["batch_results"] = _run_batches(forward, seed=seed)
        result["executed"] = all(
            row.get("ok") for row in result["batch_results"].values()
        )
        with torch.no_grad():
            export_out = forward(example)
        result["float_to_export_error"] = _parity_errors(float_ref, export_out)
        result["peak_rss_bytes"] = _peak_rss_bytes()
        result["memory_bounded"] = result["peak_rss_bytes"] < RSS_BOUND_BYTES
        result["viable"] = bool(
            result["exported"]
            and result["reloaded"]
            and result["executed"]
            and result["has_static_8bit_ops"]
            and result["memory_bounded"]
            and result["float_to_export_error"] is not None
            and result["sandbox_packages_only"]
        )
    except Exception as exc:  # noqa: BLE001 — candidate probe
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def try_torch_jit_float(
    *,
    seed: int,
    n_blocks: int,
    trunk_channels: int | None = None,
    expansion: int | None = None,
    work_dir: Path,
) -> dict[str, Any]:
    """Float TorchScript control path — not a static 8-bit candidate."""
    result = _candidate_base(
        "torch_jit_float32",
        runtime="torch.jit.script",
        quantization_format="float32",
    )
    result["notes"].append(
        "Control path only. Float weights do not satisfy static 8-bit deployment."
    )
    float_model = _make_probe_sized(
        seed=seed, n_blocks=n_blocks, trunk_channels=trunk_channels, expansion=expansion
    )
    wrapper = MorpheusExportWrapper(float_model)
    example = make_input(1, seed=seed)
    with torch.no_grad():
        float_ref = _outputs_to_cpu(_forward_export_outputs(float_model, example))
    try:
        artifact = work_dir / "float_script.pt"
        scripted = torch.jit.script(wrapper)
        torch.jit.save(scripted, str(artifact))
        result["exported"] = True
        result["serialized_bytes"] = _serialize_size(artifact)
        loaded = torch.jit.load(str(artifact))
        loaded.eval()
        result["reloaded"] = True
        result["module_inventory"] = _module_type_inventory(float_model)
        result["has_static_8bit_ops"] = False

        def forward(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            with torch.no_grad():
                return _outputs_to_cpu(_forward_export_outputs(loaded, x))

        result["batch_results"] = _run_batches(forward, seed=seed)
        result["executed"] = all(
            row.get("ok") for row in result["batch_results"].values()
        )
        result["float_to_export_error"] = _parity_errors(float_ref, forward(example))
        result["peak_rss_bytes"] = _peak_rss_bytes()
        result["memory_bounded"] = result["peak_rss_bytes"] < RSS_BOUND_BYTES
        result["viable"] = False
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def try_safetensors_weight_only_int8(
    *,
    seed: int,
    n_blocks: int,
    trunk_channels: int | None = None,
    expansion: int | None = None,
    work_dir: Path,
) -> dict[str, Any]:
    """Pack Conv/Linear weights as int8 tensors; run float after dequant.

    This path stores 8-bit weights but is not a static quantized graph. It is
    reported so Part 04 does not confuse weight packing with static int8 ops.
    """
    result = _candidate_base(
        "safetensors_weight_only_int8",
        runtime="safetensors+torch_float_dequant",
        quantization_format="weight_only_int8_affine",
    )
    result["notes"].append(
        "Weight-only int8 with float runtime dequant. Not a static 8-bit graph."
    )
    result["fallback_operators"].append("entire_forward:float32_after_dequant")
    float_model = _make_probe_sized(
        seed=seed, n_blocks=n_blocks, trunk_channels=trunk_channels, expansion=expansion
    )
    example = make_input(1, seed=seed)
    with torch.no_grad():
        float_ref = _outputs_to_cpu(_forward_export_outputs(float_model, example))
    try:
        artifact_dir = work_dir / "weight_only_int8"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        scales: dict[str, float] = {}
        packed: dict[str, torch.Tensor] = {}
        for name, tensor in float_model.state_dict().items():
            if tensor.ndim >= 2 and tensor.dtype == torch.float32:
                max_abs = float(tensor.detach().abs().max().item()) or 1.0
                scale = max_abs / 127.0
                q = torch.clamp(
                    torch.round(tensor.detach() / scale), -128, 127
                ).to(torch.int8)
                packed[name] = q.contiguous()
                scales[name] = scale
            else:
                packed[name] = tensor.detach().contiguous()
        save_file(packed, str(artifact_dir / "weights.safetensors"))
        (artifact_dir / "scales.json").write_text(
            json.dumps(scales, indent=2, sort_keys=True) + "\n"
        )
        result["exported"] = True
        result["serialized_bytes"] = _serialize_size(artifact_dir)

        # Reload into a fresh float model (dequant).
        restored = _make_probe_sized(
            seed=seed ^ 0xABCDEF,
            n_blocks=n_blocks,
            trunk_channels=trunk_channels,
            expansion=expansion,
        )
        from safetensors.torch import load_file

        loaded_tensors = load_file(str(artifact_dir / "weights.safetensors"))
        loaded_scales = json.loads((artifact_dir / "scales.json").read_text())
        state: dict[str, torch.Tensor] = {}
        for name, tensor in loaded_tensors.items():
            if name in loaded_scales:
                state[name] = tensor.float() * float(loaded_scales[name])
            else:
                state[name] = tensor
        restored.load_state_dict(state)
        restored.eval()
        result["reloaded"] = True
        result["has_static_8bit_ops"] = False
        result["module_inventory"] = _module_type_inventory(restored)

        def forward(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            with torch.no_grad():
                return _outputs_to_cpu(_forward_export_outputs(restored, x))

        result["batch_results"] = _run_batches(forward, seed=seed)
        result["executed"] = all(
            row.get("ok") for row in result["batch_results"].values()
        )
        result["float_to_export_error"] = _parity_errors(float_ref, forward(example))
        result["peak_rss_bytes"] = _peak_rss_bytes()
        result["memory_bounded"] = result["peak_rss_bytes"] < RSS_BOUND_BYTES
        result["viable"] = False
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def probe_all_candidates(
    *,
    seed: int = 0,
    n_blocks: int = 12,
    trunk_channels: int | None = None,
    expansion: int | None = None,
    work_dir: Path | None = None,
) -> list[dict[str, Any]]:
    own_tmpdir = work_dir is None
    root = Path(tempfile.mkdtemp(prefix="morpheus-export-")) if own_tmpdir else work_dir
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)

    size = {"trunk_channels": trunk_channels, "expansion": expansion}
    candidates: list[dict[str, Any]] = []
    # Always attempt qnnpack (ARM/macOS and many mobile CPU builds).
    candidates.append(
        try_torch_fx_static(
            engine="qnnpack", seed=seed, n_blocks=n_blocks, work_dir=root, **size
        )
    )
    # Attempt x86 engines when present (Linux competition image).
    for engine in ("fbgemm", "x86"):
        candidates.append(
            try_torch_fx_static(
                engine=engine, seed=seed, n_blocks=n_blocks, work_dir=root, **size
            )
        )
    candidates.append(
        try_torch_jit_float(seed=seed, n_blocks=n_blocks, work_dir=root, **size)
    )
    candidates.append(
        try_safetensors_weight_only_int8(
            seed=seed, n_blocks=n_blocks, work_dir=root, **size
        )
    )
    return candidates
