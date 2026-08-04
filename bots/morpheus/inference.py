"""Load versioned Morpheus static 8-bit exports for match-time inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from export import (
    DEFAULT_ARTIFACT_FILE,
    MANIFEST_NAME,
    POLICY_ARTIFACT_FILE,
    POLICY_WDL_ARTIFACT_FILE,
    forward_exported,
    forward_policy_exported,
    forward_policy_wdl_exported,
)
from network import BOARD, IN_CHANNELS, N_ARMY_BINS, N_BLOCKS, MorpheusOutput
from schema import (
    ACTION_SCHEMA_VERSION,
    ARCHITECTURE_VERSION,
    ARMY_BIN_EDGES,
    MANIFEST_VERSION,
    TENSOR_SCHEMA_VERSION,
    read_manifest,
    sha256_file,
)

ARTIFACT_DIRNAME = "artifact"


@dataclass
class InferenceSession:
    """Loaded TorchScript session plus manifest metadata.

    Online search uses dedicated entry points:
    - ``forward_policy`` for belief proposal and enemy priors;
    - ``forward_policy_wdl`` for root and leaf evaluation.

    ``forward`` keeps the full-head artifact for recovery and diagnostics.
    """

    manifest: dict[str, Any]
    module: torch.jit.ScriptModule
    device: torch.device
    policy_module: Optional[torch.jit.ScriptModule] = None
    policy_wdl_module: Optional[torch.jit.ScriptModule] = None

    def forward(self, x: torch.Tensor) -> MorpheusOutput:
        self.module.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return forward_exported(self.module, x)

    def forward_policy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy + pass only. Does not evaluate auxiliary heads."""
        x = x.to(self.device)
        with torch.no_grad():
            if self.policy_module is not None:
                self.policy_module.eval()
                return forward_policy_exported(self.policy_module, x)
            # Legacy artifact fallback: use full export, drop aux outputs.
            out = self.forward(x)
            return out.policy, out.pass_logit

    def forward_policy_wdl(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy + pass + WDL. Does not evaluate auxiliary heads."""
        x = x.to(self.device)
        with torch.no_grad():
            if self.policy_wdl_module is not None:
                self.policy_wdl_module.eval()
                return forward_policy_wdl_exported(self.policy_wdl_module, x)
            out = self.forward(x)
            return out.policy, out.pass_logit, out.wdl_logits


def default_artifact_dir(bot_dir: Optional[Path] = None) -> Path:
    base = bot_dir or Path(__file__).resolve().parent
    return base / ARTIFACT_DIRNAME


def validate_manifest(manifest: dict[str, Any], artifact_path: Path) -> None:
    required = (
        "manifest_version",
        "tensor_schema",
        "action_schema",
        "architecture_version",
        "architecture",
        "quantization",
        "weights_sha256",
        "army_bin_edges",
        "artifact_file",
        "runtime",
    )
    for key in required:
        if key not in manifest:
            raise ValueError(f"manifest missing {key!r}")
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ValueError(
            f"manifest version mismatch: {manifest['manifest_version']!r} "
            f"!= {MANIFEST_VERSION!r}"
        )
    if manifest["tensor_schema"] != TENSOR_SCHEMA_VERSION:
        raise ValueError(
            f"tensor schema mismatch: {manifest['tensor_schema']} != {TENSOR_SCHEMA_VERSION}"
        )
    if manifest["action_schema"] != ACTION_SCHEMA_VERSION:
        raise ValueError(
            f"action schema mismatch: {manifest['action_schema']} != {ACTION_SCHEMA_VERSION}"
        )
    if manifest["architecture_version"] != ARCHITECTURE_VERSION:
        raise ValueError(
            f"architecture mismatch: {manifest['architecture_version']} != {ARCHITECTURE_VERSION}"
        )

    arch = manifest["architecture"]
    if not isinstance(arch, dict):
        raise ValueError("manifest architecture must be an object")
    expected_arch = {
        "in_channels": IN_CHANNELS,
        "board": BOARD,
        "n_army_bins": N_ARMY_BINS,
        "n_blocks": N_BLOCKS,
    }
    for key, expected in expected_arch.items():
        if key not in arch:
            raise ValueError(f"manifest architecture missing {key!r}")
        if arch[key] != expected:
            raise ValueError(
                f"manifest architecture {key} mismatch: {arch[key]!r} != {expected!r}"
            )

    quant = manifest["quantization"]
    if not isinstance(quant, dict):
        raise ValueError("manifest quantization must be an object")
    if "engine" not in quant:
        raise ValueError("manifest quantization missing 'engine'")
    if "format" not in quant:
        raise ValueError("manifest quantization missing 'format'")

    edges = tuple(float(x) for x in manifest["army_bin_edges"])
    if edges != ARMY_BIN_EDGES:
        raise ValueError("army_bin_edges in manifest do not match schema ARMY_BIN_EDGES")

    digest = sha256_file(artifact_path)
    if digest != manifest["weights_sha256"]:
        raise ValueError(
            f"weights SHA-256 mismatch: manifest {manifest['weights_sha256']} != file {digest}"
        )


def _require_qengine(engine: str) -> None:
    supported = list(torch.backends.quantized.supported_engines)
    if engine not in supported:
        raise ValueError(
            f"quantized engine {engine!r} not in supported_engines={supported}; "
            "re-export the artifact on a host that provides this engine, "
            "or export with an engine available on the judge image"
        )
    torch.backends.quantized.engine = engine


def _load_online_module(
    artifact_dir: Path,
    *,
    filename: str,
    expected_digest: Optional[str],
    device: torch.device,
) -> torch.jit.ScriptModule:
    path = artifact_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing online entry artifact: {path}")
    if expected_digest is not None:
        digest = sha256_file(path)
        if digest != expected_digest:
            raise ValueError(
                f"online weights SHA-256 mismatch for {filename}: "
                f"manifest {expected_digest} != file {digest}"
            )
    module = torch.jit.load(str(path), map_location=device)
    module.eval()
    return module


def load_session(
    artifact_dir: Path,
    *,
    device: Optional[torch.device] = None,
) -> InferenceSession:
    """Load ``manifest.json`` and traced models from ``artifact_dir``."""
    manifest_path = artifact_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = read_manifest(manifest_path)
    artifact_name = manifest.get("artifact_file", DEFAULT_ARTIFACT_FILE)
    artifact_path = artifact_dir / artifact_name
    if not artifact_path.is_file():
        raise FileNotFoundError(f"missing artifact: {artifact_path}")
    validate_manifest(manifest, artifact_path)
    _require_qengine(manifest["quantization"]["engine"])
    dev = device or torch.device("cpu")
    module = torch.jit.load(str(artifact_path), map_location=dev)
    module.eval()

    entry_points = manifest.get("online_entry_points") or {}
    digests = manifest.get("online_weights_sha256") or {}
    policy_module = None
    policy_wdl_module = None
    if entry_points:
        policy_name = str(entry_points.get("policy", POLICY_ARTIFACT_FILE))
        policy_wdl_name = str(entry_points.get("policy_wdl", POLICY_WDL_ARTIFACT_FILE))
        policy_module = _load_online_module(
            artifact_dir,
            filename=policy_name,
            expected_digest=digests.get("policy"),
            device=dev,
        )
        policy_wdl_module = _load_online_module(
            artifact_dir,
            filename=policy_wdl_name,
            expected_digest=digests.get("policy_wdl"),
            device=dev,
        )

    return InferenceSession(
        manifest=manifest,
        module=module,
        device=dev,
        policy_module=policy_module,
        policy_wdl_module=policy_wdl_module,
    )


def load_default_session(
    bot_dir: Optional[Path] = None,
    *,
    device: Optional[torch.device] = None,
) -> InferenceSession:
    return load_session(default_artifact_dir(bot_dir), device=device)


def warm_export_batches(session: InferenceSession, *, seed: int = 0) -> None:
    """Exercise root, leaf, and enemy-proposal batch shapes on online entry points."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    batches = (1, 4, 64)
    for batch in batches:
        x = torch.randn(batch, IN_CHANNELS, BOARD, BOARD, generator=gen)
        session.forward_policy(x)
        session.forward_policy_wdl(x)


def write_session_meta(session: InferenceSession, path: Path) -> None:
    path.write_text(json.dumps(session.manifest, indent=2, sort_keys=True) + "\n")
