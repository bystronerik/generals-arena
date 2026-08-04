"""Fully convolutional Morpheus residual network — policy, WDL, and auxiliary heads."""
from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from action import encode_action, N_CELLS
from schema import ARMY_BIN_EDGES, N_ARMY_BINS

IN_CHANNELS = 49
BOARD = 21
TRUNK_CHANNELS = 64
EXPANSION = 128
POLICY_CHANNELS = 9
N_BLOCKS = 12
GROUP_NORM_GROUPS = 8
DILATION_CYCLE = (1, 2, 4)
GLOBAL_FEAT_DIM = TRUNK_CHANNELS * 2


class MorpheusOutput(NamedTuple):
    policy: torch.Tensor
    pass_logit: torch.Tensor
    wdl_logits: torch.Tensor
    hidden_owner: torch.Tensor
    enemy_army_bins: torch.Tensor
    enemy_general: torch.Tensor
    hidden_castle: torch.Tensor
    land_margin: torch.Tensor
    army_margin: torch.Tensor
    castle_margin: torch.Tensor
    turns_to_termination: torch.Tensor


class InvertedResidual(nn.Module):
    """64 → 128 → 64 inverted residual with depthwise dilated 3×3 and GroupNorm."""

    def __init__(self, dilation: int) -> None:
        super().__init__()
        mid = EXPANSION
        ch = TRUNK_CHANNELS
        self.pw_expand = nn.Conv2d(ch, mid, kernel_size=1, bias=False)
        self.gn_expand = nn.GroupNorm(GROUP_NORM_GROUPS, mid)
        self.dw = nn.Conv2d(
            mid,
            mid,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=mid,
            bias=False,
        )
        self.gn_dw = nn.GroupNorm(GROUP_NORM_GROUPS, mid)
        self.pw_project = nn.Conv2d(mid, ch, kernel_size=1, bias=False)
        self.gn_project = nn.GroupNorm(GROUP_NORM_GROUPS, ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.gn_expand(self.pw_expand(x)))
        y = self.act(self.gn_dw(self.dw(y)))
        y = self.gn_project(self.pw_project(y))
        return self.act(x + y)


def masked_global_features(trunk: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked global mean and max of ``trunk``; ``mask`` is ``N×1×H×W``."""
    m = mask.to(dtype=torch.float32)
    denom = m.sum(dim=(2, 3))
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    mean = (trunk * m).sum(dim=(2, 3)) / safe_denom
    masked = trunk.masked_fill(m < 0.5, -1e9)
    mx = masked.amax(dim=(2, 3))
    mx = torch.where(denom > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=1)


class MorpheusNet(nn.Module):
    """Shared-weight trunk with policy, WDL, hidden-state, margin, and termination heads."""

    def __init__(self, n_blocks: int = N_BLOCKS) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        self.stem = nn.Conv2d(
            IN_CHANNELS, TRUNK_CHANNELS, kernel_size=3, padding=1, bias=False
        )
        self.gn_stem = nn.GroupNorm(GROUP_NORM_GROUPS, TRUNK_CHANNELS)
        self.act = nn.ReLU6(inplace=True)
        dilations = [DILATION_CYCLE[i % len(DILATION_CYCLE)] for i in range(n_blocks)]
        self.blocks = nn.ModuleList([InvertedResidual(dilation=d) for d in dilations])

        self.policy = nn.Conv2d(TRUNK_CHANNELS, POLICY_CHANNELS, kernel_size=1, bias=True)
        self.pass_fc = nn.Linear(GLOBAL_FEAT_DIM, 1)
        self.wdl = nn.Linear(GLOBAL_FEAT_DIM, 3)

        self.hidden_owner = nn.Conv2d(TRUNK_CHANNELS, 1, kernel_size=1, bias=True)
        self.enemy_army_bins = nn.Conv2d(
            TRUNK_CHANNELS, N_ARMY_BINS, kernel_size=1, bias=True
        )
        self.enemy_general = nn.Conv2d(TRUNK_CHANNELS, 1, kernel_size=1, bias=True)
        self.hidden_castle = nn.Conv2d(TRUNK_CHANNELS, 1, kernel_size=1, bias=True)

        self.land_margin = nn.Linear(GLOBAL_FEAT_DIM, 1)
        self.army_margin = nn.Linear(GLOBAL_FEAT_DIM, 1)
        self.castle_margin = nn.Linear(GLOBAL_FEAT_DIM, 1)
        self.turns_to_termination = nn.Linear(GLOBAL_FEAT_DIM, 1)

    def trunk_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = x.narrow(1, 0, 1)
        h = self.act(self.gn_stem(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        return h, mask

    def forward_policy(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy + pass only — belief proposal and enemy-prior online path."""
        h, mask = self.trunk_forward(x)
        feat = masked_global_features(h, mask)
        return self.policy(h), self.pass_fc(feat)

    def forward_policy_wdl(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy + pass + WDL — root and leaf online path (no auxiliary heads)."""
        h, mask = self.trunk_forward(x)
        feat = masked_global_features(h, mask)
        return self.policy(h), self.pass_fc(feat), self.wdl(feat)

    def forward(self, x: torch.Tensor) -> MorpheusOutput:
        h, mask = self.trunk_forward(x)
        feat = masked_global_features(h, mask)
        return MorpheusOutput(
            policy=self.policy(h),
            pass_logit=self.pass_fc(feat),
            wdl_logits=self.wdl(feat),
            hidden_owner=self.hidden_owner(h),
            enemy_army_bins=self.enemy_army_bins(h),
            enemy_general=self.enemy_general(h),
            hidden_castle=self.hidden_castle(h),
            land_margin=self.land_margin(feat),
            army_margin=self.army_margin(feat),
            castle_margin=self.castle_margin(feat),
            turns_to_termination=self.turns_to_termination(feat),
        )


def make_model(*, seed: int = 0, n_blocks: int = N_BLOCKS) -> MorpheusNet:
    torch.manual_seed(seed)
    model = MorpheusNet(n_blocks=n_blocks)
    model.eval()
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def architecture_summary(model: MorpheusNet) -> dict[str, Any]:
    return {
        "in_channels": IN_CHANNELS,
        "board": BOARD,
        "trunk_channels": TRUNK_CHANNELS,
        "expansion": EXPANSION,
        "policy_channels": POLICY_CHANNELS,
        "n_blocks": len(model.blocks),
        "n_army_bins": N_ARMY_BINS,
        "army_bin_edges": list(ARMY_BIN_EDGES),
        "group_norm_groups": GROUP_NORM_GROUPS,
        "dilation_cycle": list(DILATION_CYCLE),
        "parameter_count": parameter_count(model),
        "outputs": {
            "policy": f"N×{POLICY_CHANNELS}×{BOARD}×{BOARD}",
            "pass_logit": "N×1",
            "wdl_logits": "N×3",
            "hidden_owner": f"N×1×{BOARD}×{BOARD}",
            "enemy_army_bins": f"N×{N_ARMY_BINS}×{BOARD}×{BOARD}",
            "enemy_general": f"N×1×{BOARD}×{BOARD}",
            "hidden_castle": f"N×1×{BOARD}×{BOARD}",
            "land_margin": "N×1",
            "army_margin": "N×1",
            "castle_margin": "N×1",
            "turns_to_termination": "N×1",
        },
    }


def flatten_policy_logits(policy: torch.Tensor, pass_logit: torch.Tensor) -> torch.Tensor:
    """``N×9×21×21`` spatial policy + pass → ``N×3970``."""
    flat = policy.reshape(policy.shape[0], -1)
    return torch.cat([flat, pass_logit], dim=1)


def wdl_value(wdl_logits: torch.Tensor) -> torch.Tensor:
    """``V = p_win - p_loss`` from perspective-player WDL logits."""
    probs = F.softmax(wdl_logits, dim=-1)
    return probs[:, 0] - probs[:, 2]


def legal_normalized_policy(
    policy: torch.Tensor,
    pass_logit: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax over legal actions only; illegal logits stay zero."""
    logits = flatten_policy_logits(policy, pass_logit)
    mask = legal_mask.to(dtype=torch.bool, device=logits.device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(logits.shape[0], -1)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    dist = F.softmax(masked_logits, dim=-1)
    return dist * mask.to(dtype=dist.dtype)


def backup_value(wdl_logits: torch.Tensor, *, from_root: bool) -> torch.Tensor:
    """Search backup: root keeps ``V``; enemy perspective negates ``V``."""
    v = wdl_value(wdl_logits)
    return v if from_root else -v


def wdl_logits_swap_seats(wdl_logits: torch.Tensor) -> torch.Tensor:
    """Swap win/loss channels for the opposite seat perspective."""
    return wdl_logits[:, [2, 1, 0]]


def transform_policy_spatial(
    policy: torch.Tensor,
    sym_name: str,
) -> torch.Tensor:
    """Remap ``N×9×21×21`` policy channels under a board symmetry."""
    from symmetry import get_symmetry

    sym = get_symmetry(sym_name)
    batch = policy.shape[0]
    spatial = 9 * N_CELLS
    out = torch.zeros_like(policy)
    for b in range(batch):
        flat = policy[b].reshape(-1).cpu().numpy()
        from action import decode_action

        remapped = np.zeros(spatial, dtype=flat.dtype)
        for idx in range(spatial):
            action = decode_action(idx)
            from symmetry import transform_action_tuple

            new_action = transform_action_tuple(action, sym)
            remapped[encode_action(new_action)] = flat[idx]
        out[b] = torch.from_numpy(remapped).to(policy.device).reshape(9, BOARD, BOARD)
    return out


def transform_spatial_head(head: torch.Tensor, sym_name: str) -> torch.Tensor:
    """Remap ``N×C×21×21`` auxiliary spatial head under a board symmetry."""
    from symmetry import get_symmetry, transform_plane

    sym = get_symmetry(sym_name)
    batch, channels = head.shape[0], head.shape[1]
    out = torch.zeros_like(head)
    for b in range(batch):
        for c in range(channels):
            plane = head[b, c].cpu().numpy()
            out[b, c] = torch.from_numpy(transform_plane(plane, sym)).to(head.device)
    return out


def policy_symmetry_equivariant(
    model: MorpheusNet,
    tensor_np: np.ndarray,
    sym_name: str,
    *,
    device: Optional[torch.device] = None,
    atol: float = 1e-4,
) -> bool:
    """``f(Tx)`` policy matches ``T f(x)`` for spatial symmetries."""
    from symmetry import get_symmetry, transform_tensor

    sym = get_symmetry(sym_name)
    dev = device or torch.device("cpu")
    x = torch.from_numpy(tensor_np).unsqueeze(0).to(dev)
    xt = transform_tensor(tensor_np, sym)
    xt_t = torch.from_numpy(xt).unsqueeze(0).to(dev)
    with torch.no_grad():
        p0 = model(x).policy
        p1 = model(xt_t).policy
    pred = transform_policy_spatial(p0, sym_name)
    return bool(torch.allclose(p1, pred, atol=atol, rtol=1e-3))
