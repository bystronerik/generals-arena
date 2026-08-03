"""Random-weight probe of the Morpheus conv / GroupNorm / policy / WDL path."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

IN_CHANNELS = 49
BOARD = 21
TRUNK_CHANNELS = 64
EXPANSION = 128
POLICY_CHANNELS = 9
N_BLOCKS = 12
GROUP_NORM_GROUPS = 8
DILATION_CYCLE = (1, 2, 4)


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


class ProbeNet(nn.Module):
    """Static-shape probe: stem, residual trunk, spatial policy, pass, WDL.

    Auxiliary heads are out of scope for Part 00b. Part 04 owns the full model.
    """

    def __init__(self, n_blocks: int = N_BLOCKS) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        self.stem = nn.Conv2d(
            IN_CHANNELS, TRUNK_CHANNELS, kernel_size=3, padding=1, bias=False
        )
        self.gn_stem = nn.GroupNorm(GROUP_NORM_GROUPS, TRUNK_CHANNELS)
        self.act = nn.ReLU6(inplace=True)
        dilations = [
            DILATION_CYCLE[i % len(DILATION_CYCLE)] for i in range(n_blocks)
        ]
        self.blocks = nn.ModuleList(
            [InvertedResidual(dilation=d) for d in dilations]
        )
        self.policy = nn.Conv2d(
            TRUNK_CHANNELS, POLICY_CHANNELS, kernel_size=1, bias=True
        )
        # Masked global mean and max of the trunk → pass and WDL.
        self.pass_fc = nn.Linear(TRUNK_CHANNELS * 2, 1)
        self.wdl = nn.Linear(TRUNK_CHANNELS * 2, 3)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.act(self.gn_stem(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        policy = self.policy(h)
        mean = h.mean(dim=(2, 3))
        mx = h.amax(dim=(2, 3))
        feat = torch.cat([mean, mx], dim=1)
        return policy, self.pass_fc(feat), self.wdl(feat)


def make_probe(
    *,
    seed: int = 0,
    n_blocks: int = N_BLOCKS,
) -> ProbeNet:
    torch.manual_seed(seed)
    model = ProbeNet(n_blocks=n_blocks)
    model.eval()
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def architecture_summary(model: ProbeNet) -> dict[str, Any]:
    return {
        "in_channels": IN_CHANNELS,
        "board": BOARD,
        "trunk_channels": TRUNK_CHANNELS,
        "expansion": EXPANSION,
        "policy_channels": POLICY_CHANNELS,
        "n_blocks": len(model.blocks),
        "group_norm_groups": GROUP_NORM_GROUPS,
        "dilation_cycle": list(DILATION_CYCLE),
        "parameter_count": parameter_count(model),
        "outputs": {
            "policy": f"N×{POLICY_CHANNELS}×{BOARD}×{BOARD}",
            "pass_logit": "N×1",
            "wdl_logits": "N×3",
        },
    }
