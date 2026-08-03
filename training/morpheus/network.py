"""Re-export the shared Morpheus network from the bot closure for training."""
from __future__ import annotations

from morpheus.network import (
    BOARD,
    DILATION_CYCLE,
    EXPANSION,
    GLOBAL_FEAT_DIM,
    GROUP_NORM_GROUPS,
    IN_CHANNELS,
    N_ARMY_BINS,
    N_BLOCKS,
    POLICY_CHANNELS,
    TRUNK_CHANNELS,
    InvertedResidual,
    MorpheusNet,
    MorpheusOutput,
    architecture_summary,
    make_model,
    parameter_count,
)
from morpheus.schema import ARMY_BIN_EDGES, ARMY_SCALE_DEFAULT

# Part 00b alias — training preflight still calls ``make_probe``.
ProbeNet = MorpheusNet
make_probe = make_model

__all__ = [
    "BOARD",
    "DILATION_CYCLE",
    "EXPANSION",
    "GLOBAL_FEAT_DIM",
    "GROUP_NORM_GROUPS",
    "IN_CHANNELS",
    "N_ARMY_BINS",
    "N_BLOCKS",
    "POLICY_CHANNELS",
    "TRUNK_CHANNELS",
    "InvertedResidual",
    "MorpheusNet",
    "MorpheusOutput",
    "ProbeNet",
    "ARMY_BIN_EDGES",
    "ARMY_SCALE_DEFAULT",
    "architecture_summary",
    "make_model",
    "make_probe",
    "parameter_count",
]
