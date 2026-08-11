"""Training configuration loaded from YAML with CLI overrides.

Ported from AverageJoe ``config.py`` with the plan's resolutions applied
(docs/research/strategies/averagejoe-competition-plan.md):

- D5: the vestigial keys (``reward_fn``, ``opponent``, ``population_*``,
  ``ckpt_pool_size``, ``magnet_policy``) are not ported. The reward is always
  sparse terminal win/loss.
- D2: no magnet keys — the entropy bonus is plain entropy.
- Competition env fields ``build_castles`` and ``deathtouch_turn`` are added;
  ``castle_val_*`` / ``num_cities_*`` are dropped (build_castles strips
  neutral castles, so they have no effect in this ruleset).
- ``hl_sigma`` defaults to the paper's 0.04, not the released 0.75 trap.
- The reference-Elo eval keys are not ported: no frozen reference checkpoints
  exist under competition rules. Exported bots are rated in the arena instead.
"""

from dataclasses import dataclass, fields
from typing import List, Optional

import yaml


@dataclass
class CurriculumStage:
    """A single curriculum stage. Advances to the next stage when eval
    win-rate vs random >= win_rate_threshold of the *next* stage (D4)."""
    min_generals_distance: int
    max_generals_distance: Optional[int] = None  # None = unbounded (preset)
    win_rate_threshold: float = 0.6


@dataclass
class Config:
    # Run
    run_name: str = "joe"

    # Environment — GeneralsEnv(mode="competition") numbers. The tests pin
    # these fields against _MODE_PRESETS["competition"] so they cannot drift.
    pad_to: int = 21
    min_grid_size: int = 18
    max_grid_size: int = 21
    truncation: int = 1200
    mountain_density_min: float = 0.24
    mountain_density_max: float = 0.26
    build_castles: bool = True
    deathtouch_turn: int = 800
    pool_size: int = 200_000
    reset_pool_every: int = 20

    # Network
    network: str = "history_transformer"
    init_checkpoint: str = ""  # .eqx with (network, opt_state) or network-only
    ema_checkpoint: str = ""   # .eqx with the EMA network (default: copy init)
    ema_decay: float = 0.999
    depth: int = 5
    embed_dim: int = 384
    n_head: int = 8
    ff_factor: int = 3
    patch_size: int = 3
    use_bf16: bool = True

    # Rollouts
    num_envs: int = 2048
    num_steps: int = 256
    num_iters: int = 50_000
    minibatch_size: int = 2048
    seed: int = 44

    # PPO
    num_epochs: int = 1
    gamma: float = 1.0
    gae_lambda: float = 0.9
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.267
    target_kl: Optional[float] = 0.02
    adv_top_frac: float = 0.25  # D3: fraction kept, ranked by |advantage|
    iteration_offset: int = 0   # shift schedules on checkpoint resumption

    # Entropy schedule (D2: plain entropy bonus)
    ent_schedule: str = "power_law"  # "power_law" or "linear"
    ent_coef_start: float = 0.05
    ent_power: float = 0.2           # start / (t+1)^power
    ent_coef_min: float = 0.001
    ent_coef_end: float = 0.001      # linear schedule only
    ent_coef_decay_iters: int = 5000

    # Learning rate
    lr: float = 1e-4
    lr_schedule: str = "power_law"   # "power_law" or "linear"
    lr_power_law_numerator: float = 0.5
    lr_power_law_exponent: float = 1.1
    lr_power_law_min: float = 5e-6
    lr_power_law_max: float = 1e-4
    final_lr: float = 0.0            # linear schedule only
    lr_decay_iters: int = 0

    # Value loss
    value_loss: str = "ce"     # "ce" (HL-Gauss categorical) or "mse"
    num_bins: int = 128
    v_min: float = -1.0
    v_max: float = 1.0
    hl_sigma: float = 0.04     # paper value; the released default 0.75 is a trap

    # Evaluation / checkpoints
    eval_every: int = 50
    eval_every_after: int = 0  # eval frequency on the last stage (0 = same)
    eval_games: int = 512
    ckpt_every: int = 500      # EMA-only checkpoint cadence
    save_every: int = 500      # full (network + optimizer + EMA) cadence
    debug: bool = False

    # Curriculum (list of stage dicts from YAML, or None)
    curriculum: Optional[list] = None

    @property
    def num_actions(self):
        # 10 per-cell channels: 4 full, 4 half, pass, build (plan section 4)
        return 10 * self.pad_to * self.pad_to

    @property
    def curriculum_stages(self) -> Optional[List[CurriculumStage]]:
        """Parse the YAML curriculum list into CurriculumStage objects."""
        if not self.curriculum:
            return None
        known = {f.name for f in fields(CurriculumStage)}
        return [CurriculumStage(**{k: v for k, v in s.items() if k in known})
                for s in self.curriculum]

    def __post_init__(self):
        for f in fields(self):
            if f.name == "curriculum":
                continue
            val = getattr(self, f.name)
            if val is None:
                continue
            if f.type in (float, Optional[float]) and not isinstance(val, float):
                object.__setattr__(self, f.name, float(val))
            elif f.type in (int, Optional[int]) and not isinstance(val, int):
                object.__setattr__(self, f.name, int(val))

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, source=path)

    @classmethod
    def from_dict(cls, data: dict, source: str = "<dict>") -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = [k for k in data if k not in known]
        if unknown:
            print(f"WARNING: ignoring unknown config keys in {source}: {unknown}")
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
