"""Bounded per-node particle reservoir for information-set search.

Search (Part 06) keys nodes by observable history. Each node stores a reservoir
capped by the configured particle count. After a real observation, Morpheus
replaces the child's reservoir with the filtered root belief.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from belief import BeliefState, N_PARTICLES, Particle, normalize_weights, resample


@dataclass
class ParticleReservoir:
    """Bounded particle bag attached to one search node."""

    capacity: int = N_PARTICLES
    particles: list[Particle] = field(default_factory=list)
    admitted_count: int = 0
    # Bumped when the particle list changes; backup caches enemy hashes on this.
    version: int = 0

    @property
    def n(self) -> int:
        return len(self.particles)

    def clear(self) -> None:
        self.particles.clear()
        self.admitted_count = 0
        self.version += 1

    def replace_from_belief(self, belief: BeliefState) -> None:
        """Copy the filtered belief into this reservoir (equal capacity)."""
        positive = [p for p in belief.particles if p.weight > 0.0]
        if not positive:
            self.particles = []
            self.admitted_count = 0
            self.version += 1
            return
        rng = np.random.default_rng(0)
        n = min(self.capacity, belief.config.n_particles)
        self.particles = resample(normalize_weights(positive), n, rng)
        self.admitted_count = len(self.particles)
        self.version += 1

    def sample(self, rng: np.random.Generator) -> Particle:
        if not self.particles:
            raise ValueError("empty particle reservoir")
        weights = np.asarray([max(p.weight, 0.0) for p in self.particles], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0:
            idx = int(rng.integers(0, len(self.particles)))
        else:
            idx = int(rng.choice(len(self.particles), p=weights / total))
        return self.particles[idx]

    def admit(self, particle: Particle, rng: np.random.Generator) -> None:
        """Admit one arriving particle; Algorithm R when over capacity."""
        arriving = Particle(
            state=particle.state,
            weight=1.0,
            enemy_memory=particle.enemy_memory,
            enemy_prev_action=particle.enemy_prev_action,
            history=particle.history,
        )
        self.admitted_count += 1
        if self.n < self.capacity:
            self.particles.append(arriving)
            self.version += 1
            return
        t = int(self.admitted_count)
        if float(rng.random()) < (self.capacity / float(t)):
            idx = int(rng.integers(0, self.capacity))
            self.particles[idx] = arriving
            self.version += 1

    def as_belief(self, seat: int, config=None) -> BeliefState:
        from belief import BeliefConfig

        return BeliefState(
            seat=seat,
            particles=list(self.particles),
            config=config or BeliefConfig(n_particles=self.capacity),
            collapsed=False,
        )
