# Curriculum data

Part 10 closes the sparse-reward curriculum gaps. Design intent stays in
[training.md](training.md#sparse-reward-solution). This page owns the
executable classifiers, belief-seed derivation, and the pilot confidence rule.

## Curriculum item

A curriculum item is never a bare board. It is:

| Field | Meaning |
| --- | --- |
| `engine_version` | Competition submodule revision that produced the trajectory |
| `map_seed` | Competition map seed |
| `source_label` | Provenance (`fixed_panel`, `league`, `self_play`, `full_start`, …) |
| `prefix_len` | Number of joint actions from turn 0 |
| `game_id` | Source trajectory id (absent for class-5 full starts) |

Training replays the prefix under the matching engine era, then continues. It
does not invent states, remove fog, or change deathtouch.

Raw scraped leaderboard JSON files are not curriculum items and do not supply
policy, value, or hidden-state labels by themselves. Fully reconstructed
trajectories may enter under a separate source label
`<player>_reconstructions` (for example `ResBot_reconstructions`). Measure
those sources apart from `fixed_panel` and `full_start` before promoting them.

## Part 10 executable definitions

### Contact

Seats are in **contact** when a cell owned by seat 0 is orthogonally adjacent
to a cell owned by seat 1. Ownership is the engine `(2, H, W)` bool stack.

### Enemy-general sight

**Sight** holds on a turn when either seat's fog observation has a general cell
that is also an opponent cell (`generals & opponent_cells`).

### Both generals alive

Both seats still own at least one general cell:
`(generals & ownership[seat]).any()` for `seat ∈ {0, 1}`.

### One tactical sequence (class 1)

A turn `t` is class 1 when both generals are alive and either:

1. **Capture or defense threat:** a seat owns a cell with army ≥ 2 that is
   orthogonally adjacent to the enemy general; or
2. **Terminal horizon:** the trajectory is decisive, and
   `0 < terminal_turn − t ≤ TACTICAL_HORIZON` with
   `TACTICAL_HORIZON = 8`.

Classes 1–3 use decisive fixed-panel trajectories before Morpheus wins its own
games. The current research-scope runs instead draw classes 1–4 from
`<player>_reconstructions` sources plus class-5 full starts
([`scraped-classes13.md`](../../morpheus-implementation/scraped-classes13.md)).

### Class assignment (exclusive, priority order)

For a reachable prefix ending at turn `t` on a panel trajectory:

1. Class **1** if the tactical-sequence predicate holds.
2. Else class **2** if sight has already occurred on some turn `≤ t`.
3. Else class **3** if contact has already occurred on some turn `≤ t`.
4. Else class **4** (pre-contact). Store `pre_contact_distance` as the BFS
   length between the two generals over passable cells (`-1` if unreachable).

Class **5** is only empty prefixes (`prefix_len = 0`) on **new** competition
maps that are not panel-trajectory replays. Source label: `full_start`.

### Belief RNG seed

Trajectory files do not store particle RNG state. Deterministic reconstruction
derives one seed per seat from immutable game fields (shared across all
`prefix_len` values of that game so materialize can continue one belief
history):

```text
sha256("{engine_version}|{map_seed}|{source_label}|{game_id}|{seat}")
→ first 8 bytes as big-endian uint64, then mod 2^63
```

Empty `game_id` is used for class-5 items. Reconstruction injects the recorded
enemy action into the belief filter so the particle RNG only affects
initialization and recovery resampling.

### Pilot confidence rule

Written into every curriculum manifest before the main run. Training loss alone
cannot replace it.

| Field | Pilot value |
| --- | --- |
| Interval method | Wilson score interval |
| Confidence level | 0.95 |
| Min samples per active class | 32 |
| Non-degenerate WDL | lower win-rate bound ≥ 0.05 and upper ≤ 0.95; both wins and losses present |
| Full-start decisive rate | ≥ 50 samples; Wilson lower bound ≥ 0.05 |
| Held-out arena | pairwise contrast required per [decision-rule.md](../../arena/decision-rule.md) |
| Promotion default | do not move weight toward an earlier class until every active class is non-degenerate under this rule |

Replace thresholds only from WDL intervals by class, full-start decisive rate,
and held-out arena strength.
