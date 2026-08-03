# Diversity

## Research axis

Morpheus owns the research axis **learned belief-conditioned simultaneous-game
planning**.

It is a research bot, not a new owner of a competition-roster axis. It can
enter arena games and ratings without replacing a roster measurement bot.

## Five-axis position

| Axis | Morpheus |
| --- | --- |
| Primary objective | maximize expected game outcome: win probability minus loss probability |
| Risk posture | state- and belief-dependent equilibrium response |
| Time profile | continuous; no hard strategy windows |
| Information use | persistent memory plus explicit particle belief |
| Army handling | learned joint choice among full, half, merge, build, and pass |

The verdict is **distinct**. Morpheus shares general capture as the win
condition with some bots, but its objective and decision mechanism differ.

## Must always be true

- An RL-trained policy and WDL value guide action choice.
- Search aggregates multiple rule-consistent hidden states.
- Every search node treats the two actions as a simultaneous matrix game.
- The final action comes from learned priors plus search statistics.
- The competition transition, not a copied bot tactic, defines simulation.

## Must never be true

- A fixed heuristic action loop is the primary policy.
- One determinization stands for the fog belief.
- Alternating PUCT gives one player false knowledge of the other action.
- A hard clock switches between copied bot strategies.
- ResBot observations are presented as evidence of ResBot internals.

## Separation from roster owners

- `smoke`: Morpheus does not take the first legal move without scoring.
- `expand_plus`: Morpheus does not optimize land rate as its sole objective.
- `castle_builder`: Morpheus has no conservative fixed castle program.
- `castle_rush`: Morpheus has no aggressive fixed castle program.
- `general_hunter`: Morpheus has no candidate-prior split-probe policy.
- `phase_switch`: Morpheus has no hard turn-window controller.
- `fog_scout`: Morpheus has no fixed fog bonus or whole-map revelation rule.
- `army_convey`: Morpheus has no hand-scored interior convoy step.
- `garrison`: Morpheus has no hard threat formula or sized reserve rule.
- `late_rush`: Morpheus has no fixed rally cell or turn-700 commitment.
- `splitter`: half moves compete with all other actions; they are not the bot's
  identity.
- `choke_control`: Morpheus has no hard choke detector or mandatory hold rule.

Learning behavior that sometimes resembles one result is allowed. Copying that
bot's action mechanism, constants, or decomposition is not.

## Separation from research bots

### Sosipolis

Sosipolis uses purpose-specific search modes, geometric section priors,
mountain-pocket skip, and a thin early castle program. Morpheus uses one shared
learned network, explicit particles, and one simultaneous matrix search.

Morpheus does not inherit SearchMCTS, ContactMCTS, StrikeMCTS, Kubic timing,
section priors, or pocket rules.

### Macaria

Macaria combines a vendored heuristic core with scoped tactical search.
Morpheus has no vendored policy core. Its search priors and leaf values come
from RL, and its matrix integrates fog particles and simultaneous actions.

### Migrated strategy bots

`blitz`, `boom`, `metro`, and `aegis` provide fixed strategic programs.
`proteus` selects among those programs. Morpheus does not copy those programs
or use a mode classifier over them.

## Source test

The design sources are competition rules, the unified observation contract,
general imperfect-information search methods, self-play evidence, and measured
ResBot behavior. No existing bot supplies Morpheus's structure or parameters.

## Distance test

On shared competition seeds, compare Morpheus's action trace with every roster
owner. More than 90% identical turns fails the repository's distance test.

Also compare the five axes after measurement. Matching four or more axes, or
matching the same win/loss/draw pattern without a distinct mechanism, changes
the verdict from `distinct`.

## Alternative rejected

Replacing a roster axis owner is rejected for this specification. Morpheus is
an integrated learned-planning experiment, while roster bots isolate one
behavioral cause at a time.
