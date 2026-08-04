"""
The one registry of per-turn telemetry keys: type, meaning, and reducers.

Every key a probe emits, and every scalar series the engine records, is
declared here. Two things follow from that:

- **Values are typed.** `BOOL01` lands as `bool`, `INT` as `int`, `TOKEN` as
  `str`. The old path had one hardcoded coercion and stored everything else as
  whatever string the bot happened to print.
- **Unknown keys raise**, naming the key. In a tournament that fails the
  worker's future and aborts the round, which is intended: probes and this
  schema live in one repo, so the fix is a one-line entry in the same commit
  that adds the key. Silently landing as a string is the defect being removed.

A key's reducers say how its per-turn series collapses into `GameRecord.metrics`
on a recorded game. Reducers are pure functions of the series; nothing here
touches storage, the record, or the rating fit.

See docs/arena/trajectories.md and docs/arena/game-record-schema.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


class Kind(Enum):
    """What a telemetry value is, which fixes both coercion and truthiness."""

    BOOL01 = "bool01"  # emitted as 0/1, stored as bool
    INT = "int"
    TOKEN = "token"  # a bare categorical word, no whitespace


class Reducer(Enum):
    """How a per-turn series collapses into one metric."""

    FINAL = "final"
    MEAN = "mean"
    MAX = "max"
    ARGMAX_TURN = "argmax_turn"
    AUC = "auc"
    FIRST_TURN_TRUE = "first_turn_true"


@dataclass(frozen=True)
class TelemetryKey:
    name: str
    kind: Kind
    meaning: str
    reducers: tuple[Reducer, ...]


def _key(name: str, kind: Kind, meaning: str, *reducers: Reducer) -> TelemetryKey:
    return TelemetryKey(name=name, kind=kind, meaning=meaning, reducers=reducers)


F, MEAN, MAX, ARGMAX, AUC, FIRST = (
    Reducer.FINAL,
    Reducer.MEAN,
    Reducer.MAX,
    Reducer.ARGMAX_TURN,
    Reducer.AUC,
    Reducer.FIRST_TURN_TRUE,
)

# Probe keys: what a bot believes, sampled every turn from outside the bot.
_PROBE_KEYS = (
    _key(
        "enemy_general_sighted",
        Kind.BOOL01,
        "the bot believes it has located the enemy general",
        F,
        FIRST,  # replaces the bots' old self-reported first_sighting_turn
    ),
    _key(
        "enemy_land_visible",
        Kind.BOOL01,
        "the bot has ever seen an enemy-owned tile (sticky contact clock; "
        "first_turn aligns with Kubic first_contact)",
        F,
        MEAN,
        FIRST,
    ),
    _key("phase", Kind.TOKEN, "the bot's current strategy phase", F),
    _key("active", Kind.TOKEN, "proteus: which sub-strategy is driving", F),
    _key("label", Kind.TOKEN, "proteus: the classifier's label for the opponent", F),
    _key("reason", Kind.TOKEN, "boom: why the current phase was chosen", F),
    _key("was_attacked", Kind.BOOL01, "aegis: the bot has been attacked", F, MEAN, FIRST),
    _key("countering", Kind.BOOL01, "aegis: a counter-attack window is open", F, MEAN, FIRST),
    _key("pushing", Kind.BOOL01, "metro: the bot is pushing rather than growing", F, MEAN, FIRST),
    _key("strikes", Kind.INT, "blitz: strikes launched so far (monotone)", F),
    _key("switches", Kind.INT, "proteus: strategy switches so far (monotone)", F),
    _key(
        "stack_near",
        Kind.INT,
        "proteus: largest enemy stack ever seen within 8 steps of our general",
        F,
        MAX,
    ),
    _key(
        "turns_near",
        Kind.INT,
        "proteus: turns an enemy cell stood within 8 steps of our general",
        F,
    ),
    _key(
        "duel_turn",
        Kind.INT,
        "proteus: turn a 15+ stack first reached our door, -1 if never",
        F,
    ),
    _key(
        "structures",
        Kind.INT,
        "proteus: opponent structures (general + castles) inferred from the "
        "army aggregate, -1 if not yet estimable; reported for analysis, not "
        "consumed by classify()",
        F,
        MAX,
    ),
    _key(
        "castles_built_probe",
        Kind.INT,
        "metro's own castle count — the engine's tally is castles_built_*",
        F,
    ),
    _key("guard", Kind.INT, "boom: army held back to guard the general", F, MEAN, MAX),
    _key(
        "searched",
        Kind.BOOL01,
        "search bots (macaria): the move search ran this turn. It is scoped "
        "to a trigger window, so this is mostly 0 by design — see the bot's "
        "search.py",
        F,
        MEAN,
        AUC,
        FIRST,
    ),
    _key(
        "overrode",
        Kind.BOOL01,
        "search bots (macaria): the search returned a move other than "
        "the heuristic core's. The pair (searched, overrode) separates 'the "
        "search never ran' from 'it ran and agreed', which are different "
        "defects with the same symptom",
        F,
        MEAN,
        AUC,
    ),
    _key(
        "search_iters",
        Kind.INT,
        "search bots (macaria): search iterations completed this turn, "
        "0 when it did not run; the budget is wall-clock, so this is how much "
        "search the budget actually bought on this machine",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "move_ms",
        Kind.INT,
        "search bots (macaria): wall-clock milliseconds for the whole "
        "move, heuristic plus search (RULES.md §08 budgets 150)",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "candidate_count",
        Kind.INT,
        "sosipolis: remaining enemy-general candidate cells in persistent memory",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "top_section",
        Kind.INT,
        "sosipolis: section index with the highest general prior mass",
        F,
    ),
    _key(
        "pocket_skips",
        Kind.INT,
        "sosipolis: cells marked as mountain-enclosed dead pockets",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "clock_phase",
        Kind.TOKEN,
        "sosipolis: mod-50 gather|wave clock (or opening)",
        F,
    ),
    _key(
        "chain_head",
        Kind.TOKEN,
        "sosipolis: active chain head cell as r,c or none",
        F,
    ),
    _key(
        "recall_fired",
        Kind.INT,
        "sosipolis: rare-recall / imminent-defense firings so far",
        F,
        MAX,
    ),
    _key(
        "land_at_50",
        Kind.INT,
        "sosipolis: owned land at turn 50 (-1 until then)",
        F,
    ),
    _key(
        "first_castle_turn",
        Kind.INT,
        "sosipolis: turn of first owned castle (-1 if none yet)",
        F,
        MAX,
    ),
    _key(
        "enemy_gen",
        Kind.TOKEN,
        "sosipolis: latched enemy general as r,c or none",
        F,
    ),
    _key(
        "tip",
        Kind.TOKEN,
        "sosipolis: selected mass tip cell as r,c or none",
        F,
    ),
    _key(
        "tip_army",
        Kind.INT,
        "sosipolis: army on the selected tip (0 if none)",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "tip_dist_goal",
        Kind.INT,
        "sosipolis: Manhattan tip→objective (-1 if unknown)",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "branch",
        Kind.TOKEN,
        "sosipolis: hard gate that produced the move "
        "(kill|pass|recall|opening|castle|tip_feed|mcts|fault)",
        F,
    ),
    _key(
        "toward",
        Kind.BOOL01,
        "sosipolis: this move closed Manhattan distance to the objective",
        F,
        MEAN,
        FIRST,
    ),
    _key(
        "objective",
        Kind.TOKEN,
        "sosipolis: persistent objective cell as r,c or none",
        F,
    ),
    _key(
        "chain_continued",
        Kind.BOOL01,
        "sosipolis: move source equals prior chain head",
        F,
        MEAN,
    ),
    _key(
        "hunt",
        Kind.TOKEN,
        "sosipolis: belief hunt cell as r,c or none",
        F,
    ),
    _key(
        "muster",
        Kind.TOKEN,
        "sosipolis: gather muster cell as r,c or none",
        F,
    ),
    _key(
        "tip_ready",
        Kind.BOOL01,
        "sosipolis: tip meets march bar vs objective (0 if no tip/goal)",
        F,
        MEAN,
    ),
    _key(
        "root_n",
        Kind.INT,
        "sosipolis: MCTS root-move count this turn (0 outside MCTS)",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "prior_rank",
        Kind.INT,
        "sosipolis: prior-rank of the UCT pick (0=prior-best, -1 if no UCT pick)",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "best_visits",
        Kind.INT,
        "sosipolis: UCT visits on the chosen root (0 if no UCT pick)",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "prior0_visits",
        Kind.INT,
        "sosipolis: UCT visits on the prior-best root (0 if no UCT pick)",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "contact_waypoint",
        Kind.TOKEN,
        "sosipolis: ContactMCTS committed probe waypoint as r,c or none",
        F,
    ),
    _key(
        "contact_macro",
        Kind.TOKEN,
        "sosipolis: ContactMCTS probe macro kind (cluster/frontier/mid_edge/split/none)",
        F,
    ),
    _key(
        "contact_commit_turn",
        Kind.INT,
        "sosipolis: turn the current contact probe commitment started (-1 if none)",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "contact_switches",
        Kind.INT,
        "sosipolis: number of contact probe commitment switches so far",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "contact_macro_score",
        Kind.INT,
        "sosipolis: committed contact probe score * 1000 (normalized)",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "contact_candidate_mass",
        Kind.INT,
        "sosipolis: belief mass * 1000 covered by the committed contact probe",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "completed_simulations",
        Kind.INT,
        "morpheus: search simulations that reached backup this turn",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "forward_equivalents",
        Kind.INT,
        "morpheus: network forward-equivalents consumed this turn",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "belief_ess",
        Kind.INT,
        "morpheus: belief effective sample size * 1000",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "recovery",
        Kind.BOOL01,
        "morpheus: belief update deferred or recovery path used this turn",
        F,
        MEAN,
        FIRST,
    ),
    _key(
        "tree_size",
        Kind.INT,
        "morpheus: information-set nodes retained after the turn",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "fallback_level",
        Kind.TOKEN,
        "morpheus: degradation band (pass|policy|visit|average)",
        F,
    ),
    _key(
        "cost_belief_ms",
        Kind.INT,
        "morpheus: belief-component wall milliseconds this turn",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "cost_root_ms",
        Kind.INT,
        "morpheus: root-inference wall milliseconds this turn",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "cost_search_ms",
        Kind.INT,
        "morpheus: search-loop wall milliseconds this turn",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "cost_reply_ms",
        Kind.INT,
        "morpheus: reply/serialization cost milliseconds this turn",
        F,
        MEAN,
        MAX,
    ),
    _key(
        "belief_plus_root_ok",
        Kind.BOOL01,
        "morpheus: belief update and root inference both completed this turn",
        F,
        MEAN,
        FIRST,
    ),
)

# Engine keys: ground truth, recorded per turn by the trajectory recorder.
# Per-seat series (`land`, `army`) reduce to `<key>_<reducer>_a` / `_b`;
# `land_margin` is one A-perspective series.
_ENGINE_KEYS = (
    _key("land", Kind.INT, "engine truth: cells owned by the seat", MEAN, MAX, ARGMAX),
    _key("army", Kind.INT, "engine truth: total army of the seat", MEAN, MAX, ARGMAX),
    _key(
        "land_margin",
        Kind.INT,
        "engine truth: land_a - land_b",
        F,
        AUC,
        FIRST,  # first turn A was ahead: `>0`, by the INT truthiness rule
    ),
)

ENGINE_KEY_NAMES = tuple(k.name for k in _ENGINE_KEYS)

TELEMETRY_SCHEMA: dict[str, TelemetryKey] = {
    k.name: k for k in _PROBE_KEYS + _ENGINE_KEYS
}


class UnknownTelemetryKey(KeyError):
    """A probe emitted a key no schema entry declares."""


def declared(name: str) -> TelemetryKey:
    """The schema entry for `name`, or a loud failure naming it."""
    try:
        return TELEMETRY_SCHEMA[name]
    except KeyError:
        raise UnknownTelemetryKey(
            f"telemetry key {name!r} is not declared in TELEMETRY_SCHEMA; add an "
            f"entry (kind, meaning, reducers) in the commit that emits it"
        ) from None


def coerce(name: str, value: Any) -> bool | int | str:
    """Coerce one raw probe value to its declared type."""
    kind = declared(name).kind
    if kind is Kind.BOOL01:
        coerced = int(value)
        if coerced not in (0, 1):
            raise ValueError(f"{name} is BOOL01 but got {value!r}")
        return bool(coerced)
    if kind is Kind.INT:
        return int(value)
    text = str(value)
    if not text or any(c.isspace() for c in text):
        raise ValueError(f"{name} is TOKEN but got {value!r}; tokens are non-empty \\S+")
    return text


def validate_extras(extras: dict[str, Any]) -> dict[str, bool | int | str]:
    """Coerce a whole probe dict, raising on the first undeclared key."""
    return {name: coerce(name, value) for name, value in extras.items()}


def is_true(name: str, value: Any) -> bool:
    """
    Truthiness for `first_turn_true`, fixed by the key's kind.

    Explicit because Python's own rule is wrong for two of the three kinds: a
    negative `land_margin` is truthy to Python but means "behind", and `"0"`
    would be truthy as a token.
    """
    kind = declared(name).kind
    if kind is Kind.BOOL01:
        return bool(value)
    if kind is Kind.INT:
        return int(value) > 0
    return bool(str(value))


def reduce_series(name: str, series: Sequence[Any]) -> dict[str, Any]:
    """
    Apply a key's declared reducers to its per-turn series.

    Returns `{suffix: value}` where the suffix is appended to the key name by
    the caller (which also adds the seat). An empty series reduces to nothing:
    missing means not measured, never zero.
    """
    key = declared(name)
    values = list(series)
    if not values:
        return {}

    out: dict[str, Any] = {}
    for reducer in key.reducers:
        if reducer is Reducer.FINAL:
            out[""] = values[-1]
        elif reducer is Reducer.MEAN:
            out["_mean"] = round(sum(values) / len(values), 3)
        elif reducer is Reducer.MAX:
            out["_max"] = max(values)
        elif reducer is Reducer.ARGMAX_TURN:
            # Turns are 1-based, and the first maximum wins: for a monotone
            # series that is when the peak was *reached*, not when it ended.
            out["_argmax_turn"] = values.index(max(values)) + 1
        elif reducer is Reducer.AUC:
            out["_auc"] = sum(values)
        elif reducer is Reducer.FIRST_TURN_TRUE:
            for turn, value in enumerate(values, start=1):
                if is_true(name, value):
                    out["_first_turn"] = turn
                    break
            # Never true: emit nothing. "Never happened" is absence, not 0.
    return out


def metric_keys(name: str, series: Sequence[Any], *, seat: str) -> dict[str, Any]:
    """`reduce_series` with the metric key names the record stores."""
    return {
        f"{name}{suffix}_{seat}": value
        for suffix, value in reduce_series(name, series).items()
    }
