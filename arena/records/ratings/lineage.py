"""
Lineage: what changed, in what order, and by how much.

Order comes from the registry's `steps` — an append-only log written when a
program actually ran — never from the fit and never from `finished_at`. That
separation is deliberate: it means lineage lives entirely in the reporting
layer and cannot leak back into the fit, so asking "did this change help?"
cannot disturb the order-independence guarantee.

A hash that reappears after a revert is the **same entity** (its games pool)
but a **new step** (`seq` 3 pointing at `seq` 1's hash). The delta is always
reported against the previous *step*, not the previous occurrence of that hash.

Lineage is **per round**, because a fit is. A step measured in round N against a
predecessor measured in round N−1 is the +46 Elo trap in its purest form: the
number is real arithmetic over two scales that were never joined. Such a step
now renders `—` with the reason `not_in_this_round`, and a bot with steps in
three rounds gets three tables.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.records.ratings.fit import Delta, RatingFit
from arena.records.ratings.policy import entity_key
from arena.records.registry import BotVersion, Registry, Step, closure_delta


@dataclass(frozen=True)
class StepDelta:
    """One lineage step and how it scored against the step before it."""

    seq: int
    content_hash: str
    entity: str
    at: str
    note: str | None
    rated: bool  # the entity played eligible games and is in the fit
    rating: float | None
    se: float | None
    games: int
    previous_seq: int | None
    previous_hash: str | None
    delta: Delta | None
    # Whether the predecessor played in *this round*. False with a predecessor
    # present in the registry means there is no contrast to report, and the
    # missing number is the point: the two steps were measured on two scales.
    previous_rated: bool = False
    # q4: the hash moved only because a bot this one imports changed. `proteus`
    # dispatches to aegis/blitz/boom/metro, so editing any of those forks
    # proteus's lineage. The fork is behaviourally correct and must stay; the
    # label stops it being read as a proteus experiment.
    inherited: bool = False

    @property
    def p_stronger(self) -> float | None:
        return None if self.delta is None else self.delta.p_stronger


def steps(bot_id: str, registry: Registry) -> list[Step]:
    """The bot's ordered lineage. Empty when the bot has never been registered."""
    entry = registry.load(bot_id)
    return list(entry.steps) if entry is not None else []


def _own_directory_changed(
    bot_id: str, before: BotVersion | None, after: BotVersion | None
) -> bool:
    if before is None or after is None:
        return True
    prefix = f"bots/{bot_id}/"
    delta = closure_delta(before, after)
    touched = delta["added"] + delta["removed"] + delta["changed"]
    return any(path.startswith(prefix) for path in touched)


def lineage_deltas(bot_id: str, fit: RatingFit, registry: Registry) -> list[StepDelta]:
    """
    Walk the registry's ordered steps and score each against the one before.

    `fit` is one round's fit, so "rated" means "played in that round". A step
    whose predecessor did not is reported without a delta rather than against a
    predecessor from somewhere else.
    """
    entry = registry.load(bot_id)
    if entry is None:
        return []

    out: list[StepDelta] = []
    previous: Step | None = None
    for step in entry.steps:
        entity = entity_key(bot_id, step.content_hash)
        rated = entity in fit.entities
        previous_entity = (
            entity_key(bot_id, previous.content_hash) if previous is not None else None
        )
        previous_rated = previous_entity is not None and previous_entity in fit.entities
        delta = None
        if rated and previous_rated:
            delta = fit.delta(previous_entity, entity)

        inherited = False
        if previous is not None and previous.content_hash != step.content_hash:
            inherited = not _own_directory_changed(
                bot_id,
                entry.find(previous.content_hash),
                entry.find(step.content_hash),
            )

        out.append(
            StepDelta(
                seq=step.seq,
                content_hash=step.content_hash,
                entity=entity,
                at=step.at,
                note=step.note,
                rated=rated,
                rating=fit.rating(entity) if rated else None,
                se=fit.se(entity) if rated else None,
                games=fit.games(entity) if rated else 0,
                previous_seq=previous.seq if previous is not None else None,
                previous_hash=previous.content_hash if previous is not None else None,
                delta=delta,
                previous_rated=previous_rated,
                inherited=inherited,
            )
        )
        previous = step
    return out


LINEAGE_TABLE_HEADER = (
    "| Step | Hash | Rating | Games | Δ vs prev | 95% CI | P(better) | Note |",
    "| ---: | --- | ---: | ---: | ---: | --- | ---: | --- |",
)


def lineage_table_lines(
    bot_id: str, fit: RatingFit, registry: Registry, *, round_name: str | None = None
) -> list[str]:
    """
    Render one bot's improvement history inside one round, as a markdown table.

    `round_name` defaults to the fit's own round. It names the scale the deltas
    are on, so the table cannot be read as a history across rounds.
    """
    rows = lineage_deltas(bot_id, fit, registry)
    if not rows:
        return [f"_no registered versions for `{bot_id}`_"]

    label = round_name or fit.round
    lines = [f"**`{bot_id}` in round `{label}`**", ""] if label else []
    lines += list(LINEAGE_TABLE_HEADER)
    for row in rows:
        rating = "—" if row.rating is None else f"{row.rating:.1f}"
        incomparable = row.delta is not None and not row.delta.comparable
        if row.delta is None:
            change = ci = probability = "—"
        elif incomparable:
            # The arithmetic difference exists but means nothing: the two
            # versions never met, directly or through any chain of opponents.
            # Printing it as a number is what produced a confident +474 Elo for
            # an edit that could not change a move.
            change = ci = probability = "—"
        else:
            low, high = row.delta.ci
            change = f"{row.delta.value:+.1f}"
            ci = f"[{low:+.1f}, {high:+.1f}]"
            probability = f"{row.delta.p_stronger:.2f}"
        notes = [n for n in (row.note, "inherited" if row.inherited else None) if n]
        if incomparable:
            notes.append("**no games link this step to the previous one**")
        if not row.rated:
            notes.append("not_in_this_round")
        elif row.previous_seq is not None and not row.previous_rated:
            # The predecessor exists, but not here. Its rating is on another
            # round's scale, so there is no delta to print — that is the fix,
            # not a gap in the report.
            notes.append("predecessor not_in_this_round")
        lines.append(
            f"| {row.seq} | `{row.content_hash}` | {rating} | {row.games} "
            f"| {change} | {ci} | {probability} | {'; '.join(notes)} |"
        )
    return lines


__all__ = ["StepDelta", "lineage_deltas", "lineage_table_lines", "steps"]
