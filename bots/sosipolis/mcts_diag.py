"""Shared root-pick diagnostics for flat UCT MCTS (probe only)."""
from __future__ import annotations


def record_root_pick(stats, children, best) -> None:
    """
    Fill override / rank / visit fields on ``stats``.

    ``children`` must be in prior-descending order (child 0 = prior-best).
    """
    stats.overrode = 0
    stats.prior_rank = -1
    stats.best_visits = 0
    stats.prior0_visits = 0
    if not children or best is None:
        return
    stats.best_visits = int(best.visits)
    stats.prior0_visits = int(children[0].visits)
    rank = 0
    for i, child in enumerate(children):
        if child is best or child.action == best.action:
            rank = i
            break
    stats.prior_rank = int(rank)
    stats.overrode = 1 if rank > 0 else 0
