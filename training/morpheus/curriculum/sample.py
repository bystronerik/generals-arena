"""Curriculum class sampler with the Part 10 promotion default."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from training.morpheus.curriculum.confidence import (
    DEFAULT_CONFIDENCE_RULE,
    may_advance_toward_earlier_class,
)
from training.morpheus.curriculum.schema import CurriculumItem, CurriculumManifest


def class_weights(
    *,
    active_classes: Sequence[int],
    class_wdl: dict[int, dict[str, int]] | None = None,
    rule: dict[str, Any] | None = None,
    prefer_earlier: bool = False,
) -> dict[int, float]:
    """
    Sampling weights over active classes.

    By default every active class has equal weight. Weight moves toward earlier
    classes only when ``prefer_earlier`` is True **and** every active class
    passes the non-degenerate WDL confidence rule.
    """
    rule = rule or DEFAULT_CONFIDENCE_RULE
    classes = [int(c) for c in active_classes]
    if not classes:
        return {}

    gate = may_advance_toward_earlier_class(
        class_wdl or {},
        active_classes=classes,
        rule=rule,
    )
    if prefer_earlier and gate["ok"]:
        # Earlier class ids are closer to terminal tactics (class 1).
        raw = {c: float(len(classes) - i) for i, c in enumerate(sorted(classes))}
    else:
        raw = {c: 1.0 for c in classes}

    total = sum(raw.values())
    return {c: raw[c] / total for c in classes}


def sample_item(
    manifest: CurriculumManifest,
    *,
    rng: np.random.Generator,
    active_classes: Sequence[int] | None = None,
    class_wdl: dict[int, dict[str, int]] | None = None,
    prefer_earlier: bool = False,
) -> CurriculumItem:
    """Draw one curriculum item under the promotion-gated class weights."""
    by_class: dict[int, list[CurriculumItem]] = {}
    for item in manifest.items:
        by_class.setdefault(item.class_id, []).append(item)

    classes = list(active_classes) if active_classes is not None else sorted(by_class)
    classes = [c for c in classes if by_class.get(c)]
    if not classes:
        raise ValueError("manifest has no items in the requested active classes")

    weights = class_weights(
        active_classes=classes,
        class_wdl=class_wdl,
        rule=manifest.confidence_rule,
        prefer_earlier=prefer_earlier,
    )
    probs = np.asarray([weights[c] for c in classes], dtype=np.float64)
    probs = probs / probs.sum()
    chosen_class = int(rng.choice(classes, p=probs))
    pool = by_class[chosen_class]
    return pool[int(rng.integers(0, len(pool)))]
