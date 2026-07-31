---
name: structure-audit
description: >-
  Analyze codebase structure, package/folder organization, missing shared
  modules, and harmful logic duplication. Use when the user asks for a
  structure review, architecture cleanup, DRY audit, package organization
  check, or to find repeated logic that should be unified.
---

# Structure Audit

Read-only analysis. Do not edit files unless the user asks for changes after the report.

## Goal

Find weak structure and harmful duplication. Prefer evidence over taste.

## Workflow

1. Map top-level layout and naming conventions from existing code.
2. Sample entrypoints, domain modules, shared utilities, and cross-package imports.
3. Check file/package boundaries: one clear responsibility per module; related code co-located; no dumping grounds.
4. Find unification gaps: same concept implemented in several places with no shared owner.
5. Score duplication: report only when logic is non-trivial and reused (or near-copied) across multiple call sites. Skip incidental similarity.
6. Write the report. Cap noise: prefer fewer high-confidence findings.

## Severity

- **P1**: Wrong package boundary or duplicated core logic that already causes drift/bugs.
- **P2**: Clear misplacement or repeated domain logic with a safe extraction path.
- **P3**: Mild organization debt; optional cleanup.

## Output

### Summary
One short paragraph: overall structure health.

### Findings
For each:
- `[P#] Title — path(s)`
- Evidence (what is repeated or misplaced)
- Impact
- Suggested move/extract (target module/package name)

### Keep as-is
List intentional duplication or good structure so the team does not “fix” it.

### Top actions
At most three refactors, ordered by value vs risk.