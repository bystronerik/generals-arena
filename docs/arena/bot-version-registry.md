# Bot version registry

`data/bot_versions/<bot_id>.json`, one file per bot, **committed to git**.

A content hash names the exact program that played a game. The registry is what
turns that opaque digest back into something a human can read: the file closure
behind it, the commit that was checked out when it ran, and a real git tree you
can diff — even when the closure was never committed.

One file per bot rather than one per hash, so that

```bash
git log -p data/bot_versions/expand_plus.json
```

*is* the bot's improvement history.

## Two structures, deliberately different shapes

- **`versions`** is a *set*, unique by content hash, append-only, never
  mutated. This is what the fit rates. A reverted hash is the **same program**,
  so its games from before and after the revert pool into one entity — which is
  correct, and the reason the estimate keeps getting sharper rather than
  restarting.
- **`steps`** is an ordered *sequence* that may repeat a hash. This is the
  lineage. A revert appends a third step pointing at the first hash, and it
  reports as "returns to seq 1's entity; Δ vs seq 2 = −34 ± 19 Elo".

Lineage order comes from `steps` — an append-only log written when a program
actually ran — **never** from `finished_at` and never from git. That is what
keeps it independent of everything the fit's order-independence guarantee cares
about.

## Format

```json
{
  "version": 1,
  "bot_id": "expand_plus",
  "versions": [
    {
      "content_hash": "ab12cd34ef56",
      "first_seen_at": "2026-07-31T20:14:03Z",
      "git_commit": "b67125f1c0…",
      "git_dirty": false,
      "closure_ref": "refs/bot-versions/ab12cd34ef56",
      "closure_tree": "4b825dc642…",
      "files": [
        {"path": "bots/_common/wire.py", "sha256": "…", "blob": "…"},
        {"path": "bots/expand_plus/agent.py", "sha256": "…", "blob": "…"}
      ]
    }
  ],
  "steps": [
    {"seq": 1, "content_hash": "ab12cd34ef56", "at": "2026-07-31T20:14:03Z"},
    {"seq": 2, "content_hash": "cd34ef5678ab", "at": "2026-08-01T09:02:11Z"},
    {"seq": 3, "content_hash": "ab12cd34ef56", "at": "2026-08-01T13:40:55Z",
     "note": "revert to seq 1"}
  ]
}
```

`files` carries the closure with both the per-file SHA-256 the content hash is
built from and the git blob SHA, so a registry entry alone proves which bytes
were rated.

## Who writes it

**The match runner registers at hash time, in the parent process only.**

- `arena/tournaments/competition.py` already hashed the whole roster once
  before the pool starts; registration hooks in exactly there, once per round.
- `arena/matches/run_match.py` registers for single matches.
- **Pool workers never write.** They call `require_registered` and fail the
  match if a hash is missing. One writer means no concurrent writes.

Registration is idempotent: an existing hash is a no-op, an existing entry is
never mutated, and a step is appended only when the hash differs from the
current lineage head.

Manual use:

```bash
python -m arena.records.registry --register expand_plus blitz
```

```bash
python -m arena.records.registry --verify
```

A **pre-commit hook was rejected**: matches routinely run from dirty trees, so
a commit hook would register versions that never played a game while missing
the ones that did — exactly backwards.

## Dirty working trees

The content hash is over the working tree, so a hash can exist that lives in no
commit. Two mechanisms cover it.

1. **The closure is anchored in git's object database under a ref.** At
   registration the closure files are written as blobs, assembled into a tree,
   committed, and pointed at by `refs/bot-versions/<hash>`. The ref keeps the
   objects reachable across `git gc` and gives every hash a permanently
   diffable name — including across the repo's rebase-onto-main workflow, which
   rewrites the commits recorded in `git_commit` but leaves these independent
   commit objects intact.
2. **The context is recorded.** `git_commit` (HEAD at registration) and
   `git_dirty`, plus a warning on stdout. Dirtiness is scoped to the closure
   itself, so an unrelated edit elsewhere in the repo does not make a bot's
   `git_commit` a lie.

`--strict` refuses to register a closure that differs from HEAD. Use it for
published rounds:

```bash
python -m arena.tournaments.competition bots/*/run.sh --round round6 --strict-versions
```

## What changed between two hashes

Primary — works for dirty-tree hashes, survives rebases:

```bash
python -m arena.records.registry --diff <hash_a> <hash_b>
```

That prints the closure delta (added / removed / changed) and then the real
diff. The underlying git commands, if you want them directly:

```bash
git diff refs/bot-versions/<hash_a> refs/bot-versions/<hash_b>
```

```bash
git ls-tree -r --name-only refs/bot-versions/<hash_a>
```

Fallback when the refs were never fetched, using the recorded commit and
closure:

```bash
git diff <commit_a>..<commit_b> -- $(python -m arena.records.registry --files <hash_a>)
```

## Sharing the refs

Local by default — this is a single-developer repo, and `git_commit` plus the
`files` closure already let anyone else reconstruct a diff. To share them:

```bash
git push origin 'refs/bot-versions/*:refs/bot-versions/*'
```

```bash
git fetch origin 'refs/bot-versions/*:refs/bot-versions/*'
```

## When lineage and git disagree

**`steps` is authoritative for lineage; git is authoritative only for diffs.**
They disagree when a hash was registered from a dirty tree and committed later
or never, when history was rewritten (this repo rebases onto main), or when
work was registered on a side branch before its parent landed.

`--verify` reports entries whose `git_commit` is unreachable or whose
`closure_ref` is missing as `git-unresolvable`, and **never changes a rating**.
A hash whose diff cannot be resolved is still a perfectly valid rated entity:
its identity comes from content, not from git.

## Bots outside `bots/`

`competition-module`'s own `expander_python` lives outside `bots/`, so
`fingerprint` cannot resolve its imports and its hash names nothing useful. It
stays runnable ad hoc, is refused as a tournament roster entry, and its games
are counted under `excluded.unregistered_hash` rather than silently pooled.
`bots/cm_expander/` is the in-repo wrapper over the same upstream agent, and it
is the rating anchor.

## Related

- [ratings.md](ratings.md)
- [game-record-schema.md](game-record-schema.md)
- [decision-rule.md](decision-rule.md)
