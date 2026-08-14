# Modal jobs

Anything under `scripts/*_modal_*.py` runs in a remote container and can fail in
ways the local process never reports. The local `modal run` stays alive, prints
nothing, and returns no error while the container is already dead.

**Check the job's own output a few minutes after you start it, before you do
anything else, and confirm it got past startup.** A Modal job that dies at
import looks exactly like a Modal job that is working.

---

## Reading the container's output

```bash
modal app list                  # find the ephemeral app id
modal app logs <app-id>         # the container's own traceback
```

Do not pipe a backgrounded `modal run` through `tail` or `head`: they buffer
until the process exits, which turns a crash into an apparent hang. Redirect to
a file, or read the app logs.

## Two failure shapes to expect

- **Module-level code runs twice.** Modal re-imports the script inside the
  container to find the function, so every top-level statement executes there
  too — with only that file mounted. A `from arena... import ...` at module
  scope resolves locally and raises `ModuleNotFoundError` in the container,
  before the job's first line. Guard repo imports with `modal.is_local()`.
- **Missing local inputs fail late.** `add_local_file` is evaluated at image
  build, so a stale or deleted path raises after the run has apparently
  started.

## Storage

Modal Volumes are the primary storage for remote Morpheus training runs, and
they hold Joe's prototyping runs. Joe's interruptible vast.ai runs keep their
durable state in the R2 bucket instead — see
[`joe-vast-train.md`](joe-vast-train.md).

Nothing a Modal job produces is an arena match. Training shards, checkpoints,
and materializations never enter `data/games/`, `data/ratings/`, or
`data/bot_versions/` (root `AGENTS.md`, "What is never stored or rated").

## Related

- [`joe-vast-train.md`](joe-vast-train.md) — the interruptible, R2-backed
  alternative to a Modal training run
- [`local-matchup.md`](local-matchup.md) — matches that *do* feed `data/games/`
