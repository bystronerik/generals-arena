#!/usr/bin/env python3
"""Download the replays behind one generals.bot sprint results asset.

A sprint asset (`https://www.generals.bot/assets/<tourney-id>.json`) is a
tournament report, not a leaderboard query: it carries the run's configuration,
standings, cross tables, and one row per game. The replays themselves are not
served by the leaderboard API at all — each row points at a gzipped blob on
public storage, so `scripts/scrape_replays.py` cannot reach them and its
per-player `win/lose/draw` filing has nothing to key on (a row names two bot
checkpoints and a winning *seat*, never a queried player).

The payload behind each blob is byte-identical in schema to a scraped
leaderboard replay, so everything under `arena/instrument/replay/` reads these
unchanged once they are on disk.

    python scripts/scrape_sprint.py
    python scripts/scrape_sprint.py https://www.generals.bot/assets/sprint-2026-08-08.json
    python scripts/scrape_sprint.py path/to/sprint.json --limit 5

Like the leaderboard scraper, runs are incremental and writes are atomic, so an
interrupted or throttled run resumes by re-running.

Sprint replays are observational data — bot-vs-bot games run by someone else's
harness. They never enter `data/games/`, `data/ratings/`, or a rating fit. See
`docs/engine/sprint-replays.md`.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

if TYPE_CHECKING:
    import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRAPER = REPO_ROOT / "competition-scraper" / "scrape.py"
SPRINTS_DIR = REPO_ROOT / "competition-replays" / "_sprints"
DEFAULT_SOURCE = "https://www.generals.bot/assets/sprint-2026-08-08.json"
SUPPORTED_SCHEMA = "sprint-results/1"

# Blob storage is a CDN rather than the rate-limited leaderboard API, so this is
# politeness, not a measured cap. The shared throttle gate still applies.
DEFAULT_RATE = 8.0


def load_scraper() -> ModuleType:
    """Import the submodule script by path — it is not a package on sys.path.

    Its pacing primitives (`RateLimiter`, `ThrottleGate`) and atomic write are
    reused here rather than reimplemented; only the request shape differs.
    """
    if not SCRAPER.exists():
        raise SystemExit(
            f"{SCRAPER} is missing — run: git submodule update --init competition-scraper"
        )
    spec = importlib.util.spec_from_file_location("competition_scraper", SCRAPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves the module's postponed annotations through sys.modules,
    # so register before executing.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"{exc} — install the scraper dependency: pip install httpx") from exc
    return module


@dataclass
class Job:
    """One blob to fetch, with the match rows that referenced it."""

    url: str
    replay_path: Path
    meta_path: Path
    rows: list[dict] = field(default_factory=list)


@dataclass
class Fetcher:
    """A client plus the two shared pacing controls, keyed on absolute URLs.

    The submodule's `Fetcher` builds every request against
    `/api/leaderboard`; sprint blobs live on a different host entirely.
    """

    client: httpx.AsyncClient
    limiter: Any
    gate: Any

    async def get(self, url: str) -> httpx.Response:
        await self.gate.wait()
        await self.limiter.acquire()
        return await self.client.get(url)


def decode_replay(body: bytes) -> bytes:
    """Return replay JSON bytes from a blob that may or may not arrive gzipped.

    The blobs are stored gzipped and served as `application/gzip`, which httpx
    does not transparently decode — but a proxy that sets `Content-Encoding`
    would make it do exactly that, and then the body is already plain JSON.
    """
    try:
        return gzip.decompress(body)
    except gzip.BadGzipFile:
        json.loads(body)  # raises if it is neither gzip nor JSON
        return body


async def fetch_replay(scraper: ModuleType, fetcher: Fetcher, url: str) -> bytes:
    """Fetch and decompress one replay blob, retrying transient failures.

    Mirrors the submodule's retry policy: a throttle is the server asking for
    time, so it waits on the shared gate and does not spend an attempt; broken
    bodies and 5xx do.
    """
    import httpx

    attempt = 0
    while True:
        try:
            response = await fetcher.get(url)
            if response.status_code in scraper.THROTTLE_STATUSES:
                delay = await fetcher.gate.trip(scraper.retry_after_seconds(response))
                print(
                    f"  throttled ({response.status_code}) on {url}, "
                    f"all workers pausing {delay:.0f}s",
                    file=sys.stderr,
                )
                continue
            if response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"server error {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            body = decode_replay(response.content)
            json.loads(body)  # sanity check before it hits disk
            fetcher.gate.clear()
            return body
        except (httpx.HTTPError, json.JSONDecodeError, gzip.BadGzipFile) as exc:
            attempt += 1
            if attempt >= scraper.MAX_ATTEMPTS:
                raise
            print(
                f"  retry {attempt}/{scraper.MAX_ATTEMPTS - 1} for {url}: {exc}",
                file=sys.stderr,
            )
            await asyncio.sleep(scraper.BACKOFF_BASE * 2 ** (attempt - 1))


async def download(
    scraper: ModuleType, fetcher: Fetcher, semaphore: asyncio.Semaphore, job: Job
) -> Job | None:
    """Download and save one replay plus its sidecar. Returns the job on failure."""
    async with semaphore:
        try:
            body = await fetch_replay(scraper, fetcher, job.url)
        except scraper.Throttled:
            raise
        except Exception as exc:
            print(f"  FAILED {job.replay_path.name}: {exc}", file=sys.stderr)
            return job
    scraper.write_atomic(job.replay_path, body)
    scraper.write_atomic(job.meta_path, json.dumps(sidecar(job), indent=2, sort_keys=True).encode())
    return None


def sidecar(job: Job) -> dict:
    """Provenance for one replay: the sprint rows that pointed at it, verbatim.

    The rows are nested rather than spliced in at the top level on purpose. A
    leaderboard sidecar's `winner` is `A`/`B`/`D` against `a_name`/`b_name`; a
    sprint row's is `p0`/`p1`/`draw` against two checkpoint keys. Flattening
    would hand `arena.instrument.replay.Meta` a field it would read in the
    wrong vocabulary, so it sees nothing and the outcome stays derived from the
    replay's own `winner`, which is the one source that cannot be filed wrong.
    """
    stages = sorted({row.get("stage") for row in job.rows if row.get("stage")})
    return {
        "source": SUPPORTED_SCHEMA,
        "replay_gz": job.url,
        "stages": stages,
        "matches": job.rows,
    }


def blob_stem(url: str) -> str | None:
    """`.../<pair>/<seed>-<side>.json.gz` -> `<seed>-<side>`, or None if unusable.

    Taken from the URL rather than rebuilt from the row's fields: the URL is
    what actually identifies the blob. Anything that could escape the output
    directory is rejected instead of sanitized.
    """
    tail = unquote(url.rsplit("/", 1)[-1])
    stem = tail.removesuffix(".gz").removesuffix(".json")
    if not stem or "/" in stem or "\\" in stem or stem.startswith("."):
        return None
    return stem


def pair_dirname(row: dict) -> str | None:
    """`pair` -> one directory per matchup; keys are stable, display names are not.

    From `pair`, not from `p0_key`/`p1_key`: those two swap between the rows for
    the two sides of the same matchup, which would scatter one pair's games
    across two directories that differ only in order. `pair` keeps the upstream
    grouping, where `<seed>-0` and `<seed>-1` sit side by side.
    """
    pair = row.get("pair") or ""
    keys = pair.split("|")
    if len(keys) != 2 or not all(keys):
        return None
    if any(c in pair for c in "/\\."):
        return None
    return "_".join(keys)


def plan_jobs(matches: list[dict], out_dir: Path, limit: int | None) -> tuple[list[Job], int, int]:
    """Group match rows into one job per distinct blob.

    Rows outnumber blobs: a pair's first seeds are played in both the qualifier
    and the deeper final round, and the same game is then listed once per stage
    under one URL. Downloading per row would fetch those twice.
    """
    jobs: dict[str, Job] = {}
    unusable = 0
    for row in matches:
        url = row.get("replay_gz")
        stem = blob_stem(url) if url else None
        pair = pair_dirname(row)
        if not url or stem is None or pair is None:
            print(f"  unusable row {json.dumps(row)[:120]}…, skipping", file=sys.stderr)
            unusable += 1
            continue
        job = jobs.get(url)
        if job is None:
            pair_dir = out_dir / pair
            job = Job(url, pair_dir / f"{stem}.json", pair_dir / f"{stem}.meta.json")
            jobs[url] = job
        job.rows.append(row)

    for job in jobs.values():
        results = {(r.get("winner"), r.get("turns")) for r in job.rows}
        if len(results) > 1:
            # Same blob, two reported results: the file can only match one of
            # them, so say so rather than silently keeping the first row's.
            print(
                f"  {job.replay_path.name}: rows disagree on the result {sorted(results)}",
                file=sys.stderr,
            )

    pending = [job for job in jobs.values() if not job.replay_path.exists()]
    skipped = len(jobs) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    for job in pending:
        job.replay_path.parent.mkdir(parents=True, exist_ok=True)
    return pending, skipped, unusable


def load_source(source: str) -> tuple[dict, bytes]:
    """Read the sprint asset from a local path or fetch it over HTTP."""
    local = Path(source)
    if local.is_file():
        raw = local.read_bytes()
    else:
        try:
            import httpx
        except ModuleNotFoundError as exc:
            raise SystemExit(f"{exc} — install the scraper dependency: pip install httpx") from exc
        response = httpx.get(source, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        raw = response.content
    results = json.loads(raw)
    schema = results.get("schema")
    if schema != SUPPORTED_SCHEMA:
        # Not fatal: the fields this needs may well survive a bump, and a
        # one-time pull is cheap to inspect. Say it loudly and continue.
        print(
            f"warning: {source} declares schema {schema!r}, expected {SUPPORTED_SCHEMA!r}",
            file=sys.stderr,
        )
    return results, raw


def write_manifest(results: dict, raw: bytes, out_dir: Path, scraper: ModuleType) -> None:
    """Keep the asset itself, plus a key->name index for the checkpoint keys.

    The replay directories are named by checkpoint key because keys are stable
    and display names are neither unique nor immutable; this is how a key gets
    read back as a bot.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scraper.write_atomic(out_dir / "sprint.json", raw)
    bots = {
        entry["key"]: {
            "name": entry.get("name"),
            "user_id": entry.get("user_id"),
            "rank": entry.get("rank"),
            "stage": entry.get("stage"),
        }
        for entry in results.get("standings_r1", []) + results.get("standings", [])
        if entry.get("key")
    }
    scraper.write_atomic(out_dir / "bots.json", json.dumps(bots, indent=2, sort_keys=True).encode())


def print_inventory(out_dir: Path) -> None:
    paths = [p for p in out_dir.rglob("*.json") if not p.name.endswith(".meta.json")]
    replays = [p for p in paths if p.parent != out_dir]
    if not replays:
        print(f"no replays on disk under {out_dir}")
        return
    megabytes = sum(p.stat().st_size for p in replays) / 1e6
    pairs = len({p.parent for p in replays})
    print(
        f"{len(replays)} replays on disk across {pairs} pairs — "
        f"{megabytes:.0f} MB in {out_dir}"
    )


async def main_async(args: argparse.Namespace, scraper: ModuleType) -> int:
    results, raw = load_source(args.source)
    tourney_id = results.get("tourney_id") or Path(args.source).stem
    out_dir = args.out / tourney_id
    matches = results.get("matches", [])

    jobs, skipped, unusable = plan_jobs(matches, out_dir, args.limit)
    print(
        f"{tourney_id}: {len(matches)} match rows, {len(jobs)} blobs to download, "
        f"{skipped} already on disk, {unusable} unusable"
    )
    if args.dry_run:
        # Planning is offline, so it works from a local asset with nothing installed.
        for job in jobs[:10]:
            print(f"  would fetch {job.url} -> {job.replay_path.relative_to(args.out)}")
        if len(jobs) > 10:
            print(f"  … and {len(jobs) - 10} more")
        return 0

    write_manifest(results, raw, out_dir, scraper)
    if not jobs:
        print_inventory(out_dir)
        return 0

    import httpx

    exit_code = 0
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(30.0, connect=10.0)
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        fetcher = Fetcher(client, scraper.RateLimiter(args.rate), scraper.ThrottleGate())
        # Collected rather than raised: the gate trips every worker at once, so
        # letting the first one propagate would tear the client out from under
        # the rest mid-write. Writes are atomic and runs are incremental, so
        # what landed stays good and re-running picks up the remainder.
        outcomes = await asyncio.gather(
            *(download(scraper, fetcher, semaphore, job) for job in jobs),
            return_exceptions=True,
        )

    blocked = [o for o in outcomes if isinstance(o, scraper.Throttled)]
    if blocked:
        print(f"{tourney_id}: {blocked[0]}; re-run to resume", file=sys.stderr)
        print_inventory(out_dir)
        return 1
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome

    failures = [job for job in outcomes if job is not None]
    if failures:
        exit_code = 1
    print(
        f"{tourney_id}: {len(jobs) - len(failures)} downloaded, "
        f"{skipped} skipped, {len(failures)} failed"
    )
    print_inventory(out_dir)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_SOURCE,
        help=f"sprint results URL or local path (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="parallel downloads (default: 8)")
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        help=f"max requests per second (default: {DEFAULT_RATE}; 0 disables pacing)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SPRINTS_DIR,
        help=f"output root; replays go under <out>/<tourney_id>/ (default: {SPRINTS_DIR})",
    )
    parser.add_argument("--limit", type=int, default=None, help="download at most N replays")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and download nothing"
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        return 2
    if args.rate < 0:
        print("--rate must be >= 0 (0 disables pacing)", file=sys.stderr)
        return 2

    scraper = load_scraper()
    try:
        return asyncio.run(main_async(args, scraper))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
