"""Physical-core job caps and ProcessPool worker init for competition batches."""

from __future__ import annotations

import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")


def physical_cpu_count() -> int:
    """Return physical CPU core count (not hyperthread logical count when possible)."""
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"],
                text=True,
            ).strip()
            n = int(out)
            if n >= 1:
                return n
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    elif system == "Linux":
        try:
            cores: set[tuple[str, str]] = set()
            physical: str | None = None
            core: str | None = None
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("physical id"):
                        physical = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":", 1)[1].strip()
                    elif line.strip() == "":
                        if physical is not None and core is not None:
                            cores.add((physical, core))
                        physical = None
                        core = None
                if physical is not None and core is not None:
                    cores.add((physical, core))
            if cores:
                return len(cores)
        except OSError:
            pass

    logical = os.cpu_count() or 1
    return max(1, logical)


def default_jobs() -> int:
    return physical_cpu_count()


def cap_jobs(jobs: int) -> int:
    """Clamp requested workers to [1, physical_cpu_count()]."""
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1 (got {jobs})")
    return min(jobs, physical_cpu_count())


def pin_worker_threads() -> None:
    """Limit BLAS/OpenMP/XLA to one thread per worker process."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Prefer not to overwrite a caller-provided XLA_FLAGS; append when empty.
    if "XLA_FLAGS" not in os.environ or not os.environ["XLA_FLAGS"].strip():
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false "
            "--xla_force_host_platform_device_count=1"
        )


def worker_initializer() -> None:
    """ProcessPoolExecutor initializer: pin threads before JAX imports."""
    pin_worker_threads()


def run_pool(
    payloads: Iterable[dict[str, Any]],
    worker: Callable[[dict[str, Any]], T],
    *,
    jobs: int,
    on_result: Callable[[int, int, T], None] | None = None,
) -> list[T]:
    """
    Run payloads with a ProcessPoolExecutor.

    When jobs == 1, run in-process (still applies pin_worker_threads once).
    """
    items = list(payloads)
    total = len(items)
    if total == 0:
        return []

    capped = cap_jobs(jobs)
    if capped == 1:
        pin_worker_threads()
        results: list[T] = []
        for i, payload in enumerate(items, start=1):
            result = worker(payload)
            results.append(result)
            if on_result is not None:
                on_result(i, total, result)
        return results

    results_out: list[T] = []
    with ProcessPoolExecutor(
        max_workers=capped,
        initializer=worker_initializer,
    ) as pool:
        futures = {pool.submit(worker, p): p for p in items}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            results_out.append(result)
            if on_result is not None:
                on_result(done, total, result)
    return results_out
