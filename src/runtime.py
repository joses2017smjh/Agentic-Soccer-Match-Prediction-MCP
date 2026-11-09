"""Process-level runtime guards shared by every entrypoint.

Native thread pools are sized per *calling* thread: XGBoost (OpenMP) opens one
worker per core for each thread that calls predict, and the gateway serves
requests from a thread pool. On a many-core host with a process cap (an HPC
login node, a container with a pids limit) that multiplies past the cap, and
libgomp responds by calling ``exit(1)`` -- no traceback, no log line.

Single-row inference gains nothing from extra threads, so the default is one.
An explicit setting in the environment always wins; ``MODEL_THREADS`` changes
the default for both pools at once.

This module must stay import-light (stdlib only): it has to run before numpy
or xgboost are imported, which is why the package ``__init__`` files call it.
"""

from __future__ import annotations

import os

THREAD_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def cap_native_threads(default: str | None = None) -> dict[str, str]:
    """Give each native thread pool a bounded size unless one is already set.

    Returns the effective values so callers (and ``/health``) can report them.
    """
    n = default if default is not None else os.environ.get("MODEL_THREADS", "1")
    if not n.isdigit() or int(n) < 1:
        raise ValueError(f"MODEL_THREADS must be a positive integer, got {n!r}")
    for var in THREAD_VARS:
        os.environ.setdefault(var, n)
    return {var: os.environ[var] for var in THREAD_VARS}
