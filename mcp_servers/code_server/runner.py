"""Sandbox child process — executes agent-authored analysis code.

Never invoked directly by the agent: ``sandbox.run_code`` spawns this with a
JSON payload on stdin and reads a JSON verdict on stdout, so a runaway cell
kills only this process.

Defence in depth (the parent already AST-validated the code):
1. heavy libs are imported FIRST, then the address-space limit is applied —
   otherwise pandas/numpy's own allocations would trip the limit at import;
2. POSIX rlimits cap CPU seconds, memory, and file size;
3. sockets are replaced with a raising stub, so nothing can phone home;
4. execution runs against a restricted globals map with no ``open``,
   ``eval``, ``exec``, ``__import__`` or friends;
5. stdout is captured and truncated.

This is a hardened *subprocess*, not a kernel-level jail — see the honest
limitations note in docs/SWARM.md.
"""

from __future__ import annotations

import io
import json
import sys


def _install_limits(cpu_seconds: int, memory_mb: int) -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    mem = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _block_network() -> None:
    import socket

    def _denied(*_args, **_kwargs):
        raise PermissionError("network access is disabled in the sandbox")

    socket.socket = _denied            # type: ignore[assignment,misc]
    socket.create_connection = _denied  # type: ignore[assignment]
    socket.getaddrinfo = _denied        # type: ignore[assignment]


def _safe_builtins(allowed_imports: list[str]) -> dict:
    """Everything harmless, nothing that reaches the interpreter or disk.

    ``import`` statements need ``__import__``, so it is reinstated as a
    *guarded* version enforcing the same allow-list the parent's AST check
    applies — a second, runtime line of defence rather than a hole.
    """
    import builtins

    blocked = {
        "eval", "exec", "compile", "open", "__import__", "input", "exit",
        "quit", "help", "breakpoint", "globals", "locals", "vars", "memoryview",
        "getattr", "setattr", "delattr", "hasattr", "dir", "id", "object",
        "super", "type", "classmethod", "staticmethod", "property",
    }
    safe = {k: v for k, v in vars(builtins).items()
            if not k.startswith("_") and k not in blocked}

    real_import = builtins.__import__
    permitted = set(allowed_imports)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] not in permitted:
            raise ImportError(f"import of {name!r} is not allowed in the sandbox")
        return real_import(name, globals, locals, fromlist, level)

    safe["__import__"] = guarded_import
    return safe


def main() -> int:
    payload = json.loads(sys.stdin.read())
    limits = payload.get("limits", {})

    # 1. import the heavy allowed libs before capping the address space
    import datetime as _datetime
    import json as _json
    import math as _math
    import statistics as _statistics

    import numpy as _np
    import pandas as _pd

    _install_limits(int(limits.get("cpu_seconds", 10)),
                    int(limits.get("memory_mb", 512)))
    _block_network()

    datasets: dict[str, str] = payload.get("datasets", {})

    def load(name: str):
        """Read one of the whitelisted, on-disk datasets (read-only)."""
        if name not in datasets:
            raise KeyError(
                f"unknown dataset {name!r}; available: {sorted(datasets)}"
            )
        return _pd.read_csv(datasets[name])

    env: dict = {
        "__builtins__": _safe_builtins(payload.get("allowed_imports", [])),
        "pd": _pd, "pandas": _pd, "np": _np, "numpy": _np,
        "math": _math, "statistics": _statistics, "json": _json,
        "datetime": _datetime, "load": load, "datasets": sorted(datasets),
    }

    buffer = io.StringIO()
    stdout, sys.stdout = sys.stdout, buffer
    verdict: dict = {"ok": True, "error": "", "result": None}
    try:
        for cell in payload.get("cells", []):
            exec(compile(cell, "<cell>", "exec"), env)  # noqa: S102 — the sandbox
    except BaseException as exc:  # noqa: BLE001 — report, never leak a traceback object
        verdict["ok"] = False
        verdict["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sys.stdout = stdout

    if verdict["ok"] and "result" in env:
        try:
            verdict["result"] = json.loads(
                json.dumps(env["result"], default=str)[:20000]
            )
        except Exception:  # noqa: BLE001 — unserializable result
            verdict["result"] = str(env["result"])[:20000]

    verdict["stdout"] = buffer.getvalue()[:20000]
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
