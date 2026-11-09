"""Where paused runs live: checkpointer selection by environment.

A run that stops for human approval is saved through the graph's
checkpointer and resumed later by ``/approve``. The saver therefore decides
what survives between those two requests:

    CHECKPOINT_URL unset, or ``memory://``
        MemorySaver. Process-local: zero setup, right for tests, evals and a
        single-process demo. A restart, an idle shutdown or a second replica
        loses every run waiting for approval.

    ``sqlite:///relative/path.db``  or  ``sqlite:////absolute/path.db``
        SqliteSaver. A paused run survives a restart of the process. Scope:
        processes on ONE host, with the file on a persistent local disk. It
        is not a store for replicas on separate hosts, it should not sit on
        a network filesystem (SQLite's locking is unreliable there), and it
        does nothing on a host with no persistent disk -- an ephemeral
        container loses the file with the process. Needs the optional
        ``langgraph-checkpoint-sqlite`` package (``pip install -e ".[durable]"``).

Replicas on separate hosts, and hosts without a persistent disk, need a
server-backed store (LangGraph ships ``PostgresSaver``). That backend is NOT
wired here because it has not been run against a real server; asking for it
fails loudly rather than falling back to memory.

Every saver gets a serializer with the project's state classes registered.
LangGraph warns that restoring unregistered classes from a checkpoint "will
be blocked in a future version"; registering them makes resume independent
of that change and turns the allowlist strict today.
"""

from __future__ import annotations

import importlib
import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

# Modules whose Pydantic models travel through checkpoints.
STATE_MODULES: tuple[str, ...] = ("agent.state", "agent.swarm.state")

_SERVER_SCHEMES = {"postgres", "postgresql"}


def state_types() -> list[tuple[str, str]]:
    """(module, class) for every Pydantic model defined in the state modules.

    Discovered rather than listed, so a new state class cannot be forgotten.
    """
    found: list[tuple[str, str]] = []
    for module_name in STATE_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj.__module__ == module_name
            ):
                found.append((module_name, name))
    return sorted(found)


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=state_types())


def _sqlite_path(url: str) -> Path:
    # sqlite:///rel.db -> path "/rel.db" -> "rel.db";  sqlite:////abs.db -> "/abs.db"
    path = urlparse(url).path
    if not path or path == "/":
        raise ValueError(f"CHECKPOINT_URL {url!r} names no database file")
    return Path(path[1:])


def make_checkpointer(url: str | None = None) -> Any:
    """Build the checkpointer named by ``url`` (default: ``CHECKPOINT_URL``)."""
    if url is None:
        url = os.environ.get("CHECKPOINT_URL", "")
    scheme = urlparse(url).scheme if url else "memory"

    if scheme == "memory":
        return MemorySaver(serde=_serializer())

    if scheme == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "CHECKPOINT_URL asks for SQLite but langgraph-checkpoint-sqlite "
                'is not installed; run: pip install -e ".[durable]"'
            ) from exc
        path = _sqlite_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The gateway serves from a thread pool, so the connection must be
        # usable across threads (SqliteSaver serializes access with a lock).
        # Default rollback journal on purpose: WAL needs shared memory and is
        # documented as same-host only. The busy timeout lets a second local
        # process wait for the write lock instead of failing.
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30.0)
        saver = SqliteSaver(conn, serde=_serializer())
        saver.setup()
        return saver

    if scheme in _SERVER_SCHEMES:
        raise NotImplementedError(
            "a server-backed checkpointer (PostgresSaver) is what replicas on "
            "separate hosts need, but it is not wired in this build because it "
            "has not been tested against a real server. Use sqlite:// on a "
            "shared volume, or add the backend in agent/checkpoint.py."
        )

    raise ValueError(
        f"unsupported CHECKPOINT_URL {url!r}: expected memory:// or sqlite:///path"
    )
