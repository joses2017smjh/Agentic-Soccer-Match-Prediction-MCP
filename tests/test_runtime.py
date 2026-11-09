"""Tests: native thread caps (src/runtime.py).

The failure this guards against is not an exception: an unbounded OpenMP pool
per request thread makes libgomp call exit(1). So the checks are that the cap
is in place before model code loads, and that an operator's setting wins.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from src.runtime import THREAD_VARS, cap_native_threads


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in (*THREAD_VARS, "MODEL_THREADS"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_defaults_to_one_thread(clean_env: pytest.MonkeyPatch) -> None:
    assert cap_native_threads() == {var: "1" for var in THREAD_VARS}
    assert all(os.environ[var] == "1" for var in THREAD_VARS)


def test_explicit_setting_wins(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("OMP_NUM_THREADS", "8")
    got = cap_native_threads()
    assert got["OMP_NUM_THREADS"] == "8"          # operator's choice kept
    assert got["OPENBLAS_NUM_THREADS"] == "1"     # the unset pool still capped


def test_model_threads_sets_the_default(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("MODEL_THREADS", "4")
    assert cap_native_threads() == {var: "4" for var in THREAD_VARS}


@pytest.mark.parametrize("bad", ["0", "-2", "many", ""])
def test_rejects_non_positive_values(clean_env: pytest.MonkeyPatch, bad: str) -> None:
    clean_env.setenv("MODEL_THREADS", bad)
    with pytest.raises(ValueError, match="MODEL_THREADS"):
        cap_native_threads()


@pytest.mark.parametrize("package", ["src", "gateway", "mcp_servers"])
def test_importing_a_package_applies_the_cap(package: str) -> None:
    """Every entrypoint package must cap threads on import, in a fresh process
    with nothing set -- that is the state a container starts in."""
    env = {k: v for k, v in os.environ.items() if k not in (*THREAD_VARS, "MODEL_THREADS")}
    code = (
        f"import {package}, os, sys; "
        "assert 'xgboost' not in sys.modules, 'model code loaded before the cap'; "
        "print(os.environ['OMP_NUM_THREADS'], os.environ['OPENBLAS_NUM_THREADS'])"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["1", "1"]
