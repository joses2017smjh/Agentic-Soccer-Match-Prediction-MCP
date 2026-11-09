"""Caps native thread pools before any model code loads (see src/runtime.py)."""

from src.runtime import cap_native_threads

cap_native_threads()
