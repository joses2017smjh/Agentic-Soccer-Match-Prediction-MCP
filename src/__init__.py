"""Model, feature and data code. Caps native thread pools before numpy or
xgboost load (see src/runtime.py for why)."""

from src.runtime import cap_native_threads

cap_native_threads()
