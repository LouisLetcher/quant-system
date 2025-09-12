from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from ..config import StrategyConfig
from .base import BaseStrategy


def discover_external_strategies(strategies_root: Path) -> dict[str, type[BaseStrategy]]:
    """Discover all BaseStrategy subclasses under the given path by importing packages.

    This expects the external repo to be importable; if not, we temporarily
    add the root to sys.path. Classes must define a `name` attribute.
    """
    import sys

    if str(strategies_root) not in sys.path:
        sys.path.insert(0, str(strategies_root))

    found: dict[str, type[BaseStrategy]] = {}

    for py in strategies_root.rglob("*.py"):
        if py.name.startswith("_"):
            continue
        rel = py.relative_to(strategies_root)
        mod_name = ".".join(rel.with_suffix("").parts)
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                name = getattr(obj, "name", obj.__name__)
                found[name] = obj
    return found


def load_strategy(
    cfg: StrategyConfig, strategies_root: Path, external_index: dict[str, type[BaseStrategy]]
):
    """Load a strategy class either from module or from external discovery."""
    if cfg.module and cfg.cls:
        mod = importlib.import_module(cfg.module)
        cls = getattr(mod, cfg.cls)
        return cls
    # fallback by name from external index
    if cfg.name in external_index:
        return external_index[cfg.name]
    raise ImportError(f"Strategy not found: {cfg}")
