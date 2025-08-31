from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch


def _load_module():
    p = Path("scripts/prefetch_all.py")
    spec = importlib.util.spec_from_file_location("prefetch_all", p)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_prefetch_all_calls(monkeypatch):
    mod = _load_module()
    # Patch prefetch_one inside loaded module
    mock_pf = patch.object(mod, "prefetch_one").start()
    try:
        rc = mod.main(
            [
                "bonds_core",
                "indices_global_core",
                "--mode",
                "recent",
                "--interval",
                "1d",
                "--recent-days",
                "30",
            ]
        )
        assert rc == 0
        assert mock_pf.call_count == 2
    finally:
        patch.stopall()
