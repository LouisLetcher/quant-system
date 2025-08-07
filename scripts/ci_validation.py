#!/usr/bin/env python3
"""
Local CI validation script that mimics GitHub Actions workflow.
Validates core functionality without requiring dev dependencies.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and check its exit code."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        print(f"❌ {description} - FAILED")
        print(f"Error: {result.stderr}")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def check_import(module_name, description):
    """Check if a module can be imported."""
    print(f"🔍 {description}...")
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"✅ {description} - PASSED")
            return True
        print(f"❌ {description} - FAILED (module not found)")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run all CI validation checks."""
    print("🚀 Running Local CI Validation")
    print("=" * 50)

    passed = 0
    total = 0

    # Core imports
    tests = [
        (
            lambda: check_import("src.core.data_manager", "Core data manager import"),
            None,
        ),
        (
            lambda: check_import(
                "src.core.backtest_engine", "Core backtest engine import"
            ),
            None,
        ),
        (lambda: check_import("src.cli.unified_cli", "CLI module import"), None),
        (
            lambda: run_command(
                "python -m src.cli.unified_cli --help", "CLI help command"
            ),
            None,
        ),
        (
            lambda: run_command(
                "python -m src.cli.unified_cli cache stats", "Cache stats command"
            ),
            None,
        ),
        (
            lambda: run_command(
                "python -m src.cli.unified_cli portfolio backtest --symbols BTCUSDT --strategy BuyAndHold --start-date 2024-01-01 --end-date 2024-01-02",
                "Simple backtest command",
            ),
            None,
        ),
    ]

    for test_func, _ in tests:
        total += 1
        if test_func():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All CI validation checks PASSED!")
        sys.exit(0)
    else:
        print("💥 Some CI validation checks FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
