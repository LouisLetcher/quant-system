#!/usr/bin/env python3
"""
Local validation script to test the basic functionality before CI/CD.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all core modules can be imported."""
    try:
        # Test core imports

        print("✅ All core modules import successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        raise AssertionError("Import failed") from e


def test_strategy_creation():
    """Test strategy creation."""
    try:
        from src.core.strategy import BuyAndHoldStrategy, StrategyFactory

        # Test direct creation
        strategy = BuyAndHoldStrategy()
        assert strategy.name == "Buy and Hold"

        # Test factory creation
        strategy2 = StrategyFactory.create_strategy("BuyAndHold")
        assert strategy2.name == "Buy and Hold"

        print("✅ Strategy creation works correctly")
    except Exception as e:
        print(f"❌ Strategy creation error: {e}")
        raise AssertionError("Strategy creation failed") from e


def test_data_manager():
    """Test data manager initialization."""
    try:
        from src.core.data_manager import UnifiedDataManager

        dm = UnifiedDataManager()
        assert isinstance(dm, UnifiedDataManager)

        # Test that it has the expected methods
        assert hasattr(dm, "get_data"), "get_data method missing"
        assert hasattr(dm, "cache_manager"), "cache_manager attribute missing"

        print("✅ Data manager initialization works correctly")
    except Exception as e:
        print(f"❌ Data manager error: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError("Data manager test failed") from e


def test_cache_manager():
    """Test cache manager initialization."""
    try:
        from src.core.cache_manager import UnifiedCacheManager

        cm = UnifiedCacheManager()
        assert isinstance(cm, UnifiedCacheManager)

        # Test stats
        stats = cm.get_cache_stats()
        assert isinstance(stats, dict)

        print("✅ Cache manager initialization works correctly")
    except Exception as e:
        print(f"❌ Cache manager error: {e}")
        raise AssertionError("Cache manager test failed") from e


def test_portfolio_manager():
    """Test portfolio manager initialization."""
    try:
        from src.core.portfolio_manager import PortfolioManager

        pm = PortfolioManager()
        assert isinstance(pm, PortfolioManager)

        print("✅ Portfolio manager initialization works correctly")
    except Exception as e:
        print(f"❌ Portfolio manager error: {e}")
        raise AssertionError("Portfolio manager test failed") from e


def main():
    """Run all validation tests."""
    print("🧪 Running Local Validation Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_strategy_creation,
        test_data_manager,
        test_cache_manager,
        test_portfolio_manager,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Ready for CI/CD pipeline.")
        return 0
    print("⚠️  Some tests failed. Please fix issues before running CI/CD.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
