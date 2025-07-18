"""
Core module containing the unified components of the quant system.
This module consolidates all the essential functionality without duplication.
"""

from __future__ import annotations

from .backtest_engine import UnifiedBacktestEngine
from .cache_manager import UnifiedCacheManager
from .data_manager import UnifiedDataManager
from .portfolio_manager import PortfolioManager
from .result_analyzer import UnifiedResultAnalyzer

__all__ = [
    "PortfolioManager",
    "UnifiedBacktestEngine",
    "UnifiedCacheManager",
    "UnifiedDataManager",
    "UnifiedResultAnalyzer",
]
