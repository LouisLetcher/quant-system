"""
Core module containing the unified components of the quant system.
This module consolidates all the essential functionality without duplication.
"""

from .data_manager import UnifiedDataManager
from .backtest_engine import UnifiedBacktestEngine
from .result_analyzer import UnifiedResultAnalyzer
from .cache_manager import UnifiedCacheManager
from .portfolio_manager import PortfolioManager

__all__ = [
    'UnifiedDataManager',
    'UnifiedBacktestEngine', 
    'UnifiedResultAnalyzer',
    'UnifiedCacheManager',
    'PortfolioManager'
]
