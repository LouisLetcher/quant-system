"""
Unified Data Manager - Consolidates all data fetching and management functionality.
Supports multiple data sources including Bybit for crypto futures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings

import pandas as pd
import requests
import aiohttp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .cache_manager import UnifiedCacheManager

warnings.filterwarnings('ignore')


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    name: str
    priority: int
    rate_limit: float
    max_retries: int
    timeout: float
    supports_batch: bool = False
    supports_futures: bool = False
    asset_types: List[str] = None
    max_symbols_per_request: int = 1


class DataSource(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.last_request_time = 0
        self.session = self._create_session()
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
    
    def transform_symbol(self, symbol: str, asset_type: str = None) -> str:
        """Transform symbol to fit this data source's format."""
        return symbol  # Default: no transformation
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _rate_limit(self):
        """Apply rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   interval: str = "1d", **kwargs) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol."""
        pass
    
    @abstractmethod
    def fetch_batch_data(self, symbols: List[str], start_date: str, 
                        end_date: str, interval: str = "1d", **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        pass
    
    @abstractmethod
    def get_available_symbols(self, asset_type: str = None) -> List[str]:
        """Get available symbols for this source."""
        pass
    
    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data format across all sources."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'Open': 'open', 'open': 'open',
            'High': 'high', 'high': 'high',
            'Low': 'low', 'low': 'low', 
            'Close': 'close', 'close': 'close',
            'Adj Close': 'adj_close', 'adj_close': 'adj_close',
            'Volume': 'volume', 'volume': 'volume'
        }
        
        df.columns = [column_mapping.get(col, col.lower()) for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Remove invalid data
        df = df.dropna(subset=['close'])
        df = df[(df['high'] >= df['low']) & 
               (df['high'] >= df['open']) & 
               (df['high'] >= df['close']) &
               (df['low'] <= df['open']) & 
               (df['low'] <= df['close'])]
        
        return df


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source - primary for stocks, forex, commodities."""
    
    def __init__(self):
        config = DataSourceConfig(
            name="yahoo_finance",
            priority=1,
            rate_limit=1.5,
            max_retries=3,
            timeout=30,
            supports_batch=True,
            supports_futures=True,
            asset_types=["stocks", "forex", "commodities", "indices", "crypto"],
            max_symbols_per_request=100
        )
        super().__init__(config)
    
    def transform_symbol(self, symbol: str, asset_type: str = None) -> str:
        """Transform symbol for Yahoo Finance format."""
        # Yahoo Finance forex format
        if asset_type == "forex" or "=" in symbol:
            return symbol  # Already in correct format (EURUSD=X)
        
        # Handle forex pairs without =X
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", 
                      "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD", 
                      "EURCHF", "AUDNZD", "GBPAUD", "GBPCAD"]
        if symbol in forex_pairs:
            return f"{symbol}=X"
        
        # Crypto format - Yahoo uses dash format
        if asset_type == "crypto" or any(crypto in symbol.upper() for crypto in ["BTC", "ETH", "ADA", "SOL"]):
            if "USD" in symbol and "-" not in symbol:
                return symbol.replace("USD", "-USD")
        
        return symbol
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   interval: str = "1d", **kwargs) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        import yfinance as yf
        
        self._rate_limit()
        
        # Transform symbol to Yahoo Finance format
        asset_type = kwargs.get('asset_type')
        transformed_symbol = self.transform_symbol(symbol, asset_type)
        
        try:
            ticker = yf.Ticker(transformed_symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                return None
                
            return self.standardize_data(data)
            
        except Exception as e:
            self.logger.warning(f"Yahoo Finance fetch failed for {symbol}: {e}")
            return None
    
    def fetch_batch_data(self, symbols: List[str], start_date: str, 
                        end_date: str, interval: str = "1d", **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch batch data from Yahoo Finance."""
        import yfinance as yf
        
        self._rate_limit()
        
        try:
            data = yf.download(
                symbols, start=start_date, end=end_date, 
                interval=interval, group_by="ticker", progress=False
            )
            
            result = {}
            if len(symbols) == 1:
                symbol = symbols[0]
                if not data.empty:
                    result[symbol] = self.standardize_data(data)
            else:
                for symbol in symbols:
                    if symbol in data.columns.levels[0]:
                        symbol_data = data[symbol]
                        if not symbol_data.empty:
                            result[symbol] = self.standardize_data(symbol_data)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Yahoo Finance batch fetch failed: {e}")
            return {}
    
    def get_available_symbols(self, asset_type: str = None) -> List[str]:
        """Get available symbols (placeholder implementation)."""
        return []


class BybitSource(DataSource):
    """Bybit data source - primary for crypto futures trading."""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        config = DataSourceConfig(
            name="bybit",
            priority=1,  # Primary for crypto
            rate_limit=0.1,  # 10 requests per second
            max_retries=3,
            timeout=30,
            supports_batch=True,
            supports_futures=True,
            asset_types=["crypto", "crypto_futures"],
            max_symbols_per_request=50
        )
        super().__init__(config)
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Bybit endpoints
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   interval: str = "1d", category: str = "linear", **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch data from Bybit.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date
            end_date: End date
            interval: Kline interval ('1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M')
            category: Product category ('spot', 'linear', 'inverse', 'option')
        """
        self._rate_limit()
        
        try:
            # Convert interval to Bybit format
            bybit_interval = self._convert_interval(interval)
            if not bybit_interval:
                self.logger.error(f"Unsupported interval: {interval}")
                return None
            
            # Convert dates to timestamps
            start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
            end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
            
            # Fetch kline data
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': category,
                'symbol': symbol,
                'interval': bybit_interval,
                'start': start_ts,
                'end': end_ts,
                'limit': 1000
            }
            
            all_data = []
            current_end = end_ts
            
            # Fetch data in chunks (Bybit returns max 1000 records per request)
            while current_end > start_ts:
                params['end'] = current_end
                
                response = self.session.get(url, params=params, timeout=self.config.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('retCode') != 0:
                    self.logger.error(f"Bybit API error: {data.get('retMsg')}")
                    break
                
                klines = data.get('result', {}).get('list', [])
                if not klines:
                    break
                
                all_data.extend(klines)
                
                # Update end timestamp for next iteration
                current_end = int(klines[-1][0]) - 1
                
                # Rate limit between requests
                time.sleep(self.config.rate_limit)
            
            if not all_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return self.standardize_data(df)
            
        except Exception as e:
            self.logger.warning(f"Bybit fetch failed for {symbol}: {e}")
            return None
    
    def fetch_batch_data(self, symbols: List[str], start_date: str, 
                        end_date: str, interval: str = "1d", **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch batch data from Bybit (sequential due to rate limits)."""
        result = {}
        
        for symbol in symbols:
            data = self.fetch_data(symbol, start_date, end_date, interval, **kwargs)
            if data is not None:
                result[symbol] = data
                
        return result
    
    def get_available_symbols(self, asset_type: str = "linear") -> List[str]:
        """Get available trading symbols from Bybit."""
        try:
            url = f"{self.base_url}/v5/market/instruments-info"
            params = {'category': asset_type}
            
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('retCode') != 0:
                self.logger.error(f"Bybit API error: {data.get('retMsg')}")
                return []
            
            instruments = data.get('result', {}).get('list', [])
            symbols = [inst.get('symbol') for inst in instruments if inst.get('status') == 'Trading']
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Bybit symbols: {e}")
            return []
    
    def get_futures_symbols(self) -> List[str]:
        """Get crypto futures symbols."""
        return self.get_available_symbols('linear')
    
    def get_spot_symbols(self) -> List[str]:
        """Get crypto spot symbols."""
        return self.get_available_symbols('spot')
    
    def _convert_interval(self, interval: str) -> Optional[str]:
        """Convert standard interval to Bybit format."""
        mapping = {
            '1m': '1',
            '3m': '3', 
            '5m': '5',
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '2h': '120',
            '4h': '240',
            '6h': '360',
            '12h': '720',
            '1d': 'D',
            '1w': 'W',
            '1M': 'M'
        }
        return mapping.get(interval)


class AlphaVantageSource(DataSource):
    """Alpha Vantage source for additional stock data."""
    
    def __init__(self, api_key: str):
        config = DataSourceConfig(
            name="alpha_vantage",
            priority=3,
            rate_limit=12,  # 5 requests per minute
            max_retries=3,
            timeout=30,
            supports_batch=False,
            asset_types=["stocks", "forex", "commodities"],
            max_symbols_per_request=1
        )
        super().__init__(config)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   interval: str = "1d", **kwargs) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        self._rate_limit()
        
        try:
            function = self._get_function(interval)
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            if interval not in ['1d', '1w', '1M']:
                params['interval'] = self._convert_interval(interval)
            
            response = self.session.get(self.base_url, params=params, timeout=self.config.timeout)
            data = response.json()
            
            # Find time series data
            time_series_key = None
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            df.columns = [col.split('. ')[-1].lower().replace(' ', '_') for col in df.columns]
            
            # Filter by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df.index >= start) & (df.index <= end)]
            
            return self.standardize_data(df) if not df.empty else None
            
        except Exception as e:
            self.logger.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")
            return None
    
    def fetch_batch_data(self, symbols: List[str], start_date: str, 
                        end_date: str, interval: str = "1d", **kwargs) -> Dict[str, pd.DataFrame]:
        """Sequential fetch for Alpha Vantage."""
        result = {}
        for symbol in symbols:
            data = self.fetch_data(symbol, start_date, end_date, interval, **kwargs)
            if data is not None:
                result[symbol] = data
        return result
    
    def get_available_symbols(self, asset_type: str = None) -> List[str]:
        """Get available symbols (placeholder)."""
        return []
    
    def _get_function(self, interval: str) -> str:
        """Get Alpha Vantage function name."""
        if interval in ['1m', '5m', '15m', '30m', '60m']:
            return 'TIME_SERIES_INTRADAY'
        elif interval == '1d':
            return 'TIME_SERIES_DAILY_ADJUSTED'
        elif interval == '1w':
            return 'TIME_SERIES_WEEKLY_ADJUSTED'
        elif interval == '1M':
            return 'TIME_SERIES_MONTHLY_ADJUSTED'
        else:
            return 'TIME_SERIES_DAILY_ADJUSTED'
    
    def _convert_interval(self, interval: str) -> str:
        """Convert to Alpha Vantage format."""
        mapping = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '30m': '30min', '1h': '60min'
        }
        return mapping.get(interval, '1min')


class UnifiedDataManager:
    """
    Unified data manager that consolidates all data fetching functionality.
    Automatically routes requests to appropriate data sources based on asset type.
    """
    
    def __init__(self, cache_manager: UnifiedCacheManager = None):
        self.cache_manager = cache_manager or UnifiedCacheManager()
        self.sources = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize default sources
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize available data sources."""
        import os
        
        # Yahoo Finance (always available - fallback)
        self.add_source(YahooFinanceSource())
        
        # Enhanced Alpha Vantage (good for stocks/forex/crypto)
        av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if av_key:
            try:
                self.add_source(EnhancedAlphaVantageSource())
            except Exception as e:
                self.logger.warning(f"Could not add Enhanced Alpha Vantage: {e}")
                # Fallback to existing implementation
                try:
                    self.add_source(AlphaVantageSource(av_key))
                except:
                    pass
        
        # Twelve Data (excellent coverage)
        twelve_key = os.getenv('TWELVE_DATA_API_KEY')
        if twelve_key:
            try:
                self.add_source(TwelveDataSource())
            except Exception as e:
                self.logger.warning(f"Could not add Twelve Data: {e}")
        
        # Bybit for crypto futures (specialized)
        bybit_key = os.getenv('BYBIT_API_KEY')
        bybit_secret = os.getenv('BYBIT_API_SECRET')
        testnet = os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'
        
        self.add_source(BybitSource(bybit_key, bybit_secret, testnet))
    
    def add_source(self, source: DataSource):
        """Add a data source."""
        self.sources[source.config.name] = source
        self.logger.info(f"Added data source: {source.config.name}")
    
    def get_data(self, symbol: str, start_date: str, end_date: str, 
                 interval: str = "1d", use_cache: bool = True, 
                 asset_type: str = None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Get data for a symbol with intelligent source routing.
        
        Args:
            symbol: Symbol to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            use_cache: Whether to use cached data
            asset_type: Asset type hint ('crypto', 'stocks', 'forex', etc.)
            **kwargs: Additional parameters for specific sources
        """
        # Check cache first
        if use_cache:
            cached_data = self.cache_manager.get_data(symbol, start_date, end_date, interval)
            if cached_data is not None:
                self.logger.debug(f"Cache hit for {symbol}")
                return cached_data
        
        # Determine asset type if not provided
        if not asset_type:
            asset_type = self._detect_asset_type(symbol)
        
        # Get appropriate sources for asset type
        suitable_sources = self._get_sources_for_asset_type(asset_type)
        
        # Try each source in priority order
        for source in suitable_sources:
            try:
                # Pass asset_type to enable symbol transformation
                kwargs['asset_type'] = asset_type
                data = source.fetch_data(symbol, start_date, end_date, interval, **kwargs)
                if data is not None and not data.empty:
                    # Cache the data
                    if use_cache:
                        self.cache_manager.cache_data(symbol, data, interval, source.config.name)
                    
                    self.logger.info(f"Successfully fetched {symbol} from {source.config.name}")
                    return data
                    
            except Exception as e:
                self.logger.warning(f"Source {source.config.name} failed for {symbol}: {e}")
                continue
        
        self.logger.error(f"All sources failed for {symbol}")
        return None
    
    def get_batch_data(self, symbols: List[str], start_date: str, end_date: str,
                      interval: str = "1d", use_cache: bool = True, 
                      asset_type: str = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols with intelligent batching."""
        result = {}
        
        # Group symbols by asset type for optimal source selection
        symbol_groups = self._group_symbols_by_type(symbols, asset_type)
        
        for group_type, group_symbols in symbol_groups.items():
            sources = self._get_sources_for_asset_type(group_type)
            
            # Try batch sources first
            for source in sources:
                if source.config.supports_batch and len(group_symbols) > 1:
                    try:
                        batch_data = source.fetch_batch_data(
                            group_symbols, start_date, end_date, interval, **kwargs
                        )
                        
                        for symbol, data in batch_data.items():
                            if data is not None and not data.empty:
                                result[symbol] = data
                                if use_cache:
                                    self.cache_manager.cache_data(
                                        symbol, data, interval, source.config.name
                                    )
                                group_symbols.remove(symbol)
                        
                        if not group_symbols:  # All symbols fetched
                            break
                            
                    except Exception as e:
                        self.logger.warning(f"Batch fetch failed from {source.config.name}: {e}")
            
            # Fall back to individual requests for remaining symbols
            for symbol in group_symbols:
                individual_data = self.get_data(
                    symbol, start_date, end_date, interval, use_cache, group_type, **kwargs
                )
                if individual_data is not None:
                    result[symbol] = individual_data
        
        return result
    
    def get_crypto_futures_data(self, symbol: str, start_date: str, end_date: str,
                               interval: str = "1d", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Get crypto futures data specifically from Bybit."""
        bybit_source = self.sources.get('bybit')
        if not bybit_source:
            self.logger.error("Bybit source not available for futures data")
            return None
        
        # Check cache first
        if use_cache:
            cached_data = self.cache_manager.get_data(symbol, start_date, end_date, interval, 'futures')
            if cached_data is not None:
                return cached_data
        
        try:
            data = bybit_source.fetch_data(
                symbol, start_date, end_date, interval, category='linear'
            )
            
            if data is not None and use_cache:
                self.cache_manager.cache_data(symbol, data, interval, 'bybit', data_type='futures')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch futures data for {symbol}: {e}")
            return None
    
    def _detect_asset_type(self, symbol: str) -> str:
        """Detect asset type from symbol."""
        symbol_upper = symbol.upper()
        
        # Crypto patterns
        if any(pattern in symbol_upper for pattern in ['USDT', 'BTC', 'ETH', 'BNB', 'ADA']):
            return 'crypto'
        elif symbol_upper.endswith('USD') and len(symbol_upper) > 6:
            return 'crypto'
        elif '-USD' in symbol_upper:
            return 'crypto'
        
        # Forex patterns
        elif symbol_upper.endswith('=X') or len(symbol_upper) == 6:
            return 'forex'
        
        # Futures patterns
        elif symbol_upper.endswith('=F'):
            return 'commodities'
        
        # Default to stocks
        else:
            return 'stocks'
    
    def _get_sources_for_asset_type(self, asset_type: str) -> List[DataSource]:
        """Get appropriate sources for asset type, sorted by priority."""
        suitable_sources = []
        
        for source in self.sources.values():
            if not source.config.asset_types or asset_type in source.config.asset_types:
                suitable_sources.append(source)
        
        # Sort by priority (lower number = higher priority)
        if asset_type == 'crypto':
            # Prioritize Bybit for crypto
            suitable_sources.sort(key=lambda x: (0 if x.config.name == 'bybit' else x.config.priority))
        else:
            suitable_sources.sort(key=lambda x: x.config.priority)
        
        return suitable_sources
    
    def _group_symbols_by_type(self, symbols: List[str], default_type: str = None) -> Dict[str, List[str]]:
        """Group symbols by detected asset type."""
        groups = {}
        
        for symbol in symbols:
            asset_type = default_type or self._detect_asset_type(symbol)
            if asset_type not in groups:
                groups[asset_type] = []
            groups[asset_type].append(symbol)
        
        return groups
    
    def get_available_crypto_futures(self) -> List[str]:
        """Get available crypto futures symbols."""
        bybit_source = self.sources.get('bybit')
        if bybit_source:
            return bybit_source.get_futures_symbols()
        return []
    
    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources."""
        status = {}
        for name, source in self.sources.items():
            status[name] = {
                'priority': source.config.priority,
                'rate_limit': source.config.rate_limit,
                'supports_batch': source.config.supports_batch,
                'supports_futures': source.config.supports_futures,
                'asset_types': source.config.asset_types,
                'max_symbols_per_request': source.config.max_symbols_per_request
            }
        return status


# Additional Data Sources

class EnhancedAlphaVantageSource(DataSource):
    """Enhanced Alpha Vantage data source - excellent for stocks, forex, crypto."""
    
    def __init__(self):
        config = DataSourceConfig(
            name="alpha_vantage_enhanced",
            priority=2,
            rate_limit=5.0,  # 5 calls per minute for free tier
            max_retries=3,
            timeout=30.0,
            supports_batch=False,
            asset_types=["stock", "forex", "crypto", "commodity"]
        )
        super().__init__(config)
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.base_url = "https://www.alphavantage.co/query"
    
    def transform_symbol(self, symbol: str, asset_type: str = None) -> str:
        """Transform symbol for Alpha Vantage format."""
        # Alpha Vantage forex format (no =X suffix)
        if "=X" in symbol:
            return symbol.replace("=X", "")
        
        # Alpha Vantage crypto format (no dash)
        if "-USD" in symbol:
            return symbol.replace("-USD", "USD")
        
        return symbol
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                   interval: str = "1d", **kwargs) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        try:
            self._rate_limit()
            
            # Transform symbol to Alpha Vantage format
            asset_type = kwargs.get('asset_type')
            transformed_symbol = self.transform_symbol(symbol, asset_type)
            
            # Map intervals
            av_interval = self._map_interval(interval)
            function = self._get_function(transformed_symbol, interval)
            
            params = {
                'function': function,
                'symbol': transformed_symbol,
                'apikey': self.api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            if interval in ['1min', '5min', '15min', '30min', '60min']:
                params['interval'] = av_interval
            
            response = self.session.get(self.base_url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                self.logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return None
            
            # Parse data
            time_series_key = self._get_time_series_key(data)
            if not time_series_key:
                return None
            
            df = self._parse_time_series(data[time_series_key])
            if df is not None:
                df = self._filter_date_range(df, start_date, end_date)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
            return None
    
    def _map_interval(self, interval: str) -> str:
        """Map internal intervals to Alpha Vantage intervals."""
        mapping = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '60min',
            '1d': 'daily'
        }
        return mapping.get(interval, 'daily')
    
    def _get_function(self, symbol: str, interval: str) -> str:
        """Get appropriate Alpha Vantage function."""
        if '/' in symbol:  # Forex
            if interval == '1d':
                return 'FX_DAILY'
            else:
                return 'FX_INTRADAY'
        elif any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'LTC', 'XRP']):
            if interval == '1d':
                return 'DIGITAL_CURRENCY_DAILY'
            else:
                return 'CRYPTO_INTRADAY'
        else:  # Stocks
            if interval == '1d':
                return 'TIME_SERIES_DAILY'
            else:
                return 'TIME_SERIES_INTRADAY'
    
    def _get_time_series_key(self, data: dict) -> Optional[str]:
        """Find the time series key in the response."""
        for key in data.keys():
            if 'Time Series' in key:
                return key
        return None
    
    def _parse_time_series(self, time_series: dict) -> Optional[pd.DataFrame]:
        """Parse time series data into DataFrame."""
        try:
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            column_mapping = {}
            for col in df.columns:
                if 'open' in col.lower():
                    column_mapping[col] = 'Open'
                elif 'high' in col.lower():
                    column_mapping[col] = 'High'
                elif 'low' in col.lower():
                    column_mapping[col] = 'Low'
                elif 'close' in col.lower():
                    column_mapping[col] = 'Close'
                elif 'volume' in col.lower():
                    column_mapping[col] = 'Volume'
            
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing Alpha Vantage data: {e}")
            return None


class TwelveDataSource(DataSource):
    """Twelve Data source - excellent coverage for stocks, forex, crypto, indices."""
    
    def __init__(self):
        config = DataSourceConfig(
            name="twelve_data",
            priority=2,
            rate_limit=1.0,  # 8 requests per minute for free tier
            max_retries=3,
            timeout=30.0,
            supports_batch=True,
            max_symbols_per_request=8,
            asset_types=["stock", "forex", "crypto", "index", "etf"]
        )
        super().__init__(config)
        self.api_key = os.getenv('TWELVE_DATA_API_KEY', 'demo')
        self.base_url = "https://api.twelvedata.com"
    
    def transform_symbol(self, symbol: str, asset_type: str = None) -> str:
        """Transform symbol for Twelve Data format."""
        # Twelve Data forex format (use slash format)
        if "=X" in symbol:
            base_symbol = symbol.replace("=X", "")
            if len(base_symbol) == 6:  # EURUSD -> EUR/USD
                return f"{base_symbol[:3]}/{base_symbol[3:]}"
        
        # Twelve Data crypto format (no dash)
        if "-USD" in symbol:
            return symbol.replace("-USD", "USD")
        
        return symbol
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime,
                   interval: str = "1d", **kwargs) -> Optional[pd.DataFrame]:
        """Fetch data from Twelve Data."""
        try:
            self._rate_limit()
            
            # Transform symbol to Twelve Data format
            asset_type = kwargs.get('asset_type')
            transformed_symbol = self.transform_symbol(symbol, asset_type)
            
            params = {
                'symbol': transformed_symbol,
                'interval': self._map_interval(interval),
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'apikey': self.api_key,
                'format': 'JSON',
                'outputsize': 5000
            }
            
            url = f"{self.base_url}/time_series"
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if 'code' in data and data['code'] != 200:
                self.logger.error(f"Twelve Data error: {data.get('message', 'Unknown error')}")
                return None
            
            if 'values' not in data:
                self.logger.warning(f"No data returned for {symbol}")
                return None
            
            return self._parse_twelve_data(data['values'])
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} from Twelve Data: {e}")
            return None
    
    def _map_interval(self, interval: str) -> str:
        """Map internal intervals to Twelve Data intervals."""
        mapping = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1day',
            '1wk': '1week'
        }
        return mapping.get(interval, '1day')
    
    def _parse_twelve_data(self, values: list) -> Optional[pd.DataFrame]:
        """Parse Twelve Data values into DataFrame."""
        try:
            df = pd.DataFrame(values)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            
            # Convert to numeric and rename columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Select standard columns
            columns = ['Open', 'High', 'Low', 'Close']
            if 'Volume' in df.columns:
                columns.append('Volume')
            
            df = df[columns]
            return df.sort_index()
            
        except Exception as e:
            self.logger.error(f"Error parsing Twelve Data: {e}")
            return None


# Import required modules
import os
