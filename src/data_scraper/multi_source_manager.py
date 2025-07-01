"""
Multi-source data manager for comprehensive financial data aggregation.
Supports multiple free data sources with intelligent fallback and merging.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data_scraper.cache import Cache


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    priority: int  # Lower number = higher priority
    rate_limit: float  # Minimum seconds between requests
    max_retries: int
    timeout: float
    supports_batch: bool = False
    supports_intervals: List[str] = None
    max_symbols_per_request: int = 1


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.last_request_time = 0
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a session with retry strategy."""
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
                   interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol."""
        pass
    
    @abstractmethod
    def fetch_batch_data(self, symbols: List[str], start_date: str, 
                        end_date: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from this source."""
        pass


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source using yfinance."""
    
    def __init__(self):
        config = DataSourceConfig(
            name="yahoo_finance",
            priority=1,
            rate_limit=1.5,
            max_retries=3,
            timeout=30,
            supports_batch=True,
            supports_intervals=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
            max_symbols_per_request=100
        )
        super().__init__(config)
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        self._rate_limit()
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                return None
                
            # Standardize column names
            return self._standardize_columns(data)
            
        except Exception as e:
            logging.warning(f"Yahoo Finance fetch failed for {symbol}: {e}")
            return None
    
    def fetch_batch_data(self, symbols: List[str], start_date: str, 
                        end_date: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch batch data from Yahoo Finance."""
        self._rate_limit()
        
        try:
            # Split into batches if needed
            batches = [symbols[i:i + self.config.max_symbols_per_request] 
                      for i in range(0, len(symbols), self.config.max_symbols_per_request)]
            
            all_data = {}
            for batch in batches:
                batch_data = yf.download(
                    batch, start=start_date, end=end_date, 
                    interval=interval, group_by="ticker", progress=False
                )
                
                if len(batch) == 1:
                    symbol = batch[0]
                    if not batch_data.empty:
                        all_data[symbol] = self._standardize_columns(batch_data)
                else:
                    for symbol in batch:
                        if symbol in batch_data.columns.levels[0]:
                            symbol_data = batch_data[symbol]
                            if not symbol_data.empty:
                                all_data[symbol] = self._standardize_columns(symbol_data)
                
                # Rate limit between batches
                if len(batches) > 1:
                    time.sleep(self.config.rate_limit)
            
            return all_data
            
        except Exception as e:
            logging.warning(f"Yahoo Finance batch fetch failed: {e}")
            return {}
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols (placeholder - Yahoo Finance has extensive coverage)."""
        # This could be enhanced to fetch from Yahoo Finance's symbol lists
        return []
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        df = df.copy()
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        df.columns = [column_mapping.get(col, col.lower()) for col in df.columns]
        return df


class AlphaVantageSource(DataSource):
    """Alpha Vantage data source (free tier)."""
    
    def __init__(self, api_key: str = None):
        config = DataSourceConfig(
            name="alpha_vantage",
            priority=2,
            rate_limit=12,  # 5 requests per minute for free tier
            max_retries=3,
            timeout=30,
            supports_batch=False,
            supports_intervals=["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"],
            max_symbols_per_request=1
        )
        super().__init__(config)
        self.api_key = api_key or "demo"  # Demo key for testing
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        if not self.api_key or self.api_key == "demo":
            return None
            
        self._rate_limit()
        
        try:
            # Map interval to Alpha Vantage format
            av_interval = self._map_interval(interval)
            if not av_interval:
                return None
            
            function = self._get_function(av_interval)
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            if av_interval not in ["daily", "weekly", "monthly"]:
                params['interval'] = av_interval
            
            response = self.session.get(self.base_url, params=params, timeout=self.config.timeout)
            data = response.json()
            
            # Find the time series key
            time_series_key = None
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break
            
            if not time_series_key or time_series_key not in data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize columns
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adj_close',
                '6. volume': 'volume',
                '5. volume': 'volume'
            }
            
            df.columns = [column_mapping.get(col, col) for col in df.columns]
            df = df.astype(float)
            
            # Filter by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df.index >= start) & (df.index <= end)]
            
            return df if not df.empty else None
            
        except Exception as e:
            logging.warning(f"Alpha Vantage fetch failed for {symbol}: {e}")
            return None
    
    def fetch_batch_data(self, symbols: List[str], start_date: str, 
                        end_date: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch batch data (sequential for Alpha Vantage)."""
        result = {}
        for symbol in symbols:
            data = self.fetch_data(symbol, start_date, end_date, interval)
            if data is not None:
                result[symbol] = data
        return result
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from Alpha Vantage."""
        return []
    
    def _map_interval(self, interval: str) -> Optional[str]:
        """Map standard interval to Alpha Vantage format."""
        mapping = {
            "1m": "1min",
            "5m": "5min", 
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
            "1d": "daily",
            "1wk": "weekly",
            "1mo": "monthly"
        }
        return mapping.get(interval)
    
    def _get_function(self, interval: str) -> str:
        """Get Alpha Vantage function name."""
        if interval in ["1min", "5min", "15min", "30min", "60min"]:
            return "TIME_SERIES_INTRADAY"
        elif interval == "daily":
            return "TIME_SERIES_DAILY_ADJUSTED"
        elif interval == "weekly":
            return "TIME_SERIES_WEEKLY_ADJUSTED"
        elif interval == "monthly":
            return "TIME_SERIES_MONTHLY_ADJUSTED"
        else:
            return "TIME_SERIES_DAILY_ADJUSTED"


class TwelveDataSource(DataSource):
    """Twelve Data source (free tier)."""
    
    def __init__(self, api_key: str = None):
        config = DataSourceConfig(
            name="twelve_data",
            priority=3,
            rate_limit=1.0,  # 8 requests per minute for free tier
            max_retries=3,
            timeout=30,
            supports_batch=True,
            supports_intervals=["1min", "5min", "15min", "30min", "45min", "1h", "2h", "4h", "1day", "1week", "1month"],
            max_symbols_per_request=8  # Free tier limit
        )
        super().__init__(config)
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
    
    def fetch_data(self, symbol: str, start_date: str, end_date: str, 
                   interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch data from Twelve Data."""
        if not self.api_key:
            return None
            
        self._rate_limit()
        
        try:
            td_interval = self._map_interval(interval)
            if not td_interval:
                return None
            
            params = {
                'symbol': symbol,
                'interval': td_interval,
                'start_date': start_date,
                'end_date': end_date,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            response = self.session.get(f"{self.base_url}/time_series", 
                                      params=params, timeout=self.config.timeout)
            data = response.json()
            
            if 'values' not in data or not data['values']:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.sort_index()
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df if not df.empty else None
            
        except Exception as e:
            logging.warning(f"Twelve Data fetch failed for {symbol}: {e}")
            return None
    
    def fetch_batch_data(self, symbols: List[str], start_date: str, 
                        end_date: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch batch data from Twelve Data."""
        # Split into batches
        batches = [symbols[i:i + self.config.max_symbols_per_request] 
                  for i in range(0, len(symbols), self.config.max_symbols_per_request)]
        
        all_data = {}
        for batch in batches:
            batch_data = self._fetch_batch_internal(batch, start_date, end_date, interval)
            all_data.update(batch_data)
            
            # Rate limit between batches
            if len(batches) > 1:
                time.sleep(self.config.rate_limit)
        
        return all_data
    
    def _fetch_batch_internal(self, symbols: List[str], start_date: str, 
                            end_date: str, interval: str) -> Dict[str, pd.DataFrame]:
        """Internal batch fetch method."""
        if not self.api_key:
            return {}
            
        self._rate_limit()
        
        try:
            td_interval = self._map_interval(interval)
            if not td_interval:
                return {}
            
            params = {
                'symbol': ','.join(symbols),
                'interval': td_interval,
                'start_date': start_date,
                'end_date': end_date,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            response = self.session.get(f"{self.base_url}/time_series", 
                                      params=params, timeout=self.config.timeout)
            data = response.json()
            
            result = {}
            if isinstance(data, dict):
                for symbol in symbols:
                    if symbol in data and 'values' in data[symbol]:
                        df = pd.DataFrame(data[symbol]['values'])
                        if not df.empty:
                            df['datetime'] = pd.to_datetime(df['datetime'])
                            df.set_index('datetime', inplace=True)
                            df = df.sort_index()
                            
                            # Convert to numeric
                            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                            for col in numeric_cols:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            result[symbol] = df
            
            return result
            
        except Exception as e:
            logging.warning(f"Twelve Data batch fetch failed: {e}")
            return {}
    
    def get_available_symbols(self) -> List[str]:
        """Get available symbols from Twelve Data."""
        return []
    
    def _map_interval(self, interval: str) -> Optional[str]:
        """Map standard interval to Twelve Data format."""
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min", 
            "30m": "30min",
            "1h": "1h",
            "1d": "1day",
            "1wk": "1week",
            "1mo": "1month"
        }
        return mapping.get(interval)


class MultiSourceDataManager:
    """
    Advanced data manager that aggregates data from multiple sources.
    Provides intelligent fallback, data merging, and comprehensive caching.
    """
    
    def __init__(self, sources: List[DataSource] = None):
        self.sources = sources or [YahooFinanceSource()]
        self.sources.sort(key=lambda x: x.config.priority)
        self.cache = Cache()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_source(self, source: DataSource):
        """Add a new data source."""
        self.sources.append(source)
        self.sources.sort(key=lambda x: x.config.priority)
    
    def get_data(self, symbol: str, start_date: str, end_date: str, 
                 interval: str = "1d", use_cache: bool = True,
                 force_source: str = None) -> Optional[pd.DataFrame]:
        """
        Get data for a single symbol with intelligent source selection.
        
        Args:
            symbol: Symbol to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            use_cache: Whether to use cached data
            force_source: Force specific source by name
            
        Returns:
            DataFrame with standardized OHLCV data
        """
        # Check cache first
        if use_cache:
            cached_data = self._get_cached_data(symbol, start_date, end_date, interval)
            if cached_data is not None:
                return cached_data
        
        # Filter sources
        available_sources = self.sources
        if force_source:
            available_sources = [s for s in self.sources if s.config.name == force_source]
        
        # Try each source in priority order
        for source in available_sources:
            if interval not in (source.config.supports_intervals or []):
                continue
                
            try:
                data = source.fetch_data(symbol, start_date, end_date, interval)
                if data is not None and not data.empty:
                    # Cache the data
                    if use_cache:
                        self._cache_data(symbol, data, interval, source.config.name)
                    
                    self.logger.info(f"Successfully fetched {symbol} from {source.config.name}")
                    return self._validate_and_clean_data(data)
                    
            except Exception as e:
                self.logger.warning(f"Source {source.config.name} failed for {symbol}: {e}")
                continue
        
        self.logger.error(f"All sources failed for {symbol}")
        return None
    
    def get_batch_data(self, symbols: List[str], start_date: str, end_date: str,
                      interval: str = "1d", use_cache: bool = True, 
                      max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple symbols with parallel processing and smart batching.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) 
            interval: Data interval
            use_cache: Whether to use cached data
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        result = {}
        remaining_symbols = symbols.copy()
        
        # Check cache first
        if use_cache:
            cached_results = {}
            for symbol in symbols:
                cached_data = self._get_cached_data(symbol, start_date, end_date, interval)
                if cached_data is not None:
                    cached_results[symbol] = cached_data
                    remaining_symbols.remove(symbol)
            
            result.update(cached_results)
            self.logger.info(f"Found {len(cached_results)} symbols in cache")
        
        if not remaining_symbols:
            return result
        
        # Try batch sources first
        batch_sources = [s for s in self.sources if s.config.supports_batch]
        for source in batch_sources:
            if interval not in (source.config.supports_intervals or []):
                continue
                
            try:
                batch_data = source.fetch_batch_data(remaining_symbols, start_date, end_date, interval)
                
                # Process successful fetches
                for symbol, data in batch_data.items():
                    if data is not None and not data.empty:
                        validated_data = self._validate_and_clean_data(data)
                        if validated_data is not None:
                            result[symbol] = validated_data
                            if use_cache:
                                self._cache_data(symbol, validated_data, interval, source.config.name)
                            remaining_symbols.remove(symbol)
                
                self.logger.info(f"Batch fetched {len(batch_data)} symbols from {source.config.name}")
                
                if not remaining_symbols:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Batch source {source.config.name} failed: {e}")
                continue
        
        # Fall back to individual requests for remaining symbols
        if remaining_symbols:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.get_data, symbol, start_date, end_date, interval, False): symbol
                    for symbol in remaining_symbols
                }
                
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        if data is not None:
                            result[symbol] = data
                    except Exception as e:
                        self.logger.error(f"Individual fetch failed for {symbol}: {e}")
        
        self.logger.info(f"Total fetched: {len(result)}/{len(symbols)} symbols")
        return result
    
    def _get_cached_data(self, symbol: str, start_date: str, end_date: str, 
                        interval: str) -> Optional[pd.DataFrame]:
        """Get cached data if available and fresh."""
        try:
            cached_data = Cache.load_from_cache(symbol, interval)
            if cached_data is None or cached_data.empty:
                return None
            
            # Check if cache covers the requested date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            if cached_data.index[0] <= start and cached_data.index[-1] >= end:
                # Cache covers the range, check if it's fresh
                recency_thresholds = {
                    "1m": timedelta(minutes=30),
                    "5m": timedelta(hours=2),
                    "15m": timedelta(hours=6),
                    "30m": timedelta(hours=12),
                    "1h": timedelta(days=1),
                    "1d": timedelta(days=2),
                    "1wk": timedelta(days=7),
                    "1mo": timedelta(days=30),
                }
                
                threshold = recency_thresholds.get(interval, timedelta(days=2))
                if datetime.now() - cached_data.index[-1].to_pydatetime() < threshold:
                    # Filter to requested date range
                    filtered_data = cached_data[(cached_data.index >= start) & (cached_data.index <= end)]
                    return filtered_data if not filtered_data.empty else None
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache check failed for {symbol}: {e}")
            return None
    
    def _cache_data(self, symbol: str, data: pd.DataFrame, interval: str, source: str):
        """Cache data with source metadata."""
        try:
            # Add source metadata  
            data_with_meta = data.copy()
            data_with_meta.attrs['source'] = source
            data_with_meta.attrs['cached_at'] = datetime.now().isoformat()
            
            Cache.save_to_cache(symbol, data_with_meta, interval)
            
        except Exception as e:
            self.logger.warning(f"Caching failed for {symbol}: {e}")
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate and clean data."""
        if data is None or data.empty:
            return None
        
        # Required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return None
        
        # Remove rows with null critical values
        data = data.dropna(subset=['close'])
        
        # Basic validation
        if len(data) < 2:
            return None
        
        # Ensure proper data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove invalid data points
        data = data[(data['high'] >= data['low']) & 
                   (data['high'] >= data['open']) & 
                   (data['high'] >= data['close']) &
                   (data['low'] <= data['open']) & 
                   (data['low'] <= data['close'])]
        
        return data if not data.empty else None
    
    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources."""
        status = {}
        for source in self.sources:
            status[source.config.name] = {
                'priority': source.config.priority,
                'rate_limit': source.config.rate_limit,
                'supports_batch': source.config.supports_batch,
                'supports_intervals': source.config.supports_intervals,
                'max_symbols_per_request': source.config.max_symbols_per_request,
                'last_request_time': source.last_request_time
            }
        return status
