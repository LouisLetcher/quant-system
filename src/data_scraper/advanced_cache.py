"""
Advanced caching system for financial data and backtest results.
Supports hierarchical caching, compression, and intelligent cache management.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging

import pandas as pd


@dataclass
class CacheMetadata:
    """Metadata for cached items."""
    key: str
    cache_type: str  # 'data', 'backtest', 'optimization'
    created_at: datetime
    last_accessed: datetime
    expires_at: Optional[datetime]
    size_bytes: int
    source: Optional[str] = None
    symbol: Optional[str] = None
    interval: Optional[str] = None
    strategy: Optional[str] = None
    parameters_hash: Optional[str] = None
    version: str = "1.0"


class AdvancedCache:
    """
    Advanced caching system with SQLite metadata management and file-based storage.
    Supports data compression, expiration, and intelligent cache eviction.
    """
    
    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.lock = threading.RLock()
        
        # Create directory structure
        self.data_dir = self.cache_dir / "data"
        self.backtest_dir = self.cache_dir / "backtests"
        self.optimization_dir = self.cache_dir / "optimizations"
        self.metadata_db = self.cache_dir / "metadata.db"
        
        for dir_path in [self.data_dir, self.backtest_dir, self.optimization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    cache_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    expires_at TEXT,
                    size_bytes INTEGER NOT NULL,
                    source TEXT,
                    symbol TEXT,
                    interval TEXT,
                    strategy TEXT,
                    parameters_hash TEXT,
                    version TEXT DEFAULT '1.0',
                    file_path TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_metadata (cache_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON cache_metadata (symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON cache_metadata (strategy)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_metadata (expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_metadata (last_accessed)")
    
    def _generate_cache_key(self, cache_type: str, **kwargs) -> str:
        """Generate a unique cache key based on parameters."""
        # Create a deterministic key from sorted parameters
        key_parts = [cache_type]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}={v}")
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_file_path(self, cache_type: str, key: str) -> Path:
        """Get file path for cached item."""
        if cache_type == "data":
            return self.data_dir / f"{key}.gz"
        elif cache_type == "backtest":
            return self.backtest_dir / f"{key}.gz"
        elif cache_type == "optimization":
            return self.optimization_dir / f"{key}.gz"
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def _serialize_and_compress(self, data: Any) -> bytes:
        """Serialize and compress data."""
        if isinstance(data, pd.DataFrame):
            # Use pickle for DataFrames to preserve all metadata
            serialized = pickle.dumps(data)
        else:
            # Use pickle for other objects
            serialized = pickle.dumps(data)
        
        return gzip.compress(serialized)
    
    def _decompress_and_deserialize(self, compressed_data: bytes) -> Any:
        """Decompress and deserialize data."""
        decompressed = gzip.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def _save_metadata(self, metadata: CacheMetadata, file_path: Path):
        """Save metadata to database."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache_metadata
                (key, cache_type, created_at, last_accessed, expires_at, size_bytes,
                 source, symbol, interval, strategy, parameters_hash, version, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.key, metadata.cache_type, metadata.created_at.isoformat(),
                metadata.last_accessed.isoformat(),
                metadata.expires_at.isoformat() if metadata.expires_at else None,
                metadata.size_bytes, metadata.source, metadata.symbol,
                metadata.interval, metadata.strategy, metadata.parameters_hash,
                metadata.version, str(file_path)
            ))
    
    def _get_metadata(self, key: str) -> Optional[CacheMetadata]:
        """Get metadata for a cache key."""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT * FROM cache_metadata WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return CacheMetadata(
                key=row[0],
                cache_type=row[1],
                created_at=datetime.fromisoformat(row[2]),
                last_accessed=datetime.fromisoformat(row[3]),
                expires_at=datetime.fromisoformat(row[4]) if row[4] else None,
                size_bytes=row[5],
                source=row[6],
                symbol=row[7],
                interval=row[8],
                strategy=row[9],
                parameters_hash=row[10],
                version=row[11]
            )
    
    def _update_access_time(self, key: str):
        """Update last access time for a cache key."""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute(
                "UPDATE cache_metadata SET last_accessed = ? WHERE key = ?",
                (datetime.now().isoformat(), key)
            )
    
    def cache_data(self, symbol: str, data: pd.DataFrame, interval: str = "1d",
                   source: str = None, ttl_hours: int = 48) -> str:
        """
        Cache financial data.
        
        Args:
            symbol: Symbol identifier
            data: DataFrame with OHLCV data
            interval: Data interval
            source: Data source name
            ttl_hours: Time to live in hours
            
        Returns:
            Cache key
        """
        with self.lock:
            key = self._generate_cache_key(
                "data", symbol=symbol, interval=interval, source=source
            )
            
            file_path = self._get_file_path("data", key)
            compressed_data = self._serialize_and_compress(data)
            
            # Write compressed data
            file_path.write_bytes(compressed_data)
            
            # Create metadata
            now = datetime.now()
            metadata = CacheMetadata(
                key=key,
                cache_type="data",
                created_at=now,
                last_accessed=now,
                expires_at=now + timedelta(hours=ttl_hours),
                size_bytes=len(compressed_data),
                source=source,
                symbol=symbol,
                interval=interval
            )
            
            self._save_metadata(metadata, file_path)
            self._cleanup_if_needed()
            
            self.logger.info(f"Cached data for {symbol} ({interval}), size: {len(compressed_data)} bytes")
            return key
    
    def get_data(self, symbol: str, interval: str = "1d", 
                 source: str = None) -> Optional[pd.DataFrame]:
        """
        Retrieve cached financial data.
        
        Args:
            symbol: Symbol identifier
            interval: Data interval
            source: Data source name (optional filter)
            
        Returns:
            DataFrame or None if not found/expired
        """
        with self.lock:
            key = self._generate_cache_key(
                "data", symbol=symbol, interval=interval, source=source
            )
            
            metadata = self._get_metadata(key)
            if not metadata:
                return None
            
            # Check expiration
            if metadata.expires_at and datetime.now() > metadata.expires_at:
                self._remove_cache_item(key)
                return None
            
            file_path = self._get_file_path("data", key)
            if not file_path.exists():
                return None
            
            try:
                compressed_data = file_path.read_bytes()
                data = self._decompress_and_deserialize(compressed_data)
                
                # Update access time
                self._update_access_time(key)
                
                self.logger.info(f"Retrieved cached data for {symbol} ({interval})")
                return data
                
            except Exception as e:
                self.logger.warning(f"Failed to read cached data for {symbol}: {e}")
                self._remove_cache_item(key)
                return None
    
    def cache_backtest_result(self, symbol: str, strategy: str, parameters: Dict[str, Any],
                             result: Dict[str, Any], interval: str = "1d", 
                             ttl_days: int = 30) -> str:
        """
        Cache backtest result.
        
        Args:
            symbol: Symbol identifier
            strategy: Strategy name
            parameters: Strategy parameters
            result: Backtest result dictionary
            interval: Data interval
            ttl_days: Time to live in days
            
        Returns:
            Cache key
        """
        with self.lock:
            params_str = json.dumps(parameters, sort_keys=True)
            params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
            
            key = self._generate_cache_key(
                "backtest", symbol=symbol, strategy=strategy, 
                parameters_hash=params_hash, interval=interval
            )
            
            file_path = self._get_file_path("backtest", key)
            
            # Add metadata to result
            result_with_meta = {
                'result': result,
                'symbol': symbol,
                'strategy': strategy,
                'parameters': parameters,
                'interval': interval,
                'cached_at': datetime.now().isoformat()
            }
            
            compressed_data = self._serialize_and_compress(result_with_meta)
            file_path.write_bytes(compressed_data)
            
            # Create metadata
            now = datetime.now()
            metadata = CacheMetadata(
                key=key,
                cache_type="backtest",
                created_at=now,
                last_accessed=now,
                expires_at=now + timedelta(days=ttl_days),
                size_bytes=len(compressed_data),
                symbol=symbol,
                strategy=strategy,
                parameters_hash=params_hash,
                interval=interval
            )
            
            self._save_metadata(metadata, file_path)
            self._cleanup_if_needed()
            
            self.logger.info(f"Cached backtest result for {symbol}/{strategy}")
            return key
    
    def get_backtest_result(self, symbol: str, strategy: str, parameters: Dict[str, Any],
                           interval: str = "1d") -> Optional[Dict[str, Any]]:
        """
        Retrieve cached backtest result.
        
        Args:
            symbol: Symbol identifier
            strategy: Strategy name
            parameters: Strategy parameters
            interval: Data interval
            
        Returns:
            Backtest result dictionary or None
        """
        with self.lock:
            params_str = json.dumps(parameters, sort_keys=True)
            params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
            
            key = self._generate_cache_key(
                "backtest", symbol=symbol, strategy=strategy,
                parameters_hash=params_hash, interval=interval
            )
            
            metadata = self._get_metadata(key)
            if not metadata:
                return None
            
            # Check expiration
            if metadata.expires_at and datetime.now() > metadata.expires_at:
                self._remove_cache_item(key)
                return None
            
            file_path = self._get_file_path("backtest", key)
            if not file_path.exists():
                return None
            
            try:
                compressed_data = file_path.read_bytes()
                cached_data = self._decompress_and_deserialize(compressed_data)
                
                # Update access time
                self._update_access_time(key)
                
                self.logger.info(f"Retrieved cached backtest for {symbol}/{strategy}")
                return cached_data['result']
                
            except Exception as e:
                self.logger.warning(f"Failed to read cached backtest: {e}")
                self._remove_cache_item(key)
                return None
    
    def cache_optimization_result(self, symbol: str, strategy: str, 
                                 optimization_config: Dict[str, Any],
                                 result: Dict[str, Any], interval: str = "1d",
                                 ttl_days: int = 60) -> str:
        """Cache strategy optimization result."""
        with self.lock:
            config_str = json.dumps(optimization_config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
            
            key = self._generate_cache_key(
                "optimization", symbol=symbol, strategy=strategy,
                config_hash=config_hash, interval=interval
            )
            
            file_path = self._get_file_path("optimization", key)
            
            result_with_meta = {
                'result': result,
                'symbol': symbol,
                'strategy': strategy,
                'optimization_config': optimization_config,
                'interval': interval,
                'cached_at': datetime.now().isoformat()
            }
            
            compressed_data = self._serialize_and_compress(result_with_meta)
            file_path.write_bytes(compressed_data)
            
            # Create metadata
            now = datetime.now()
            metadata = CacheMetadata(
                key=key,
                cache_type="optimization",
                created_at=now,
                last_accessed=now,
                expires_at=now + timedelta(days=ttl_days),
                size_bytes=len(compressed_data),
                symbol=symbol,
                strategy=strategy,
                parameters_hash=config_hash,
                interval=interval
            )
            
            self._save_metadata(metadata, file_path)
            self._cleanup_if_needed()
            
            return key
    
    def get_optimization_result(self, symbol: str, strategy: str,
                               optimization_config: Dict[str, Any],
                               interval: str = "1d") -> Optional[Dict[str, Any]]:
        """Retrieve cached optimization result."""
        with self.lock:
            config_str = json.dumps(optimization_config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
            
            key = self._generate_cache_key(
                "optimization", symbol=symbol, strategy=strategy,
                config_hash=config_hash, interval=interval
            )
            
            metadata = self._get_metadata(key)
            if not metadata:
                return None
            
            # Check expiration
            if metadata.expires_at and datetime.now() > metadata.expires_at:
                self._remove_cache_item(key)
                return None
            
            file_path = self._get_file_path("optimization", key)
            if not file_path.exists():
                return None
            
            try:
                compressed_data = file_path.read_bytes()
                cached_data = self._decompress_and_deserialize(compressed_data)
                
                # Update access time
                self._update_access_time(key)
                
                return cached_data['result']
                
            except Exception as e:
                self.logger.warning(f"Failed to read cached optimization: {e}")
                self._remove_cache_item(key)
                return None
    
    def _remove_cache_item(self, key: str):
        """Remove a cache item and its metadata."""
        metadata = self._get_metadata(key)
        if metadata:
            file_path = self._get_file_path(metadata.cache_type, key)
            if file_path.exists():
                file_path.unlink()
        
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("DELETE FROM cache_metadata WHERE key = ?", (key,))
    
    def _cleanup_if_needed(self):
        """Clean up cache if size exceeds limit."""
        total_size = self._get_total_cache_size()
        
        if total_size > self.max_size_bytes:
            self.logger.info(f"Cache size ({total_size/1024**3:.2f} GB) exceeds limit, cleaning up...")
            self._cleanup_expired()
            
            # If still over limit, remove least recently used items
            total_size = self._get_total_cache_size()
            if total_size > self.max_size_bytes:
                self._cleanup_lru()
    
    def _get_total_cache_size(self) -> int:
        """Get total cache size in bytes."""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_metadata")
            result = cursor.fetchone()[0]
            return result or 0
    
    def _cleanup_expired(self):
        """Remove expired cache items."""
        now = datetime.now()
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT key, cache_type FROM cache_metadata WHERE expires_at < ?",
                (now.isoformat(),)
            )
            
            expired_keys = cursor.fetchall()
            
            for key, cache_type in expired_keys:
                file_path = self._get_file_path(cache_type, key)
                if file_path.exists():
                    file_path.unlink()
            
            conn.execute("DELETE FROM cache_metadata WHERE expires_at < ?", (now.isoformat(),))
            
        self.logger.info(f"Removed {len(expired_keys)} expired cache items")
    
    def _cleanup_lru(self):
        """Remove least recently used cache items."""
        target_size = int(self.max_size_bytes * 0.8)  # Clean to 80% of limit
        
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("""
                SELECT key, cache_type, size_bytes 
                FROM cache_metadata 
                ORDER BY last_accessed ASC
            """)
            
            current_size = self._get_total_cache_size()
            removed_count = 0
            
            for key, cache_type, size_bytes in cursor:
                if current_size <= target_size:
                    break
                
                file_path = self._get_file_path(cache_type, key)
                if file_path.exists():
                    file_path.unlink()
                
                conn.execute("DELETE FROM cache_metadata WHERE key = ?", (key,))
                current_size -= size_bytes
                removed_count += 1
        
        self.logger.info(f"Removed {removed_count} LRU cache items")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("""
                SELECT 
                    cache_type,
                    COUNT(*) as count,
                    SUM(size_bytes) as total_size,
                    AVG(size_bytes) as avg_size,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM cache_metadata 
                GROUP BY cache_type
            """)
            
            stats_by_type = {}
            for row in cursor:
                stats_by_type[row[0]] = {
                    'count': row[1],
                    'total_size_bytes': row[2] or 0,
                    'avg_size_bytes': row[3] or 0,
                    'oldest': row[4],
                    'newest': row[5]
                }
            
            total_size = self._get_total_cache_size()
            
            return {
                'total_size_bytes': total_size,
                'total_size_gb': total_size / 1024**3,
                'max_size_gb': self.max_size_bytes / 1024**3,
                'utilization_percent': (total_size / self.max_size_bytes) * 100,
                'by_type': stats_by_type
            }
    
    def clear_cache(self, cache_type: str = None, symbol: str = None, 
                   strategy: str = None, older_than_days: int = None):
        """
        Clear cache items based on filters.
        
        Args:
            cache_type: Clear specific cache type ('data', 'backtest', 'optimization')
            symbol: Clear items for specific symbol
            strategy: Clear items for specific strategy
            older_than_days: Clear items older than specified days
        """
        with self.lock:
            conditions = []
            params = []
            
            if cache_type:
                conditions.append("cache_type = ?")
                params.append(cache_type)
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if strategy:
                conditions.append("strategy = ?")
                params.append(strategy)
            
            if older_than_days:
                cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                conditions.append("created_at < ?")
                params.append(cutoff)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            with sqlite3.connect(self.metadata_db) as conn:
                cursor = conn.execute(
                    f"SELECT key, cache_type FROM cache_metadata WHERE {where_clause}",
                    params
                )
                
                items_to_remove = cursor.fetchall()
                
                # Remove files
                for key, ct in items_to_remove:
                    file_path = self._get_file_path(ct, key)
                    if file_path.exists():
                        file_path.unlink()
                
                # Remove metadata
                conn.execute(
                    f"DELETE FROM cache_metadata WHERE {where_clause}",
                    params
                )
            
            self.logger.info(f"Cleared {len(items_to_remove)} cache items")


# Global cache instance
advanced_cache = AdvancedCache()
