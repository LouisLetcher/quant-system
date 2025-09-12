from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .base import DataSource
from .cache import ParquetCache
from .ratelimiter import RateLimiter

YFINANCE_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
    "90m": "90m",
    "1h": "60m",
    "4h": "1h",  # will resample after fetching 1h
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


class YFinanceSource(DataSource):
    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)
        self.cache = ParquetCache(cache_dir)
        # Space requests to avoid Yahoo rate limits
        self._limiter = RateLimiter(min_interval=1.0)
        # Optional yfinance-cache integration
        self._yfc = None
        try:
            from yfinance_cache import YFCache  # type: ignore

            # Use a dedicated cache dir inside data cache
            self._yfc = YFCache(cache_dir=str(Path(cache_dir) / "yfinance-http"))
        except Exception:
            self._yfc = None

    def fetch(self, symbol: str, timeframe: str, only_cached: bool = False) -> pd.DataFrame:
        tf = timeframe.lower()
        cached = self.cache.load("yfinance", symbol, tf)
        if cached is not None and len(cached) > 0:
            return cached
        if only_cached:
            raise RuntimeError(f"Cache miss for {symbol} {tf} (yfinance) with only_cached=True")

        if tf not in YFINANCE_TF_MAP:
            raise ValueError(f"Unsupported timeframe for yfinance: {tf}")

        interval = YFINANCE_TF_MAP[tf]
        self._limiter.acquire()

        # Retry/backoff wrapper for yfinance
        def try_download() -> pd.DataFrame:
            if self._yfc is not None:
                try:
                    ticker = self._yfc.ticker.Ticker(symbol)  # type: ignore[attr-defined]
                    return ticker.history(period="max", interval=interval, auto_adjust=False)
                except Exception:
                    pass
            return yf.download(
                symbol, period="max", interval=interval, auto_adjust=False, progress=False
            )

        backoff = 1.0
        max_backoff = 30.0
        for _ in range(5):
            try:
                df = try_download()
                break
            except Exception:
                time.sleep(backoff)
                backoff = min(max_backoff, backoff * 2)
        else:
            # last attempt
            df = try_download()
        if df.empty:
            raise RuntimeError(f"No data for {symbol} {tf}")
        df = df.rename(columns={c: c.split()[0] for c in df.columns})
        df.index = df.index.tz_localize(None)

        # Resample 4h if requested
        if tf == "4h":
            df = (
                df.resample("4H")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna(how="any")
            )

        self.cache.save("yfinance", symbol, tf, df)
        return df
