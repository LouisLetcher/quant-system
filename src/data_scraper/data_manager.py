from src.data_scraper.scraper import Scraper
from src.data_scraper.cache import Cache
from src.data_scraper.data_cleaner import DataCleaner
import pandas as pd
from typing import List, Dict, Optional

class DataManager:
    @staticmethod
    def get_stock_data(ticker, start_date=None, end_date=None, interval="1d"):
        """Get stock data from cache or fetch it if not available."""
        # First try to load from cache
        data = Cache.load_from_cache(ticker, interval)
        
        if data is not None:
            print(f"✅ Data retrieved for {ticker} from cache: {len(data)} rows.")
            return data
        
        print(f"🔍 Checking cache for {ticker} data...")
        
        try:
            # If not in cache, fetch from Yahoo Finance
            # The rate limiting is now handled inside the Scraper class
            data = Scraper.fetch_data(ticker, start_date, end_date, interval)
            
            # Clean data
            data = DataCleaner.remove_missing_values(data)
            
            # Save to cache
            Cache.save_to_cache(ticker, data, interval)
            
            print(f"✅ Data retrieved for {ticker}: {len(data)} rows.")
            return data
        except Exception as e:
            print(f"❌ Error getting data for {ticker}: {e}")
            return None
    
    @staticmethod
    def get_batch_stock_data(tickers: List[str], start_date=None, end_date=None, interval="1d") -> Dict[str, pd.DataFrame]:
        """
        Get stock data for multiple tickers with batch processing and rate limiting.
        First checks cache, then fetches missing data via batch requests.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1d', '1wk', etc.)
            
        Returns:
            Dictionary mapping tickers to their respective DataFrames
        """
        result = {}
        missing_tickers = []
        
        # First check cache for all tickers
        for ticker in tickers:
            data = Cache.load_from_cache(ticker, interval)
            if data is not None:
                result[ticker] = data
                print(f"✅ Data retrieved for {ticker} from cache: {len(data)} rows.")
            else:
                missing_tickers.append(ticker)
        
        if not missing_tickers:
            return result
        
        # Fetch missing tickers with a batch request
        print(f"🔍 Fetching data for {len(missing_tickers)} tickers not found in cache...")
        try:
            batch_data = Scraper.fetch_batch_data(missing_tickers, start_date, end_date, interval)
            
            # Process and cache each ticker's data
            for ticker, data in batch_data.items():
                if data is not None and not data.empty:
                    # Clean data
                    clean_data = DataCleaner.remove_missing_values(data)
                    
                    # Save to cache
                    Cache.save_to_cache(ticker, clean_data, interval)
                    
                    # Add to result
                    result[ticker] = clean_data
            
            # Check if any tickers failed to fetch
            failed_tickers = set(missing_tickers) - set(batch_data.keys())
            if failed_tickers:
                print(f"❌ Failed to retrieve data for: {', '.join(failed_tickers)}")
                
            return result
            
        except Exception as e:
            print(f"❌ Error in batch data retrieval: {e}")
            
            # Fall back to individual requests for each ticker with retry logic
            print("⚠️ Falling back to individual requests...")
            for ticker in missing_tickers:
                for retry in range(Scraper._max_retries):
                    try:
                        data = DataManager.get_stock_data(ticker, start_date, end_date, interval)
                        if data is not None:
                            result[ticker] = data
                        break
                    except Exception as retry_error:
                        if retry < Scraper._max_retries - 1:
                            wait_time = (Scraper._backoff_factor ** retry) * Scraper._min_interval
                            print(f"⚠️ Retry {retry+1}/{Scraper._max_retries} for {ticker} after {wait_time:.2f}s")
                
            return result
