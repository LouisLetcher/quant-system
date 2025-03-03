import logging
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetService:
    """Handles fetching and processing asset data from Yahoo Finance."""

    @staticmethod
    def get_asset_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical asset data.

        :param ticker: Stock symbol (e.g., "AAPL").
        :param start_date: Start date in YYYY-MM-DD format.
        :param end_date: End date in YYYY-MM-DD format.
        :return: Pandas DataFrame with historical asset data.
        """
        try:
            logger.info(f"🔍 Fetching asset data for {ticker} from {start_date} to {end_date}...")
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                logger.warning(f"⚠️ No data found for {ticker}")
                return pd.DataFrame()
            
            logger.info(f"✅ Data fetched: {df.shape[0]} rows.")
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching asset data: {e}")
            return pd.DataFrame()