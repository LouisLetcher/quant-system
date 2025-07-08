import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.backtesting_engine.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):

    @patch("src.backtesting_engine.data_loader.yf.download")
    def test_load_data_success(self, mock_download):
        # Setup mock data
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [98, 99, 100],
                "Close": [103, 104, 105],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_download.return_value = mock_data

        # Call the method
        result = DataLoader.load_data("AAPL", period="1mo", interval="1d")

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertTrue(
            all(
                col in result.columns
                for col in ["Open", "High", "Low", "Close", "Volume"]
            )
        )
        mock_download.assert_called_once_with("AAPL", period="1mo", interval="1d")

    @patch("src.backtesting_engine.data_loader.yf.download")
    def test_load_data_empty_result(self, mock_download):
        # Setup mock to return empty DataFrame
        mock_download.return_value = pd.DataFrame()

        # Call the method
        result = DataLoader.load_data("INVALID", period="1mo", interval="1d")

        # Assertions
        self.assertIsNone(result)
        mock_download.assert_called_once()

    @patch("src.backtesting_engine.data_loader.yf.download")
    def test_load_data_with_cache(self, mock_download):
        # Setup mock data
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [98, 99, 100],
                "Close": [103, 104, 105],
                "Volume": [1000, 1100, 1200],
            },
            index=pd.date_range("2023-01-01", periods=3),
        )
        mock_download.return_value = mock_data

        # Call the method twice
        DataLoader.load_data("AAPL", period="1mo", interval="1d")
        DataLoader.load_data("AAPL", period="1mo", interval="1d")

        # Verify download was called only once (due to caching)
        mock_download.assert_called_once()


if __name__ == "__main__":
    unittest.main()
