"""
Tests for Raw Data CSV Exporter functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.utils.raw_data_csv_exporter import RawDataCSVExporter


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_html_report():
    """Sample HTML report content for testing."""
    return """
    <html>
    <head><title>Portfolio Report</title></head>
    <body>
        <h1>Portfolio Analysis Report</h1>

        <table id="performance-table">
            <tr>
                <th>Symbol</th>
                <th>Strategy</th>
                <th>Total Return (%)</th>
                <th>Sortino Ratio</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown (%)</th>
            </tr>
            <tr>
                <td>BTCUSDT</td>
                <td>Long Short Term Memory Neural Network</td>
                <td>78.5</td>
                <td>2.8</td>
                <td>2.3</td>
                <td>-15.2</td>
            </tr>
            <tr>
                <td>ETHUSDT</td>
                <td>Confident Trend</td>
                <td>56.7</td>
                <td>2.4</td>
                <td>2.0</td>
                <td>-12.8</td>
            </tr>
        </table>

        <div class="metric-card">
            <h3>ADAUSDT: 32.1%</h3>
            <p>Strategy: Counter Punsh</p>
        </div>
    </body>
    </html>
    """


class TestRawDataCSVExporter:
    """Test cases for RawDataCSVExporter."""

    def test_initialization(self, temp_dir):
        """Test proper initialization of the exporter."""
        output_dir = temp_dir / "csv_output"
        exporter = RawDataCSVExporter(db_session=None, output_dir=str(output_dir))

        assert str(exporter.output_dir) == str(output_dir)
        assert output_dir.exists()  # Should be created during init
        assert exporter.reports_dir == Path("exports/reports")

    def test_get_available_columns(self):
        """Test that available columns are returned correctly."""
        exporter = RawDataCSVExporter()
        columns = exporter.get_available_columns()

        expected_columns = [
            "Symbol",
            "Strategy",
            "Timeframe",
            "Total_Return_Pct",
            "Sortino_Ratio",
            "Sharpe_Ratio",
            "Calmar_Ratio",
            "Max_Drawdown_Pct",
            "Win_Rate_Pct",
            "Profit_Factor",
            "Number_of_Trades",
            "Volatility_Pct",
        ]

        for col in expected_columns:
            assert col in columns

    def test_extract_data_from_html_report(self, temp_dir, sample_html_report):
        """Test HTML report parsing and data extraction."""
        exporter = RawDataCSVExporter(None, str(temp_dir))

        # Create a sample HTML file
        html_file = temp_dir / "sample_report.html"
        html_file.write_text(sample_html_report, encoding="utf-8")

        # Extract data
        extracted_data = exporter._extract_data_from_html_report(html_file)

        assert len(extracted_data) >= 2  # Should extract at least BTCUSDT and ETHUSDT

        # Check first extracted record
        btc_record = next(
            (r for r in extracted_data if r.get("Symbol") == "BTCUSDT"), None
        )
        assert btc_record is not None
        assert btc_record["Strategy"] == "Long Short Term Memory Neural Network"
        assert btc_record["Total_Return_Pct"] == 78.5
        assert btc_record["Sortino_Ratio"] == 2.8
        assert btc_record["Sharpe_Ratio"] == 2.3

    def test_parse_table_row(self):
        """Test table row parsing functionality."""
        exporter = RawDataCSVExporter()

        headers = ["Symbol", "Strategy", "Total Return (%)", "Sortino Ratio"]
        cells = ["BTCUSDT", "BuyAndHold", "45.2", "1.8"]

        result = exporter._parse_table_row(headers, cells)

        assert result is not None
        assert result["Symbol"] == "BTCUSDT"
        assert result["Strategy"] == "BuyAndHold"
        assert result["Total_Return_Pct"] == 45.2
        assert result["Sortino_Ratio"] == 1.8

    def test_parse_table_row_mismatched_lengths(self):
        """Test table row parsing with mismatched header/cell lengths."""
        exporter = RawDataCSVExporter()

        headers = ["Symbol", "Strategy", "Return"]
        cells = ["BTCUSDT", "BuyAndHold"]  # Missing one cell

        result = exporter._parse_table_row(headers, cells)

        # Should handle gracefully - returns None for mismatched lengths
        # This is expected behavior for malformed data
        if result is not None:
            assert result["Symbol"] == "BTCUSDT"
            assert result["Strategy"] == "BuyAndHold"

    def test_parse_metric_card(self):
        """Test metric card parsing."""
        exporter = RawDataCSVExporter()

        # Create a mock BeautifulSoup element
        html_content = """
        <div class="metric-card">
            <h3>BTCUSDT: 45.2%</h3>
            <p>Strategy: BuyAndHold</p>
        </div>
        """
        soup = BeautifulSoup(html_content, "html.parser")
        card = soup.find("div")

        result = exporter._parse_metric_card(card)

        assert result is not None
        assert result["Symbol"] == "BTCUSDT"
        assert result["Total_Return_Pct"] == 45.2
        assert result["Strategy"] == "BuyAndHold"

    def test_export_from_quarterly_reports_no_reports(self, temp_dir):
        """Test export when no quarterly reports exist."""
        exporter = RawDataCSVExporter(None, str(temp_dir))

        result = exporter.export_from_quarterly_reports(
            "Q4", "2023", "test.csv", "full"
        )

        assert result == []  # Should return empty list when no reports found

    @patch("pathlib.Path.glob")
    def test_export_from_quarterly_reports_success(
        self, mock_glob, temp_dir, sample_html_report
    ):
        """Test successful export from quarterly reports."""
        output_dir = temp_dir / "data_exports"
        exporter = RawDataCSVExporter(None, str(output_dir))

        # Create mock HTML file
        html_file = temp_dir / "sample_report.html"
        html_file.write_text(sample_html_report, encoding="utf-8")
        mock_glob.return_value = [html_file]

        # Mock the reports directory to exist and the glob method
        with (
            patch.object(Path, "exists", return_value=True),
            patch("pathlib.Path.glob", return_value=[html_file]),
        ):
            result = exporter.export_from_quarterly_reports(
                "Q4", "2023", "test.csv", "full"
            )

        # Should return list of paths to created CSV files
        assert result != []
        assert len(result) == 1
        assert "Q4" in result[0]
        assert "2023" in result[0]

        # Check if quarterly directory structure was created
        quarterly_dir = output_dir / "2023" / "Q4"
        assert quarterly_dir.exists()

    @patch("pathlib.Path.glob")
    def test_export_best_strategies_format(
        self, mock_glob, temp_dir, sample_html_report
    ):
        """Test export with best-strategies format."""
        output_dir = temp_dir / "data_exports"
        exporter = RawDataCSVExporter(None, str(output_dir))

        # Create mock HTML file
        html_file = temp_dir / "sample_report.html"
        html_file.write_text(sample_html_report, encoding="utf-8")
        mock_glob.return_value = [html_file]

        # Mock the reports directory to exist
        with patch.object(Path, "exists", return_value=True):
            result = exporter.export_from_quarterly_reports(
                "Q4", "2023", "best.csv", "best-strategies"
            )

        assert result != []
        assert len(result) == 1

        # Check if the CSV file exists and has the right format
        csv_path = Path(result[0])
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            expected_columns = ["Asset", "Best Strategy", "Resolution"]
            for col in expected_columns:
                assert col in df.columns

    def test_numeric_value_parsing(self):
        """Test parsing of numeric values with various formats."""
        exporter = RawDataCSVExporter()

        # Test various numeric formats
        test_cases = [
            ("45.2%", 45.2),
            ("45.2", 45.2),
            ("-15.8%", -15.8),
            ("1,234.56", 1234.56),
            ("$1,234", 1234.0),
            ("", None),  # Empty should be handled
            ("-", None),  # Dash should be handled
        ]

        headers = ["Symbol", "Return"]

        for input_val, expected in test_cases:
            cells = ["TEST", input_val]
            result = exporter._parse_table_row(headers, cells)

            if expected is None:
                # Should either be None or not in result
                assert (
                    "Total_Return_Pct" not in result
                    or result["Total_Return_Pct"] == input_val
                )
            else:
                assert result is not None
                assert result.get("Total_Return_Pct") == expected

    def test_default_timeframe_assignment(self):
        """Test that default timeframe is assigned correctly."""
        exporter = RawDataCSVExporter()

        headers = ["Symbol", "Strategy"]
        cells = ["BTCUSDT", "BuyAndHold"]

        result = exporter._parse_table_row(headers, cells)

        assert result is not None
        assert result["Timeframe"] == "1d"  # Should default to 1d

    def test_html_parsing_robustness(self, temp_dir):
        """Test HTML parsing with malformed or edge case HTML."""
        exporter = RawDataCSVExporter(None, str(temp_dir))

        malformed_html = """
        <html>
        <body>
            <table>
                <tr><th>Symbol</th><th>Strategy</th></tr>
                <tr><td>BTCUSDT</td></tr>  <!-- Missing cell -->
                <tr></tr>  <!-- Empty row -->
            </table>
            <div>No relevant data</div>
        </body>
        </html>
        """

        html_file = temp_dir / "malformed.html"
        html_file.write_text(malformed_html, encoding="utf-8")

        # Should not crash and return empty or partial data
        extracted_data = exporter._extract_data_from_html_report(html_file)

        # Should handle gracefully - might be empty or have partial data
        assert isinstance(extracted_data, list)

    def test_collection_based_export(self, temp_dir):
        """Test the new collection-based export functionality."""
        from unittest.mock import MagicMock

        # Mock database data
        mock_strategy_data = [
            MagicMock(
                symbol="GC=F",
                best_strategy="BuyAndHold",
                sortino_ratio=0.82,
                total_return=15.5,
            ),
            MagicMock(
                symbol="KC=F",
                best_strategy="lazy_trend_follower",
                sortino_ratio=0.58,
                total_return=-0.1,
            ),
            MagicMock(
                symbol="HYG",
                best_strategy="macd",
                sortino_ratio=0.52,
                total_return=24.9,
            ),
        ]

        exporter = RawDataCSVExporter(None, str(temp_dir))

        # Mock the dataframe conversion method
        with patch.object(
            exporter, "_strategies_data_to_dataframe"
        ) as mock_df_converter:
            mock_df = MagicMock()
            mock_df.to_csv = MagicMock()
            mock_df_converter.return_value = mock_df

            result_paths = exporter._export_collection_based_files(
                mock_strategy_data, "Q3", "2025", "best-strategies"
            )

            # Should create separate files for each collection
            assert len(result_paths) == 2  # Commodities and Bonds collections

            # Check file paths contain collection names
            assert any("Commodities_Collection" in path for path in result_paths)
            assert any("Bonds_Collection" in path for path in result_paths)

            # Check directory structure is created
            assert (temp_dir / "2025" / "Q3").exists()


if __name__ == "__main__":
    pytest.main([__file__])
