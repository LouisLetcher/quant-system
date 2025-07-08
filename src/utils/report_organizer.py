"""
Report organizer utility for quarterly report management.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class ReportOrganizer:
    """Organizes reports by quarter and year, ensuring single report per portfolio per quarter."""

    def __init__(self, base_reports_dir: str = "reports_output"):
        self.base_reports_dir = Path(base_reports_dir)
        self.base_reports_dir.mkdir(exist_ok=True)

    def get_quarter_from_date(self, date: datetime) -> tuple[int, int]:
        """Get year and quarter from date."""
        quarter = (date.month - 1) // 3 + 1
        return date.year, quarter

    def get_quarterly_dir(self, year: int, quarter: int) -> Path:
        """Get the quarterly directory path."""
        return self.base_reports_dir / f"{year}" / f"Q{quarter}"

    def get_portfolio_name_from_filename(self, filename: str) -> Optional[str]:
        """Extract portfolio name from report filename."""
        if filename.startswith("portfolio_report_"):
            # Format: portfolio_report_{portfolio_name}_{timestamp}.html
            parts = filename.replace("portfolio_report_", "").split("_")
            if len(parts) >= 2:
                # Take all parts except the last one (timestamp)
                return "_".join(parts[:-1])
        return None

    def organize_report(
        self,
        report_path: str,
        portfolio_name: str,
        report_date: Optional[datetime] = None,
    ) -> Path:
        """
        Organize a report into quarterly structure.

        Args:
            report_path: Path to the report file
            portfolio_name: Name of the portfolio
            report_date: Date of the report (defaults to current date)

        Returns:
            Path to the organized report
        """
        if report_date is None:
            report_date = datetime.now()

        year, quarter = self.get_quarter_from_date(report_date)
        quarterly_dir = self.get_quarterly_dir(year, quarter)
        quarterly_dir.mkdir(parents=True, exist_ok=True)

        # Clean portfolio name for filename
        clean_portfolio_name = portfolio_name.replace(" ", "_").replace("/", "_")

        # New filename format: {portfolio_name}_Q{quarter}_{year}.html
        new_filename = f"{clean_portfolio_name}_Q{quarter}_{year}.html"
        target_path = quarterly_dir / new_filename

        # Check if report already exists for this portfolio/quarter
        if target_path.exists():
            print(f"Overriding existing report: {target_path}")
            target_path.unlink()  # Remove existing report

        # Copy/move the report
        source_path = Path(report_path)
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            print(f"Report organized: {target_path}")

            # Also handle compressed version if it exists
            compressed_source = source_path.with_suffix(".html.gz")
            if compressed_source.exists():
                compressed_target = target_path.with_suffix(".html.gz")
                shutil.copy2(compressed_source, compressed_target)

        return target_path

    def organize_existing_reports(self):
        """Organize all existing reports in reports_output."""
        print("Organizing existing reports...")

        # Find all portfolio reports
        for report_file in self.base_reports_dir.glob("portfolio_report_*.html"):
            portfolio_name = self.get_portfolio_name_from_filename(report_file.name)

            if portfolio_name:
                # Try to extract date from filename timestamp
                try:
                    filename_parts = report_file.stem.split("_")
                    timestamp_part = filename_parts[-1]  # Last part should be timestamp

                    # Parse timestamp (format: YYYYMMDD_HHMMSS)
                    if len(timestamp_part) >= 8:
                        date_part = timestamp_part[:8]  # YYYYMMDD
                        report_date = datetime.strptime(date_part, "%Y%m%d")
                    else:
                        report_date = datetime.now()

                except (ValueError, IndexError):
                    # If parsing fails, use current date
                    report_date = datetime.now()

                # Organize the report
                self.organize_report(str(report_file), portfolio_name, report_date)

                # Remove original file after organizing
                report_file.unlink()

                # Also remove compressed version if exists
                compressed_file = report_file.with_suffix(".html.gz")
                if compressed_file.exists():
                    compressed_file.unlink()

    def get_latest_report(self, portfolio_name: str) -> Optional[Path]:
        """Get the latest report for a portfolio."""
        clean_portfolio_name = portfolio_name.replace(" ", "_").replace("/", "_")

        latest_report = None
        latest_date = None

        # Search through all quarterly directories
        for year_dir in self.base_reports_dir.glob("????"):
            if year_dir.is_dir():
                for quarter_dir in year_dir.glob("Q?"):
                    if quarter_dir.is_dir():
                        report_path = (
                            quarter_dir
                            / f"{clean_portfolio_name}_Q{quarter_dir.name[1]}_{year_dir.name}.html"
                        )
                        if report_path.exists():
                            year = int(year_dir.name)
                            quarter = int(quarter_dir.name[1])
                            date = datetime(year, (quarter - 1) * 3 + 1, 1)

                            if latest_date is None or date > latest_date:
                                latest_date = date
                                latest_report = report_path

        return latest_report

    def list_quarterly_reports(self, year: Optional[int] = None) -> Dict[str, list]:
        """List all quarterly reports, optionally filtered by year."""
        reports = {}

        year_pattern = str(year) if year else "????"

        for year_dir in self.base_reports_dir.glob(year_pattern):
            if year_dir.is_dir():
                year_str = year_dir.name
                reports[year_str] = {}

                for quarter_dir in year_dir.glob("Q?"):
                    if quarter_dir.is_dir():
                        quarter_str = quarter_dir.name
                        reports[year_str][quarter_str] = []

                        for report_file in quarter_dir.glob("*.html"):
                            reports[year_str][quarter_str].append(report_file.name)

        return reports

    def cleanup_old_reports(self, keep_quarters: int = 8):
        """Clean up old reports, keeping only the last N quarters."""
        current_date = datetime.now()
        current_year, current_quarter = self.get_quarter_from_date(current_date)

        # Calculate cutoff date
        cutoff_quarters = []
        year, quarter = current_year, current_quarter

        for _ in range(keep_quarters):
            cutoff_quarters.append((year, quarter))
            quarter -= 1
            if quarter < 1:
                quarter = 4
                year -= 1

        # Remove directories older than cutoff
        for year_dir in self.base_reports_dir.glob("????"):
            if year_dir.is_dir():
                year_int = int(year_dir.name)

                for quarter_dir in year_dir.glob("Q?"):
                    if quarter_dir.is_dir():
                        quarter_int = int(quarter_dir.name[1])

                        if (year_int, quarter_int) not in cutoff_quarters:
                            print(f"Removing old reports: {quarter_dir}")
                            shutil.rmtree(quarter_dir)

                # Remove empty year directories
                if not list(year_dir.glob("Q?")):
                    print(f"Removing empty year directory: {year_dir}")
                    year_dir.rmdir()
