import matplotlib.pyplot as plt
import pandas as pd

class ReportVisualizer:
    """Generates charts and visualizations for reports."""

    @staticmethod
    def plot_performance(data: pd.DataFrame, output_file: str):
        """Plots backtest equity curve."""
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data["equity"], label="Equity Curve")
        plt.title("Backtest Performance")
        plt.xlabel("Date")
        plt.ylabel("Equity Value")
        plt.legend()
        plt.grid()
        plt.savefig(output_file)
        plt.close()