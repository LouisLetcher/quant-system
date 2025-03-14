import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from matplotlib.dates import DateFormatter
import numpy as np

class ReportVisualizer:
    """Generates charts and visualizations for reports."""

    @staticmethod
    def plot_performance(data: pd.DataFrame, output_file: str = None):
        """Plots backtest equity curve."""
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data["equity"], label="Equity Curve", color="#1f77b4")
        plt.title("Backtest Performance")
        plt.xlabel("Date")
        plt.ylabel("Equity Value")
        plt.legend()
        plt.grid(alpha=0.3)
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
            return output_file
        else:
            # Return base64 encoded image for HTML embedding
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png')
            plt.close()
            img_bytes.seek(0)
            return base64.b64encode(img_bytes.read()).decode('utf-8')
    
    @staticmethod
    def plot_drawdown(data: pd.DataFrame):
        """Generates drawdown chart for backtest results."""
        if 'drawdown' not in data.columns and 'equity' in data.columns:
            # Calculate drawdown if not present
            equity = data['equity'].values
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak * 100
            data = data.copy()
            data['drawdown'] = drawdown
        
        plt.figure(figsize=(10, 5))
        plt.fill_between(data.index, data['drawdown'], 0, color='red', alpha=0.3)
        plt.plot(data.index, data['drawdown'], color='red', label='Drawdown %')
        plt.title("Drawdown Over Time")
        plt.xlabel("Date")
        plt.ylabel("Drawdown %")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Return base64 encoded image
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png')
        plt.close()
        img_bytes.seek(0)
        return base64.b64encode(img_bytes.read()).decode('utf-8')
    @staticmethod
    def plot_equity_and_drawdown(equity_df):
        """
        Plots equity curve and drawdown in a single chart and returns as base64 encoded image.
    
        Args:
            equity_df: DataFrame with equity curve data
        
        Returns:
            Base64 encoded image string
        """
        import matplotlib.pyplot as plt
        import io
        import base64
    
        # Calculate drawdown
        peak = equity_df['equity'].cummax()
        drawdown = -100 * (1 - equity_df['equity'] / peak)
    
        # Create figure with two subplots sharing x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
        # Plot equity curve on top subplot
        ax1.plot(equity_df.index, equity_df['equity'], label='Equity', color='blue')
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity Value')
        ax1.grid(True)
    
        # Plot drawdown on bottom subplot
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown %')
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Date')
        ax2.grid(True)
    
        # Format x-axis dates
        fig.autofmt_xdate()
    
        # Adjust layout
        plt.tight_layout()
    
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
    
        return image_base64