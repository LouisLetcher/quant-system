from __future__ import annotations


class ObjectiveFunction:
    """Defines the function to evaluate performance."""

    @staticmethod
    def evaluate(result):
        """Evaluates strategy performance using the Sharpe Ratio."""
        return result["sharpe_ratio"]
