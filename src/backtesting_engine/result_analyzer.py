import pandas as pd

class ResultAnalyzer:
    @staticmethod
    def analyze(results):
        """Generates key performance metrics"""
        return {
            "final_value": results.broker.getvalue(),
            "pnl": results.broker.getvalue() - 10000,
            "trade_count": len(results._trades)
        }