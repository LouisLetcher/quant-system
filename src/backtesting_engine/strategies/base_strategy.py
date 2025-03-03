from backtesting import Strategy

class BaseStrategy(Strategy):
    """Base class for all trading strategies using Backtesting.py."""
    
    # Remove the custom __init__ method completely OR modify it to accept all parameters
    
    def init(self):
        """Initialize strategy indicators and parameters."""
        # Override in child classes
        self.name = self.__class__.__name__
        pass
    
    def next(self):
        """Main strategy logic, executed for each bar."""
        # Override in child classes
        pass
