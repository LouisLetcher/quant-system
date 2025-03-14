from backtesting import Strategy

class BaseStrategy(Strategy):
    """Base class for all trading strategies using Backtesting.py.
    
    This class directly extends the Backtesting.py Strategy class and adds
    common functionality needed across all strategies.
    """
    
    def init(self):
        """Initialize strategy indicators and parameters.
        This method is called once at the start of the backtest.
        """
        self.name = self.__class__.__name__
        # Common initialization code here
        def next(self):
            """Trading logic for each bar.
        
            This method should be overridden by child strategy classes
            to implement specific trading logic.
            """
            # Base implementation does nothing
            pass
    
    def position_size(self, price):
        """Calculate position size based on available capital and risk parameters.
        
        Args:
            price: Current price for the asset
            
        Returns:
            int: Number of shares to buy/sell
        """
        # Default implementation - use at most initial capital
        max_capital = getattr(self, '_initial_capital', self.equity)
        return int(max_capital / price)
    
    def buy_with_size_control(self, **kwargs):
        """Buy with position sizing control to prevent excessive leverage.
        
        This method wraps the standard buy() method with position sizing logic.
        """
        price = self.data.Close[-1]
        
        # If size is not specified, calculate it
        if 'size' not in kwargs:
            kwargs['size'] = self.position_size(price)
        else:
            # Ensure size doesn't exceed maximum based on initial capital
            max_size = self.position_size(price)
            kwargs['size'] = min(int(kwargs['size']), max_size)
        
        # Call the original buy method from Backtesting.py Strategy
        return super().buy(**kwargs)
