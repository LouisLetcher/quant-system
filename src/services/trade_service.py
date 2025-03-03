import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeService:
    """Executes trades in a simulated environment."""

    def __init__(self, starting_cash=10000):
        self.cash = starting_cash
        self.positions = {}

    def execute_trade(self, ticker: str, action: str, price: float, quantity: int):
        """
        Executes a simulated trade.

        :param ticker: Stock symbol.
        :param action: "BUY" or "SELL".
        :param price: Trade price.
        :param quantity: Trade quantity.
        """
        if action == "BUY":
            total_cost = price * quantity
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.positions[ticker] = self.positions.get(ticker, 0) + quantity
                logger.info(f"✅ Bought {quantity} shares of {ticker} at ${price:.2f}. Cash left: ${self.cash:.2f}")
            else:
                logger.warning(f"⚠️ Not enough cash to buy {quantity} shares of {ticker}.")
        
        elif action == "SELL":
            if self.positions.get(ticker, 0) >= quantity:
                self.cash += price * quantity
                self.positions[ticker] -= quantity
                logger.info(f"✅ Sold {quantity} shares of {ticker} at ${price:.2f}. New cash balance: ${self.cash:.2f}")
            else:
                logger.warning(f"⚠️ Not enough shares to sell {quantity} of {ticker}.")
        else:
            logger.error(f"❌ Invalid trade action: {action}")