class TradeExecution:
    """Handles execution of trades within backtests"""

    def __init__(self, cash=10000):
        self.cash = cash
        self.positions = {}

    def execute_trade(self, asset, price, quantity, trade_type):
        """Executes a trade and updates portfolio"""
        trade_cost = price * quantity

        if trade_type == "BUY":
            if self.cash >= trade_cost:
                self.cash -= trade_cost
                self.positions[asset] = self.positions.get(asset, 0) + quantity
                return {"status": "success", "action": "BUY", "price": price, "quantity": quantity}
            return {"status": "failed", "reason": "Insufficient cash"}

        elif trade_type == "SELL":
            if self.positions.get(asset, 0) >= quantity:
                self.cash += trade_cost
                self.positions[asset] -= quantity
                return {"status": "success", "action": "SELL", "price": price, "quantity": quantity}
            return {"status": "failed", "reason": "Insufficient shares"}

        return {"status": "failed", "reason": "Invalid trade type"}