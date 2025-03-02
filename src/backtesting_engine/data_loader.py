import yfinance as yf
import pandas as pd

class DataLoader:
    @staticmethod
    def load_data(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        data.index = pd.to_datetime(data.index)
        return data