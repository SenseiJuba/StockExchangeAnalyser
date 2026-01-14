import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

DEFAULT_MA_PERIODS = [20, 50, 200]
RSI_PERIOD = 14
VOLATILITY_WINDOW = 20


class StockAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
        self.indicators = {}

    def calculate_moving_averages(self, periods=None):
        periods = periods or DEFAULT_MA_PERIODS
        for p in periods:
            col = f'MA_{p}'
            self.data[col] = self.data['Close'].rolling(window=p).mean()
            self.indicators[col] = self.data[col]
        return self.data

    def calculate_rsi(self, period=RSI_PERIOD):
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        self.data['RSI'] = rsi
        self.indicators['RSI'] = rsi
        return rsi

    def calculate_macd(self):
        ema_12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = signal
        self.data['MACD_Histogram'] = hist
        self.indicators['MACD'] = macd
        self.indicators['MACD_Signal'] = signal
        return macd, signal, hist

    def calculate_volatility(self, period=VOLATILITY_WINDOW):
        returns = self.data['Close'].pct_change()
        vol = returns.rolling(window=period).std()
        self.data['Volatility'] = vol
        self.indicators['Volatility'] = vol
        return vol

    def get_stats(self):
        close = self.data['Close']
        vol = self.data.get('Volatility')
        return {
            'price': close.iloc[-1],
            'min': close.min(),
            'max': close.max(),
            'mean': close.mean(),
            'volatility': vol.iloc[-1] if vol is not None else None,
            'std': close.std(),
            'return_pct': (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100,
        }
