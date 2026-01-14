import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 365
DATE_FMT = '%Y-%m-%d'


class DataFetcher:
    PERIOD_MAP = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}

    def __init__(self, symbols, start_date=None, end_date=None):
        self.symbols = symbols
        self.end_date = end_date or datetime.now().strftime(DATE_FMT)
        self.start_date = start_date or (datetime.now() - timedelta(days=DEFAULT_LOOKBACK)).strftime(DATE_FMT)
        log.debug(f"fetcher init: {symbols}")

    def set_period(self, period):
        self.end_date = datetime.now().strftime(DATE_FMT)
        days = self.PERIOD_MAP.get(period, 180)
        self.start_date = (datetime.now() - timedelta(days=days)).strftime(DATE_FMT)
        
    def fetch_historical_data(self):
        try:
            data = yf.download(self.symbols, start=self.start_date, end=self.end_date, progress=False)
            log.info(f"fetched {len(self.symbols)} symbols")
            return data
        except Exception as e:
            log.error(f"fetch failed: {e}")
            raise

    def fetch_latest(self, symbol):
        try:
            return yf.Ticker(symbol).history(period="1d")
        except Exception as e:
            log.error(f"latest fetch failed for {symbol}: {e}")
            raise

    def get_info(self, symbol):
        try:
            return yf.Ticker(symbol).info
        except Exception as e:
            log.error(f"info fetch failed: {e}")
            return {}
