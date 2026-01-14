"""
Test suite for Stock Exchange Analyzer
"""
import unittest
import pandas as pd
import numpy as np
from src.data_fetcher.fetcher import DataFetcher
from src.analysis.analyzer import StockAnalyzer
from src.prediction.predictor import LinearRegressionPredictor, RandomForestPredictor


class TestDataFetcher(unittest.TestCase):
    """Test data fetching functionality"""

    def setUp(self):
        self.symbols = ['AAPL']
        self.fetcher = DataFetcher(self.symbols)

    def test_initialization(self):
        """Test DataFetcher initialization"""
        self.assertEqual(self.fetcher.symbols, self.symbols)
        self.assertIsNotNone(self.fetcher.start_date)
        self.assertIsNotNone(self.fetcher.end_date)


class TestStockAnalyzer(unittest.TestCase):
    """Test stock analysis functionality"""

    def setUp(self):
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100)
        self.data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Open': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        self.analyzer = StockAnalyzer(self.data)

    def test_moving_averages(self):
        """Test moving average calculation"""
        self.analyzer.calculate_moving_averages([20])
        self.assertIn('MA_20', self.analyzer.data.columns)

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        self.analyzer.calculate_rsi()
        self.assertIn('RSI', self.analyzer.data.columns)
        
        # Check RSI is between 0 and 100
        rsi = self.analyzer.data['RSI'].dropna()
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())

    def test_volatility_calculation(self):
        """Test volatility calculation"""
        self.analyzer.calculate_volatility()
        self.assertIn('Volatility', self.analyzer.data.columns)

    def test_summary_statistics(self):
        """Test summary statistics generation"""
        stats = self.analyzer.get_summary_statistics()
        
        required_keys = ['current_price', 'min_price', 'max_price', 'mean_price', 'std_dev', 'total_return']
        for key in required_keys:
            self.assertIn(key, stats)


class TestPredictors(unittest.TestCase):
    """Test prediction models"""

    def setUp(self):
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100)
        self.data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
        }, index=dates)

    def test_linear_regression_predictor(self):
        """Test Linear Regression predictor"""
        predictor = LinearRegressionPredictor(lookback_window=20)
        predictor.train(self.data)
        
        last_prices = self.data['Close'].tail(20).values
        predictions = predictor.predict(last_prices, days_ahead=10)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(np.isfinite(predictions)))

    def test_random_forest_predictor(self):
        """Test Random Forest predictor"""
        predictor = RandomForestPredictor(lookback_window=20)
        predictor.train(self.data)
        
        last_prices = self.data['Close'].tail(20).values
        predictions = predictor.predict(last_prices, days_ahead=10)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(np.isfinite(predictions)))


if __name__ == '__main__':
    unittest.main()
