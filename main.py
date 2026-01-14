import argparse
import logging
from src.data_fetcher.fetcher import DataFetcher
from src.analysis.analyzer import StockAnalyzer
from src.prediction.predictor import RandomForestPredictor
from src.scheduler.scheduler import DataScheduler
from src.utils.helpers import setup_logging, load_config, ensure_directories, save_predictions


class StockExchangeAnalyzer:
    def __init__(self):
        ensure_directories()
        self.log = setup_logging()
        self.config = load_config()
        self.scheduler = DataScheduler()
        self.log.info("Analyzer ready")

    def fetch_and_analyze(self, symbol=None):
        symbols = [symbol] if symbol else self.config['symbols']
        
        try:
            fetcher = DataFetcher(symbols)
            data = fetcher.fetch_historical_data()
            
            if isinstance(data, pd.DataFrame):
                for sym in symbols:
                    if len(symbols) == 1:
                        sym_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                    else:
                        sym_data = data.xs(sym, level=1, axis=1)
                    
                    analyzer = StockAnalyzer(sym_data)
                    analyzer.calculate_moving_averages()
                    analyzer.calculate_rsi()
                    analyzer.calculate_macd()
                    analyzer.calculate_volatility()
                    
                    stats = analyzer.get_stats()
                    self.log.info(f"{sym}: {stats}")
            
            return data
        except Exception as e:
            self.log.error(f"fetch failed: {e}")
            raise

    def predict(self, symbol, days_ahead=None):
        days_ahead = days_ahead or self.config['prediction_horizon']
        
        try:
            fetcher = DataFetcher([symbol])
            data = fetcher.fetch_historical_data()
            
            # TODO: refactor this extraction logic into a helper
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    close_df = data[['Close']]
                elif hasattr(data.columns, 'levels'):
                    close_df = data['Close'].to_frame()
                else:
                    close_df = data.iloc[:, :4] if len(data.columns) > 3 else data
            else:
                close_df = data
            
            predictor = RandomForestPredictor()
            predictor.train(close_df)
            
            last_prices = close_df['Close'].tail(predictor.lookback).values
            preds = predictor.predict(last_prices, days_ahead)
            
            fname = save_predictions(symbol, preds.tolist(), days_ahead)
            self.log.info(f"saved {fname}")
            
            return preds
        except Exception as e:
            self.log.error(f"prediction failed: {e}")
            raise

    def start_updates(self):
        self.scheduler.add_job(self.fetch_and_analyze, self.config['fetch_interval'], "recurring_fetch")
        self.scheduler.add_job(self.predict, self.config['retrain_interval'], "model_retrain")
        self.scheduler.start()
        self.log.info("updates started")

    def stop_updates(self):
        self.scheduler.stop()
        self.log.info("updates stopped")


if __name__ == "__main__":
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Stock Exchange Analyzer")
    parser.add_argument("--gui", action="store_true", help="Launch web dashboard")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    
    if args.gui:
        from src.gui.dashboard import run_dashboard
        run_dashboard()
    else:
        app = StockExchangeAnalyzer()
        
        print("Fetching data...")
        app.fetch_and_analyze()
        
        print("Running predictions...")
        for sym in app.config['symbols']:
            try:
                app.predict(sym, days_ahead=30)
                print(f"✓ {sym} done")
            except Exception as e:
                print(f"✗ {sym}: {e}")
