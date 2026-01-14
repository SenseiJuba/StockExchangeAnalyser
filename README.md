# Stock Exchange Analyzer

A comprehensive Python application for analyzing stock market data and predicting future price movements using machine learning.

## Features

- **Recurring Data Fetching**: Automatically fetch stock data at configurable intervals using Yahoo Finance
- **Technical Analysis**: Calculate moving averages, RSI, MACD, and volatility indicators
- **Machine Learning Predictions**: Multiple prediction models (Linear Regression, Random Forest)
- **Scheduled Tasks**: Background scheduler for continuous data updates and model retraining
- **Extensible Architecture**: Modular design for easy addition of new data sources and prediction models
- **Comprehensive Logging**: Track all operations and errors

## Project Structure

```
StockExchangeAnalyser/
├── src/
│   ├── data_fetcher/      # Data fetching from Yahoo Finance
│   ├── analysis/          # Technical analysis indicators
│   ├── prediction/        # ML prediction models
│   ├── scheduler/         # Recurring task scheduling
│   ├── utils/             # Helper functions and utilities
│   └── __init__.py
├── data/                  # Fetched data storage
├── models/                # Trained model storage
├── tests/                 # Unit tests
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup configuration
├── .env.example         # Environment variables template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Steps

1. **Clone or download the project**
   ```bash
   cd StockExchangeAnalyser
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```

   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your desired settings:
   - `SYMBOLS`: Comma-separated stock symbols (e.g., AAPL,MSFT,GOOGL)
   - `FETCH_INTERVAL`: Data fetch interval in seconds (default: 3600 = 1 hour)
   - `PREDICTION_HORIZON`: Number of days to predict ahead (default: 30)
   - `MODEL_TYPE`: Prediction model type (lstm, arima, random_forest)
   - `RETRAIN_INTERVAL`: Model retraining interval in seconds (default: 604800 = 1 week)

## Usage

### Basic Usage

```python
from main import StockExchangeAnalyzer

# Initialize analyzer
analyzer = StockExchangeAnalyzer()

# Fetch and analyze data for configured symbols
analyzer.fetch_and_analyze()

# Generate 30-day price predictions
predictions = analyzer.predict_future("AAPL", days_ahead=30)

# Print predictions
print(predictions)
```

### Running the Main Application

```bash
python main.py
```

### Starting Recurring Updates

```python
from main import StockExchangeAnalyzer

analyzer = StockExchangeAnalyzer()

# Start scheduled data fetching and model retraining
analyzer.start_recurring_updates()

# Keep the program running
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    analyzer.stop_recurring_updates()
    print("Analyzer stopped")
```

## Core Modules

### Data Fetcher (`src/data_fetcher/fetcher.py`)
- **DataFetcher**: Handles fetching historical and real-time stock data from Yahoo Finance
  - `fetch_historical_data()`: Retrieve OHLCV data for configured date range
  - `fetch_latest_data(symbol)`: Get the latest daily data
  - `get_stock_info(symbol)`: Get stock metadata

### Analysis (`src/analysis/analyzer.py`)
- **StockAnalyzer**: Performs technical analysis on stock data
  - `calculate_moving_averages(periods)`: MA 20, 50, 200
  - `calculate_rsi(period)`: Relative Strength Index (default 14 periods)
  - `calculate_macd()`: MACD and signal line
  - `calculate_volatility(period)`: Historical volatility
  - `get_summary_statistics()`: Aggregated metrics

### Prediction (`src/prediction/predictor.py`)
- **StockPredictor**: Base class for prediction models
- **LinearRegressionPredictor**: Simple linear regression approach
- **RandomForestPredictor**: Ensemble learning with multiple decision trees

### Scheduler (`src/scheduler/scheduler.py`)
- **DataScheduler**: Manages background tasks
  - `add_fetch_job()`: Schedule data fetching
  - `add_retrain_job()`: Schedule model retraining
  - `add_cron_job()`: Schedule specific time-based tasks

### Utilities (`src/utils/helpers.py`)
- Logging configuration
- Environment variable loading
- Directory management
- Prediction save/load functionality

## Configuration

### Environment Variables (`.env`)

```ini
# Stock symbols to track
SYMBOLS=AAPL,MSFT,GOOGL,TSLA

# Fetch data every N seconds (3600 = 1 hour)
FETCH_INTERVAL=3600

# Number of days to predict
PREDICTION_HORIZON=30

# ML model type: lstm, arima, or random_forest
MODEL_TYPE=random_forest

# Retrain model every N seconds (604800 = 1 week)
RETRAIN_INTERVAL=604800

# Database settings
DB_TYPE=sqlite
DB_PATH=data/stocks.db

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO
LOG_FILE=logs/analyzer.log
```

## Data Storage

- **Raw Data**: Stored in `data/` directory as CSV or pickle files
- **Models**: Trained ML models saved in `models/` directory
- **Logs**: Application logs in `logs/analyzer.log`

## Requirements

Key dependencies:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `yfinance`: Yahoo Finance data fetching
- `APScheduler`: Task scheduling
- `python-dotenv`: Environment variable management
- `tensorflow/keras`: Deep learning (optional, for LSTM models)

## Next Steps

### Enhancements to Consider

1. **Database Integration**: Add persistent storage (PostgreSQL, MongoDB)
2. **Advanced Models**: Implement LSTM and ARIMA models
3. **Web Dashboard**: Create a web interface using Flask/Django
4. **API Integration**: Add support for multiple data providers (Alpha Vantage, IEX Cloud)
5. **Backtesting**: Implement strategy backtesting framework
6. **Alert System**: Email/SMS notifications for price movements
7. **Risk Analysis**: Portfolio risk metrics and optimization
8. **Unit Tests**: Comprehensive test suite

## Troubleshooting

### ModuleNotFoundError
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

### No data returned from Yahoo Finance
- Verify stock symbols are valid (AAPL, MSFT, etc.)
- Check internet connection
- Yahoo Finance may have temporary outages

### Scheduler not running
- Ensure APScheduler is properly installed
- Check that `start_recurring_updates()` is called
- Verify the main script doesn't exit immediately

## Contributing

Feel free to extend this project with:
- New prediction models
- Additional technical indicators
- Support for cryptocurrency or other markets
- Performance optimizations
- Bug fixes and improvements

## License

MIT License - Feel free to use and modify for your purposes.

## Support

For issues or questions, refer to:
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [APScheduler Documentation](https://apscheduler.readthedocs.io/)
