import os
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

LOG_FILE = "logs/analyzer.log"
DATA_DIR = "data"
REQUIRED_DIRS = ['data', 'models', 'logs']


def setup_logging(log_file=LOG_FILE, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def load_config():
    load_dotenv()
    return {
        'symbols': os.getenv('SYMBOLS', 'AAPL,MSFT').split(','),
        'fetch_interval': int(os.getenv('FETCH_INTERVAL', 3600)),
        'prediction_horizon': int(os.getenv('PREDICTION_HORIZON', 30)),
        'model_type': os.getenv('MODEL_TYPE', 'random_forest'),
        'retrain_interval': int(os.getenv('RETRAIN_INTERVAL', 604800)),
        'db_path': os.getenv('DB_PATH', 'data/stocks.db'),
    }


def ensure_directories():
    for d in REQUIRED_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_predictions(symbol, predictions, horizon):
    ensure_directories()
    fname = f"{DATA_DIR}/{symbol}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'horizon_days': horizon,
            'predictions': predictions
        }, f, indent=2)
    return fname


def load_predictions(fname):
    with open(fname) as f:
        return json.load(f)
