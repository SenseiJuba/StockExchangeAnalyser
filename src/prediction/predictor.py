import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import logging

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 30
N_ESTIMATORS = 100


class StockPredictor:
    def __init__(self, lookback=DEFAULT_LOOKBACK):
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def _prepare(self, data):
        prices = data['Close'].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(prices)
        X, y = [], []
        for i in range(len(scaled) - self.lookback):
            X.append(scaled[i:i + self.lookback])
            y.append(scaled[i + self.lookback])
        return np.array(X), np.array(y)


# Classic predictors

class LinearRegressionPredictor(StockPredictor):
    def __init__(self, lookback=DEFAULT_LOOKBACK):
        super().__init__(lookback)
        self.model = LinearRegression()
        self.residual_std = None

    def train(self, data):
        X, y = self._prepare(data)
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y.ravel())
        
        # compute residual std for prediction intervals
        y_pred = self.model.predict(X_flat)
        self.residual_std = np.std(y.ravel() - y_pred)
        log.debug(f"LR trained, residual_std={self.residual_std:.4f}")

    def predict(self, last_prices, days_ahead=30):
        preds = []
        seq = self.scaler.transform(last_prices.reshape(-1, 1))
        for _ in range(days_ahead):
            p = self.model.predict(seq.reshape(1, -1))
            preds.append(p[0])
            seq = np.append(seq[1:], p)
        return self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    def predict_with_ci(self, last_prices, days_ahead=30, ci=0.95):
        """Predict with confidence intervals based on residual error"""
        preds = []
        seq = self.scaler.transform(last_prices.reshape(-1, 1))
        z = 1.96 if ci == 0.95 else 1.645
        
        # uncertainty grows with forecast horizon
        for i in range(days_ahead):
            p = self.model.predict(seq.reshape(1, -1))[0]
            preds.append(p)
            seq = np.append(seq[1:], [[p]], axis=0)
        
        preds = np.array(preds)
        
        # scale uncertainty by sqrt of steps ahead (error accumulation)
        steps = np.arange(1, days_ahead + 1)
        uncertainty = self.residual_std * z * np.sqrt(steps)
        
        lower = preds - uncertainty
        upper = preds + uncertainty
        
        # inverse transform
        mean_prices = self.scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        lower_prices = self.scaler.inverse_transform(lower.reshape(-1, 1)).flatten()
        upper_prices = self.scaler.inverse_transform(upper.reshape(-1, 1)).flatten()
        
        return mean_prices, lower_prices, upper_prices


class RandomForestPredictor(StockPredictor):
    def __init__(self, lookback=DEFAULT_LOOKBACK, n_estimators=N_ESTIMATORS):
        super().__init__(lookback)
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    def train(self, data):
        X, y = self._prepare(data)
        self.model.fit(X.reshape(X.shape[0], -1), y.ravel())
        log.debug("RF trained")

    def predict(self, last_prices, days_ahead=30):
        preds = []
        seq = self.scaler.transform(last_prices.reshape(-1, 1))
        for _ in range(days_ahead):
            p = self.model.predict(seq.reshape(1, -1))
            preds.append(p[0])
            seq = np.append(seq[1:], p)
        return self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    def predict_with_ci(self, last_prices, days_ahead=30, ci=0.95):
        """Predict with confidence intervals from individual trees"""
        preds_mean = []
        preds_lower = []
        preds_upper = []
        
        seq = self.scaler.transform(last_prices.reshape(-1, 1))
        z = 1.96 if ci == 0.95 else 1.645  # 95% or 90% CI
        
        for _ in range(days_ahead):
            # get predictions from all trees
            tree_preds = np.array([t.predict(seq.reshape(1, -1))[0] for t in self.model.estimators_])
            
            mean_pred = tree_preds.mean()
            std_pred = tree_preds.std()
            
            preds_mean.append(mean_pred)
            preds_lower.append(mean_pred - z * std_pred)
            preds_upper.append(mean_pred + z * std_pred)
            
            seq = np.append(seq[1:], [[mean_pred]], axis=0)
        
        # inverse transform all arrays
        mean_prices = self.scaler.inverse_transform(np.array(preds_mean).reshape(-1, 1)).flatten()
        lower_prices = self.scaler.inverse_transform(np.array(preds_lower).reshape(-1, 1)).flatten()
        upper_prices = self.scaler.inverse_transform(np.array(preds_upper).reshape(-1, 1)).flatten()
        
        return mean_prices, lower_prices, upper_prices


# Model registry - add new models here
AVAILABLE_MODELS = {
    "rf": ("Random Forest", RandomForestPredictor),
    "lr": ("Linear Regression", LinearRegressionPredictor),
}


def get_all_models():
    """Return dict of model_id -> (display_name, predictor_class)"""
    return AVAILABLE_MODELS


def get_model(model_id):
    """Instantiate a model by its ID"""
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_id}")
    return AVAILABLE_MODELS[model_id][1]()


def get_model_options():
    """Return list of options for dropdown"""
    return [(model_id, name) for model_id, (name, _) in AVAILABLE_MODELS.items()]

