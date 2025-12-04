# Stock Price Predictor Module
import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# Big ticker list
ALL_TICKERS = [
    # Large Caps / Tech
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AVGO','ASML','AMD','INTC','QCOM','MU','TXN','AMAT','LRCX','IBM','ORCL','SAP','CRM',
    # Fin/Payments
    'JPM','BAC','C','GS','MS','V','MA','AXP','PYPL','SQ',
    # Consumer / Services
    'DIS','NFLX','NKE','SBUX','MCD','WMT','TGT','COST','KO','PEP','ABNB','UBER','LYFT','SHOP',
    # Energy / Industrials / Health
    'XOM','CVX','BP','SHEL','UNH','PFE','JNJ','MRK','CAT','GE',
    # China / Taiwan / Others (ADRs/common)
    'BABA','BIDU','TSM','TCEHY','NIO',
    # ETFs
    'SPY','QQQ','DIA','IWM'
]

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def pinball_loss(tau=0.6):
    """Asymmetric pinball loss function"""
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(tau*e, (tau-1)*e))
    return loss

class StockPricePredictor:
    """LSTM-based stock price predictor with multi-horizon forecasting"""
    
    def __init__(self, symbol='AAPL', lookback=60, horizons=(1,30,90,180)):
        self.symbol = symbol
        self.lookback = lookback
        self.horizons = tuple(sorted(horizons))
        self.scaler = StandardScaler()
        self.model = None
        self.data = None
        self.rets_scaled = None
        self.logp = None
        self.val_bias = None

    def fetch_data(self, period='3y', interval='1d'):
        """Fetch stock data from Yahoo Finance"""
        df = yf.Ticker(self.symbol).history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
        df = df.dropna().copy()
        df['LogP'] = np.log(df['Close'])
        df['Ret'] = df['LogP'].diff()
        df = df.dropna()
        self.data = df
        return df

    def _build_supervised(self, arr):
        """Build supervised learning dataset from time series"""
        X, Y = [], []
        H = max(self.horizons)
        for i in range(self.lookback, len(arr) - H + 1):
            past = arr[i-self.lookback:i]
            fut = [np.sum(arr[i:i+h]) for h in self.horizons]  
            X.append(past)
            Y.append(fut)
        X = np.array(X)[..., np.newaxis]
        Y = np.array(Y)
        return X, Y

    def preprocess(self, train_ratio=0.8):
        """Preprocess data and create train/test splits"""
        self.logp = self.data['LogP'].values.astype('float32')
        rets = self.data['Ret'].values.astype('float32').reshape(-1,1)

        n = len(rets)
        n_train = int(n * train_ratio)

       
        self.scaler.fit(rets[:n_train])
        rets_z = self.scaler.transform(rets).flatten()
        self.rets_scaled = rets_z

        X, Y = self._build_supervised(rets_z)
        n_samples = X.shape[0]
        n_train_samples = int(n_samples * train_ratio)
        return X[:n_train_samples], Y[:n_train_samples], X[n_train_samples:], Y[n_train_samples:]

    def build(self, units=64, dropout=0.2, lr=1e-3, tau=0.6):
        """Build LSTM model architecture"""
        Hlen = len(self.horizons)
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(self.lookback,1),
                 kernel_regularizer=regularizers.l2(0.01)),
            Dropout(dropout),
             LSTM(units, kernel_regularizer=regularizers.l2(0.01)),
            Dropout(dropout),
            Dense(64, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
            Dense(Hlen, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss=pinball_loss(tau), metrics=['mae'])
        self.model = model
        return model

    def train(self, X_train, Y_train, epochs=30, batch_size=32, val_split=0.1, verbose=1):
        """Train the LSTM model"""
        cbs = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0),
        ]
        hist = self.model.fit(
            X_train, Y_train,
            validation_split=val_split,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=verbose,
            callbacks=cbs
        )
       # bias correction
        k = max(1, int(0.1 * len(X_train)))
        Xb, Yb = X_train[-k:], Y_train[-k:]
        pred_b = self.model.predict(Xb, verbose=0)
        self.val_bias = np.mean(Yb - pred_b, axis=0)
        return hist

    def evaluate(self, X_test, Y_test):
        """Evaluate model on test set"""
        Y_hat = self.model.predict(X_test, verbose=0)
        if self.val_bias is not None:
            Y_hat = Y_hat + self.val_bias

        m, s = self.scaler.mean_[0], np.sqrt(self.scaler.var_[0])
        horizons_arr = np.array(self.horizons)
        Y_hat_real = Y_hat * s + horizons_arr * m
        Y_test_real = Y_test * s + horizons_arr * m

        mse = mean_squared_error(Y_test_real, Y_hat_real)
        mae = mean_absolute_error(Y_test_real, Y_hat_real)
        mape = mean_absolute_percentage_error(Y_test_real, Y_hat_real)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'mape': mape * 100
        }

        return Y_hat_real, Y_test_real, metrics

    def multi_horizon_prices(self):
        """Predict future prices for multiple horizons"""
        seq = self.rets_scaled[-self.lookback:].copy()
        x = seq.reshape(1, self.lookback, 1)
        cum_z = self.model.predict(x, verbose=0)[0]
        if self.val_bias is not None:
            cum_z += self.val_bias

        m, s = self.scaler.mean_[0], np.sqrt(self.scaler.var_[0])
        horizons_arr = np.array(self.horizons)
        cum_logret = cum_z * s + horizons_arr * m

        last_price = float(np.exp(self.logp[-1]))
        out = {}
        for h, r in zip(self.horizons, cum_logret):
            out[h] = last_price * float(np.exp(r))
        return out

    def get_historical_prices(self):
        """Get historical price data for visualization"""
        if self.data is None:
            return None
        return self.data[['Close']].copy()
