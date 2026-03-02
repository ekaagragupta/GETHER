

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Utility Metrics Function


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


#linear regression

def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> LinearRegression:

    model = LinearRegression()
    model.fit(X_train, y_train)

    logger.info("Linear Regression trained.")
    return model


#random forest regression

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> RandomForestRegressor:

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    logger.info("Random Forest trained.")
    return model

#BASIC LSTM (LAYER 1    )

def build_basic_lstm(input_shape: Tuple[int, int]) -> tf.keras.Model:

    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    logger.info("Basic LSTM built.")
    return model