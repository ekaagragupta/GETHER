# src/preprocessing.py

import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import List

logger = logging.getLogger(__name__)


class EmissionPreprocessor:
    """
    Complete preprocessing pipeline for emission dataset.
    """

    def __init__(self, scaler_path: str = "models/scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler()

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["timestamp"])

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("timestamp")
        df = df.fillna(method="ffill", limit=6)
        df = df.interpolate(method="linear")
        return df

    def remove_outliers(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[
                (df[col] >= mean - 3 * std) &
                (df[col] <= mean + 3 * std)
            ]
        return df

    def normalize(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df[columns] = self.scaler.fit_transform(df[columns])

        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        """

        df = self.remove_duplicates(df)
        df = self.handle_missing(df)

        numeric_cols = ["CO2", "NO2", "SO2", "temperature"]
        df = self.remove_outliers(df, numeric_cols)

        scale_cols = [
            "CO2", "NO2", "SO2",
            "temperature", "humidity", "wind_speed"
        ]
        df = self.normalize(df, scale_cols)

        return df