import pandas as pd
import numpy as np

from src.feature_engineering import create_sequences
from src.models.baseline_models import run_baseline_experiments
from src.models.lstm_model import chronological_split

data = pd.read_csv("data/processed/emissions_clean.csv")


if "timestamp" in data.columns:
    data = data.drop(columns=["timestamp"])


data_values = data.values

X, y = create_sequences(data_values, sequence_length=30)


X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

results = run_baseline_experiments(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

print("\nBaseline Results:")
print(results)