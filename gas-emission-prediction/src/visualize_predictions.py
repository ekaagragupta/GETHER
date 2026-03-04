import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from feature_engineering import create_sequences


# Load processed dataset
df = pd.read_csv("data/processed/emissions_clean.csv")

# Target column
target_column = "AQI"

X = df.drop(columns=[target_column]).values
y = df[target_column].values

# Convert to sequences
sequence_length = 30
data = np.column_stack((y, X))

X_seq, y_seq = create_sequences(data, sequence_length)

# Train / validation / test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq, test_size=0.3, shuffle=False
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=False
)

# Load trained model
model = load_model("models/aqi_lstm_model.h5")

# Make predictions
y_pred = model.predict(X_test)

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual AQI", marker='o', markersize=3)
plt.plot(y_pred, label="Predicted AQI", marker='s', markersize=3)

plt.legend()
plt.title("AQI Prediction: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("AQI Value")
plt.grid(True, alpha=0.3)
plt.show()