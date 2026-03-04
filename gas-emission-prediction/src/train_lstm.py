import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping

from models.advanced_lstm import build_advanced_lstm
from feature_engineering import create_sequences


# load processed dataset
df = pd.read_csv("data/processed/emissions_clean.csv")

# target
target_column = "AQI"

X = df.drop(columns=[target_column]).values
y = df[target_column].values


# convert to sequences
sequence_length = 30

data = np.column_stack((y, X))

X_seq, y_seq = create_sequences(data, sequence_length)


# train / validation / test split

X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq, test_size=0.3, shuffle=False
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=False
)


print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)


# build model
model = build_advanced_lstm(
    input_shape=(X_train.shape[1], X_train.shape[2])
)


# early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    shuffle=False
)


# evaluate
test_loss, test_mae = model.evaluate(X_test, y_test)

print("\nTest Loss:", test_loss)
print("Test MAE:", test_mae)

model.save("models/aqi_lstm_model.h5")
print("Model saved.")