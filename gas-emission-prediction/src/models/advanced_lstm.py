import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def build_advanced_lstm(input_shape):

    model = Sequential()

    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(64))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model