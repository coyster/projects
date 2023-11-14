import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense


def load_boston_housing():
    save_path = os.path.join(
        pathlib.Path(__file__).parent, "data", "boston_housing.npz"
    )
    if not os.path.exists(save_path):
        from keras.datasets import boston_housing

        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        np.savez(
            save_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )
    else:
        data = np.load(save_path)
        print(data)
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
    return x_train, y_train, x_test, y_test


def transform_preprocessing(x_train, x_test) -> (np.array, np.array):
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def build_model():
    model = Sequential()
    model.add(Dense(100, input_dim=13, activation="relu"))
    model.add(Dense(100, input_dim=13, activation="relu"))
    model.add(Dense(50, input_dim=13, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])
    print(model.summary())
    return model


def train_model(model, x_train, y_train, x_test=None, y_test=None):
    model.fit(x_train, y_train, epochs=200, validation_split=0.2)
    
    return model


def plot_history_training(history):
    plt.plot(range(1, len(history.history["loss"]) + 1), history.history["loss"])
    plt.plot(
        range(1, len(history.history["val_loss"]) + 1), history.history["val_loss"]
    )
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["Train Loss", "Validation Loss"], loc="upper left")
    plt.show()


def evaluate_model(model, x_train, y_train, x_test, y_test):
    score = model.evaluate(x_train, y_train)
    print("Train MAE: %.2f" % score[1])
    score = model.evaluate(x_test, y_test)
    print("Test MAE: %.2f" % score[1])
    print(f"test score: {score}")


def predict_model(model, x, y):
    y_pred = model.predict(x)
    r2_score_ = r2_score(y, y_pred)
    print(f"r2_score: {r2_score_}")
    return r2_score_


######################################
######################################


x_train, y_train, x_test, y_test = load_boston_housing()
x_train, x_test = transform_preprocessing(x_train, x_test)

model = build_model()
model = train_model(model, x_train, y_train)

plot_history_training(model.history)
evaluate_model(model, x_train, y_train, x_test, y_test)
predict_model(model, x_test, y_test)
