import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from keras.datasets import mnist


import os
import pathlib


def download_data(path):
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)

    print(x_train.shape, y_train.shape)


def load_data(path):
    data = np.load(path)
    (x_train, y_train), (x_test, y_test) = (data["x_train"], data["y_train"]), (
        data["x_test"],
        data["y_test"],
    )
    # (x_train, y_train), (x_test, y_test) = np.load(path)

    # print(x_train.shape, y_train.shape)
    return (x_train, y_train), (x_test, y_test)


def explore_balance(x_train, y_train, x_test, y_test):
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Train Labels: {dict(zip(unique, counts))}")


def one_hot_encode(y, num_classes=10):
    from keras.utils import to_categorical

    y = to_categorical(y, num_classes=num_classes)
    print(y[:4])
    return y



path = os.path.join(pathlib.Path(os.path.dirname(__file__)).parent, "data/mnist.npz")
(x_train, y_train), (x_test, y_test) = load_data(path)

# explore_balance(x_train, y_train, x_test, y_test)
y_test = one_hot_encode(y_test, num_classes=10)
y_train = one_hot_encode(y_train, num_classes=10)












