import numpy as np
import pandas as pd

from sklearn.utils import shuffle


def relu(x):
    return x * (x > 0)


def error_rate(p, t):
    return np.mean(p != t)


def getKaggleMNIST():
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    train = pd.read_csv(
        '../large_files/digit_train.csv').values.astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000, 1:] / 255
    # Xtrain = train[:-1000, 1:]
    # Xtrain = (Xtrain - Xtrain.mean())/Xtrain.std()
    Ytrain = train[:-1000, 0].astype(np.int32)

    Xtest = train[-1000:, 1:] / 255
    # Xtest = train[-1000:, 1:]
    # Xtest = (Xtest - Xtest.mean()) / Xtest.std()
    Ytest = train[-1000:, 0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest


def init_weights(shape):
    w = np.random.randn(*shape) / np.sqrt(sum(shape))
    return w.astype(np.float32)
