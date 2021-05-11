# made by mike

import numpy as np


class Neuron:
    def __init__(self, num_inputs, num_outputs, rate=0.1):
        self.weights = np.zeros((num_inputs, num_outputs))
        self.bias = np.zeros((1, num_outputs))
        self.rate = rate

    def version(self):
        return "ver 1.1"

    def train_one(self, x: np.ndarray, y) -> bool:
        pred = self.predict(x)
        if pred == y:
            return False

        x: np.ndarray = np.reshape(x, self.weights.shape)
        normalize_value = np.sum(np.abs(x))
        x = x / normalize_value
        if y == 1:
            self.weights += x
        else:
            self.weights -= x
        return True

    def train(self, X: np.ndarray, Y):
        # pred = x @ self.weights
        # pred = self.predict(x)
        if(len(X.shape) == 1):
            X = X.reshape(1, -1)
        # print(X)
        _size = X.shape[0]
        X = X.T

        if(len(Y.shape) == 1):
            Y = Y.reshape(1, -1)
        _size = Y.shape[0]
        Y = Y.T

        pred = self._noutput(X)
        err = (Y - pred)**2

        gradient = 2*(Y - pred)
        binc = np.ones((_size, 1))

        self.weights += self.rate * (gradient @ X.T).T / _size
        self.bias += self.rate * (gradient @ binc) / _size
        return err

    def train_all(self, x, y):
        train_err = 0
        for i in range(x.shape[0]):
            train_err += self.train(x[i], y[i])
        return train_err
    # def lsm_train(self, x: np.ndarray, y):
    #     pred = x @ self.weights
    #     err = (y - pred)**2
    #     gradient = 2*(y - pred)
    #     print(-x)
    #     print(gradient)
    #     return err

    def copy(self):
        return self

    def _noutput(self, X: np.ndarray):
        return self.weights.T @ X + self.bias

    def predict(self, X: np.ndarray, log=False):
        if(len(X.shape) == 1):
            X = X.reshape(1, -1)
        # print(X)
        X = X.T

        pred = self._noutput(X)

        if(log):
            print(pred)
        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        return pred.T
