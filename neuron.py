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

    def adelaine_train(self, x: np.ndarray, y):
        # pred = x @ self.weights
        # pred = self.predict(x)
        pred = self._noutput(x)
        err = (y - pred)**2
        x: np.ndarray = np.reshape(x, self.weights.shape)
        gradient = 2*(y - pred)
        # print(gradient)
        self.weights += self.rate * gradient * x
        self.bias += self.rate * gradient
        return err

    def train_all(self, x, y):
        train_err = 0
        for i in range(x.shape[0]):
            train_err += self.adelaine_train(x[i], y[i])
        return train_err
    # def lsm_train(self, x: np.ndarray, y):
    #     pred = x @ self.weights
    #     err = (y - pred)**2
    #     gradient = 2*(y - pred)
    #     print(-x)
    #     print(gradient)
    #     return err

    def _noutput(self, x: np.ndarray):
        return x @ self.weights + self.bias

    def predict(self, x: np.ndarray):
        pred = self._noutput(x)
        if pred > 0:
            return 1
        return -1
