from numpy import ndarray
import numpy as np
import math

from numpy.lib.function_base import gradient


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        if x > 0:
            return 1
        return 0


def dSigmoid(x):
    return (1 - sigmoid(x))*sigmoid(x)


sigmoid_v = np.vectorize(sigmoid)
dsigmoid_v = np.vectorize(dSigmoid)


class NeuralNetwork:
    def __init__(self, layers: ndarray, learning_rate=0.2, momentum=0.5):
        """
        Layers : at index 0, expects the number of inputs ( excluding bias )
        other-layers : are the hidden layers and the number of neurons.
        Last-layer : number of outputs.

        Network implementing learning rate that is adjusting if the error is low!
        so we can put a high learning rate and when the net almost converged it will adjust it self to small jumps.

        """
        self.num_layers = len(layers) - 1
        self.layers = layers.copy()
        self.weights: dict(ndarray) = {}
        self.biases: dict(ndarray) = {}
        self.output = {}
        self.output_in = {}
        self.learning_rate = learning_rate
        self.trained = 1000
        self.err1000 = 1000
        self.alpha = 1
        self.momentum = momentum

        self.gradient = {}
        self.gradient_bias = {}
        for i in range(self.num_layers):
            self.gradient[i] = 0
            self.gradient_bias[i] = 0
            self.weights[i] = np.random.rand(layers[i], layers[i+1])
            self.biases[i] = np.random.rand(layers[i+1], 1)

    def copy(self):
        c = NeuralNetwork(self.layers, self.learning_rate, self.momentum)
        for i in range(self.num_layers):
            c.weights[i] = self.weights[i].copy()
            c.biases[i] = self.biases[i].copy()
        return c

    def predict(self, X: ndarray):
        """
        Given x ( row ) predict output

        Y(k) = sum i to n : w(i,k) * X(i)
        => Y = W.T @ X.T <=> (X @ W).T   

        X(i) = Y_out^-1(i) <= output of prev layer
        Y_out(k) = f(Y(k))
        """
        if(len(X.shape) == 1):
            X = X.reshape(1, -1)
        # print(X)
        X = X.T
        if X.shape[0] != self.layers[0]:
            print("wrong input size!!", self.layers[0], X.shape[1])
            return
        # pred = 0
        inputs = X.shape[1]
        binc = np.ones((1, inputs))
        self.output[0] = X  # the output of layer 0 ( input )
        # self.output_in[0] = X
        for i in range(self.num_layers):
            # Compute Y
            self.output_in[i + 1] = self.weights[i].T @ self.output[i] + \
                self.biases[i] @ binc
            # Compute Y_out
            self.output[i + 1] = sigmoid_v(self.output_in[i + 1])
        return self.output[self.num_layers].T

    def train(self, X: ndarray, Y: ndarray):
        """
        Given some input X, and disired Y, train the network
        using Back Propogration
        """
        v = self.predict(X)
        if(v is None):
            print("Error in input X")
            return

        prediction = self.output[self.num_layers]

        if(len(Y.shape) == 1):
            Y = Y.reshape(1, -1)
        _size = Y.shape[0]
        Y = Y.T

        error = np.square(Y - prediction).sum()
        self.trained += _size

        p = _size / 1000
        self.err1000 = error + self.err1000 * (1 - p)
        self.alpha = 1
        if self.err1000 < 33:
            self.alpha = (self.err1000 / 100) ** 0.2 + 0.2
        self.alpha /= math.sqrt(_size)

        delta = {}
        delta[self.num_layers] = (Y - prediction) * \
            dsigmoid_v(self.output_in[self.num_layers])

        for i in range(self.num_layers - 1, 0, -1):
            # self.weights[i - 1] += self.learning_rate * delta[i + 1]
            delta[i] = self.weights[i] @ delta[i + 1]
            delta[i] = delta[i] * dsigmoid_v(self.output_in[i])

        binc = np.ones((1, _size))
        for i in range(self.num_layers):
            oldg = self.gradient[i]
            oldgb = self.gradient_bias[i]

            # calculate new gradient
            self.gradient[i] = (self.output[i] @ delta[i + 1].T)
            self.gradient_bias[i] = (delta[i + 1] @ binc.T)

            self.gradient[i] *= self.alpha * self.learning_rate
            self.gradient_bias[i] *= self.alpha * self.learning_rate

            # add old gradient as "momentum"
            self.gradient[i] += self.momentum * oldg
            self.gradient_bias[i] += self.momentum * oldgb

            # print(delta[i + 1], binc, gradient_bias)
            self.weights[i] += self.gradient[i]
            self.biases[i] += self.gradient_bias[i]

        return error / _size  # Error
