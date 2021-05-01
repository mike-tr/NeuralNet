from numpy import ndarray
import numpy as np
import math


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
    def __init__(self, layers: ndarray, learning_rate=0.1):
        """
        Layers : at index 0, expects the number of inputs ( excluding bias )
        other-layers : are the hidden layers and the number of neurons.
        Last-layer : number of outputs.

        """
        self.num_layers = len(layers) - 1
        self.layers = layers.copy()
        self.weights: dict(ndarray) = {}
        self.biases: dict(ndarray) = {}
        self.output = {}
        self.output_in = {}
        self.learning_rate = learning_rate
        for i in range(self.num_layers):
            self.weights[i] = np.random.rand(layers[i], layers[i+1])
            self.biases[i] = np.random.rand(layers[i+1], 1)

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
        # print(Y)
        Y = Y.T

        delta = {}
        delta[self.num_layers] = (Y - prediction) * \
            dsigmoid_v(self.output_in[self.num_layers])

        for i in range(self.num_layers - 1, 0, -1):
            # self.weights[i - 1] += self.learning_rate * delta[i + 1]
            delta[i] = self.weights[i] @ delta[i + 1]
            delta[i] = delta[i] * dsigmoid_v(self.output_in[i])

        for i in range(self.num_layers):
            gradient = (self.output[i] @ delta[i + 1].T)
            self.weights[i] += self.learning_rate * gradient

        return np.square(Y - prediction).sum()  # Error
