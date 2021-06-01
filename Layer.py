import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def delta_sigmoid(sigmoid):
    return sigmoid * (1 - sigmoid)


class Layer:
    def __init__(self, input_size, next_layer, activation=True):
        self.input_size = input_size
        self.next_layer = next_layer
        self.activation = activation
        self.weights = np.random.normal(0, 1, (self.input_size, self.next_layer.input_size))
        self.delta_weights = np.zeros_like(self.weights)

    def compute(self, input_values, train=False):
        assert (len(input_values) == self.input_size)
        input_values = np.nan_to_num(input_values)
        activation_res = np.dot(input_values, self.weights)
        if self.activation:
            activation_res = sigmoid(activation_res)
        values, error = self.next_layer.compute(activation_res, train)
        if train:
            error = self.compute_error(input_values, error, activation_res)
        return values, error

    def compute_error(self, input_values, error, activation_res):
        delta_weights = np.dot(
            input_values.reshape(self.input_size, 1),
            error.reshape(1, self.next_layer.input_size))
        if self.activation:
            delta_weights *= delta_sigmoid(activation_res)
        self.delta_weights += delta_weights
        return np.dot(self.weights, error)

    def update_weights(self, multiplier):
        self.weights -= self.delta_weights * multiplier
        self.delta_weights.fill(0.)
        self.next_layer.update_weights(multiplier)
