import numpy as np


class TestLayer:

    def __init__(self, expected_data):
        self.expected_data = expected_data
        self.mse = np.zeros_like(expected_data)
        self.input_size = len(expected_data[0])

    def compute(self, input_values, train=False):
        error = np.nan_to_num(input_values - self.expected_data)
        self.mse += error ** 2
        if train:
            error = 2 * error
        return input_values, error

    def update_weights(self, multiplier):
        pass
