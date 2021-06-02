import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Layer import Layer
from TestLayer import TestLayer
from threading import Thread
from datetime import datetime
import winsound


def test(network, test_input_data, test_output_data, last_layer, print_info=True):
    output_G_total = []
    output_kgf = []
    expected_G_total = []
    expected_kgf = []

    for i, input_data in enumerate(test_input_data):
        last_layer.expected_data = test_output_data[i]
        output, unused = network.compute(input_data, train=False)
        output_G_total.append(output[0])
        output_kgf.append(output[1])
        expected_G_total.append(test_output_data[i][0])
        expected_kgf.append(test_output_data[i][1])

    if print_info:
        x = np.arange(len(test_input_data))
        # fig, ax = plt.subplots()
        # ax.plot(x, output_G_total, label='result', marker='o')
        # ax.plot(x, expected_G_total, label='expected', marker='o')
        # plt.show()
        fig, ax = plt.subplots()
        ax.plot(x, output_kgf, label='result')
        ax.plot(x, expected_kgf, label='expected')
        ax.legend()
        plt.show()


def draw_mse(mse_list, test_mse_list):
    title = ['G total', 'kgf']
    for i in range(2):
        train = []
        test = []
        for j in range(len(mse_list)):
            train.append(mse_list[j][i])
            test.append(test_mse_list[j][i])
        x = np.arange(len(mse_list))
        fig, ax = plt.subplots()
        ax.plot(x, train, label=title[i] + ' train')
        ax.plot(x, test, label=title[i] + ' test')
        ax.legend()
        plt.show()


def train(input_data, output_data, test_input_data,
          test_output_data, epoch_num=5000, learning_rate=0.044,
          stochastic=False):
    batch_size = len(input_data)
    last_layer = TestLayer(output_data)
    last_hidden_layer = Layer(input_size=63, next_layer=last_layer, activation=False)
    network = Layer(input_size=len(input_data[0]),
                    next_layer=Layer(input_size=70,
                                     next_layer=last_hidden_layer))
    mse_list = []
    test_mse_list = []
    for epoch in range(epoch_num + 1):
        order = np.random.permutation(batch_size)
        last_layer.mse = np.zeros_like(output_data[0])
        for current_data_row in order:
            last_layer.expected_data = output_data[current_data_row]
            input_values = np.nan_to_num(input_data[current_data_row])
            network.compute(input_values, train=True)
            if stochastic:
                network.update_weights(learning_rate / batch_size)
        if not stochastic:
            network.update_weights(learning_rate / batch_size)
        if epoch % 100 == 0 and False:
            mse = last_layer.mse / batch_size
            last_layer.mse = np.zeros_like(output_data[0])
            test(network, test_input_data, test_output_data, last_layer,
                 # print_info=(epoch % 200 == 0)
                 False
                 )
            test_mse = last_layer.mse / len(test_input_data)
            print('epoch', epoch, 'training mse: ', mse, 'test mse: ', test_mse)
            if epoch > 10:
                mse_list.append(mse)
                test_mse_list.append(test_mse)
    last_layer.mse = np.zeros_like(output_data[0])
    test(network, test_input_data, test_output_data, last_layer)
    print(last_layer.mse / len(test_input_data))
    # test(network, input_data, output_data, last_layer)
    # draw_mse(mse_list, test_mse_list)


def normalize(data):
    for i in range(data.shape[1]):
        max = np.nanmax(data[:, i])
        min = np.nanmin(data[:, i])
        for j in range(data.shape[0]):
            data[j, i] = (data[j, i] - min) / (max - min)


def separate_data(data):
    num = data.shape[0]
    train_num = int(0.8 * num)
    np.random.shuffle(data)
    train_dataset = data[:train_num]
    test_dataset = data[train_num:]
    return test_dataset, train_dataset


class MyThread(Thread):
    def __init__(self, train_input, train_output, test_input, test_output):
        Thread.__init__(self)
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output

    def run(self) -> None:
        train(self.train_input, self.train_output, self.test_input, self.test_output)


def main():
    data = pd.read_csv('data/data.csv', sep=';', encoding='windows-1251').to_numpy()
    normalize(data)
    test_data, train_data = separate_data(data)
    train_input = train_data[:, :-2]
    train_output = train_data[:, -2:]
    test_input = test_data[:, :-2]
    test_output = test_data[:, -2:]
    # MyThread(train_input, train_output, test_input, test_output).start()
    # MyThread(train_input, train_output, test_input, test_output).start()
    # MyThread(train_input, train_output, test_input, test_output).start()
    print('start', )

    def print_time():
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time)

    print_time()
    for j in range(2):
        train(train_input, train_output, test_input, test_output)
        print_time()


if __name__ == '__main__':
    for i in range(8):
        main()
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
