import numpy as np
import matplotlib.pyplot as plt

from NeuralNet import NeuralNetwork


def create_data(_f, _size, _maxn, _maxm):
    """
    return random sample of size : _size
    """
    fdata = []
    for _ in range(_size):
        n = (int)(np.random.random() * 2 * _maxn) - _maxn
        m = (int)(np.random.random() * 2 * _maxm) - _maxm

        x = m / _maxm
        y = n / _maxn
        fdata.append([x, y, _f(x, y)])
    fdata = np.array(fdata)
    return fdata


def generate_data_all(_f, _maxn, _maxm):
    """
    return all cases for the given fucntion,
    seperate into 2 sets negative and positive.

    @return data_negative, data_positive
    """
    data_positive = []
    data_negative = []
    for i in range(-_maxm, _maxm):
        x = i / _maxm
        for j in range(-_maxn, _maxn):
            y = j / _maxn
            v = _f(x, y)
            if v > 0:
                data_positive.append([x, y, v])
            else:
                data_negative.append([x, y, v])
    return np.array(data_negative), np.array(data_positive)


def generate_data_all_noseperation(_f, _maxn, _maxm):
    """
    return all cases for the given fucntion,
    seperate into 2 sets negative and positive.

    @return data_negative, data_positive
    """
    data = []
    for i in range(-_maxm, _maxm):
        x = i / _maxm
        for j in range(-_maxn, _maxn):
            y = j / _maxn
            v = _f(x, y)
            data.append([x, y, v])
    return np.array(data)


def generate_uniform_dataset(_f, _size, data_positive=None, data_negative=None):
    if data_positive is None or data_negative is None:
        data_negative, data_positive = generate_data_all(_f, 100, 100)
    nsize = data_negative.shape[0]
    psize = data_positive.shape[0]
    tsize = int(_size / 2)
    random_positive = np.random.choice(psize, tsize)
    random_negative = np.random.choice(nsize, tsize)

    return np.concatenate((data_negative[random_negative], data_positive[random_positive]), axis=0)


def plot_data(X, y):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, :1], X[:, 1:2], c=y, s=30, cmap='Accent')


def calculate_accuracity(net: NeuralNetwork, X, Y):
    predictions = net.predict(X)
    diff = np.abs(Y - predictions)
    positive = (diff < 0.5).sum()
    acc = positive / X.shape[0]
    return acc


def plot_diff_inner(predictions, X, Y, title, ax=None, s=20):
    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    plt.subplot(ax)

    diff = np.abs(Y - predictions)
    positive = diff < 0.5
    negative = diff >= 0.5

    neg_g = plt.scatter(X[:, :1][negative], X[:, 1:]
                        [negative], c='brown', s=s, alpha=0.8, cmap='Accent')
    pos_g = plt.scatter(X[:, :1][positive], X[:, 1:]
                        [positive], c='green', s=s, alpha=0.8, cmap='Accent')

    ax.title.set_text(title)
    plt.legend((neg_g, pos_g), ("Wrongly classified",
                                "Correctly classified"), loc='lower left')

    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    return ax


def plot_diff(net: NeuralNetwork, X, Y, title, ax=None, s=20):
    predictions = net.predict(X)
    return plot_diff_inner(predictions, X, Y, title, ax, s)


def plot_test_inner(predictions, X, title, ax=None, s=20):
    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    plt.subplot(ax)

    positive = predictions > 0.5
    negative = predictions <= 0.5

    neg_g = plt.scatter(X[:, :1][negative], X[:, 1:]
                        [negative], c='gray', s=s, alpha=0.8, cmap='Accent')
    pos_g = plt.scatter(X[:, :1][positive], X[:, 1:]
                        [positive], c='green', s=s, alpha=0.8, cmap='Accent')

    ax.title.set_text(title)
    plt.legend((neg_g, pos_g), ("Classified as 0",
                                "Classified as 1"), loc='lower left')

    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    return ax


def plot_test(net: NeuralNetwork, X, title, ax=None, s=20):
    predictions = net.predict(X)
    return plot_test_inner(predictions, X, title, ax, s)


# def run_just_test(_n, test_data):
#     pos = test_data[:, :-1]
#     pvalue = test_data[:, -1:]

#     _predicted = predict_result(_n, pos)
#     _diff: np.array = abs(pvalue.flatten() - _predicted)
#     return [pos, pvalue, _diff, _n]
