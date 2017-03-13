""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    data = load_digits()
    num_trials = 50
    interval = 5
    train_percentages = range(5, 95, interval)  # range from 5 to 90
    test_accuracies = numpy.zeros(len(train_percentages))

    for percentage in train_percentages:
        accuracy = 0
        size = percentage / 100
        index = int(percentage / interval - 1)

        # repeates num_trials time to smooth out variability
        for i in range(num_trials):
            X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                                data.target,
                                                                train_size=size)
            model = LogisticRegression(C=10**-10)
            model.fit(X_train, y_train)
            accuracy += model.score(X_test, y_test)
        test_accuracies[index] = accuracy / num_trials  # average of accuracy

    # plots the results
    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    # display_digits()
    train_model()
