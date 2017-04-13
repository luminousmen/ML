# Gradient descent for Linear regression
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr06.html

from random import seed
from random import randrange
from csv import reader
from math import sqrt
import time


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_column_to_float(dataset, column):
    """Convert string column to float"""
    for row in dataset:
        row[column] = float(row[column].strip())


def train_test_split(dataset, split):
    """Split a dataset into a train and test set"""
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


def rmse_metric(actual, predicted):
    """Calculate root mean squared error"""
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


def evaluate_algorithm(train, test, algorithm):
    """Evaluate regression algorithm on training dataset"""
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted, b0, b1 = algorithm(train, test_set)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse, predicted, b0, b1


def mean(values):
    """Calculate the mean value of a list of numbers"""
    return sum(values) / float(len(values))


def covariance(x, mean_x, y, mean_y):
    """Calculate covariance between x and y"""
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


def variance(values, mean):
    """Calculate the variance of a list of numbers"""
    return sum([(x - mean)**2 for x in values])


def coefficients(dataset):
    """Calculate coefficients"""
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]


def simple_linear_regression(train, test):
    """Simple linear regression algorithm"""
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions, b0, b1


if __name__ == "__main__":
    # algorithm is not determined cause we have some random
    seed(time.time())
    # load and prepare data
    filename = 'insurence.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    # split coefficient in parts
    # split = 0.8 means 80% of dataset is training data and 10% is tested
    split = 0.9
    # split dataset into test and train
    train, test = train_test_split(dataset, split)
    rmse, predicted, b0, b1 = evaluate_algorithm(train, test, simple_linear_regression)

    print('Predicted coef: %s, %s' % (b0, b1))
    print('Y\'s: %s' % predicted)
    print('RMSE: %.3f' % (rmse))
