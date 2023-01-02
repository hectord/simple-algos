# -*- coding: utf-8 -*-

import math
from typing import Optional

import numpy as np
import argparse

from load import load_training, load_validation, read_images, read_label

SIZE_INPUT = 784
SIZE_LAYER_1 = 9
SIZE_LAYER_2 = 10


def init_params_linear():
    # 784 input => SIZE_LAYER_1 layers => SIZE_LAYER_2 layers => solution
    W1 = np.random.rand(SIZE_LAYER_1, SIZE_INPUT) - 0.5
    B1 = np.random.rand(SIZE_LAYER_1, 1) - 0.5

    W2 = np.random.rand(SIZE_LAYER_2, SIZE_LAYER_1) - 0.5
    B2 = np.random.rand(SIZE_LAYER_2, 1) - 0.5

    return W1, B1, W2, B2

def init_params_normal():
    W1 = np.random.randn(SIZE_LAYER_1, SIZE_INPUT)
    B1 = np.random.randn(SIZE_LAYER_1, 1)
    W2 = np.random.randn(SIZE_LAYER_2, SIZE_LAYER_1)
    B2 = np.random.randn(SIZE_LAYER_2, 1)

    return W1, B1, W2, B2

def forward_prop(W1, B1, W2, B2, X):

    Z1 = (W1 @ X  + B1)
    A1 = Z1.copy()
    A1[A1 < 0.0] = 0.0
    A1 = A1.astype(np.float128)

    Z2 = W2 @ A1 + B2
    Z2 = Z2.astype(np.float128)

    A2 = np.exp(Z2) / np.sum(np.exp(Z2), 0)

    return Z1, A1, Z2, A2

def deriv_ReLu(Z):
    return Z > 0

def one_hot(Y, n):
    one_hot_Y = np.zeros((Y.size, n))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


def backprop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size

    one_hot_Y = one_hot(Y, W2.shape[0])
    # dZ2 = 10xn
    dZ2 = A2 - one_hot_Y

    # db2 = 10x1
    db2 = 1 / m * np.sum(dZ2, axis=1).reshape(-1, 1)
    # dW2 = 10x10
    dW2 = 1 / m * (dZ2 @ A1.T)

    dZ1 = (W2.T @ dZ2) * deriv_ReLu(Z1)

    db1 = 1 / m * np.sum(dZ1, axis=1).reshape(-1, 1)
    dW1 = 1 / m * (dZ1 @ X.T)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    b2 = b2 - alpha * db2
    W2 = W2 - alpha * dW2

    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha, X_validation, Y_validation):
    # W1 = 10x684
    # W2 = 10x10
    # b1 = 10x1
    # b2 = 10x1
    W1, b1, W2, b2 = init_params_normal()

    for i in range(iterations):
        # Z1 = A1 = 10xn
        # Z2 = A2 = 10xn
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        dW1, db1, dW2, db2 = backprop(Z1, A1, Z2, A2, W2, X, Y)

        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 50 == 0:
            print(f'Iteration {i}')
            print('Accuracy[work]: ', get_accuracy(get_predictions(A2), Y))

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_validation)
            print('Accuracy[test]:', get_accuracy(get_predictions(A2), Y_validation))

    return W1, b1, W2, b2



def simple_perceptron(iterations: int, learning_rate: float, load_filter: Optional[int]):

    X_training, Y_training = load_training(load_filter)
    X_validation, Y_validation = load_validation(load_filter)

    W1, b1, W2, b2 = gradient_descent(X_training, Y_training, iterations, learning_rate, X_validation, Y_validation)

    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_validation)
    print('Accuracy[test]:', get_accuracy(get_predictions(A2), Y_validation))


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--learning-rate', default=0.1, nargs='?', type=float)
    args.add_argument('--load-filter', default=None, nargs='?', type=int)
    args.add_argument('--iterations', default=10000, nargs='?', type=int)

    args = args.parse_args()

    learning_rate = vars(args)['learning_rate']
    load_filter = vars(args)['load_filter']
    iterations = vars(args)['iterations']

    simple_perceptron(iterations, learning_rate, load_filter)
