# -*- coding: utf-8 -*-

import numpy as np
from display import display
from load import load_training, load_validation, read_images, read_label
from save import read
from run import run

if __name__ == '__main__':
    i = 9950

    W1 = read(f'output/{i}/W1.txt')
    W2 = read(f'output/{i}/W2.txt')
    b1 = read(f'output/{i}/b1.txt')
    b2 = read(f'output/{i}/b2.txt')

    X_validation, Y_validation = load_validation(None)

    ret = run(X_validation, W1, W2, b1, b2)
    diff_true = ret != Y_validation

    sel_values = np.zeros(diff_true.size)
    sel_values[diff_true] = 1

    print(1.0 - np.sum(sel_values) / sel_values.size)
    print(read(f'output/{i}/accuracty_test.txt'))

    PICK = 20

    # how good is it? examples which failed
    indicies = np.arange(Y_validation.size)
    sel_values = indicies[diff_true]
    lst = list(sel_values[:PICK])

    solutions = list(Y_validation[diff_true][:PICK])
    ret = list(ret[diff_true][:PICK])

    legends = [
        f'{a} vs {b}' for (a, b) in zip(solutions, ret)
    ]

    display([X_validation[:, i].reshape(28, 28) for i in lst], legends=legends)

