# -*- coding: utf-8 -*-

import numpy as np
from display import plot
from save import read

if __name__ == '__main__':
    serie1 = []
    serie2 = []

    for step in range(0, 10000, 50):
        accuracty_test = read(f'output/{step}/accuracty_test.txt')
        serie1.append((step, accuracty_test[0]))

        accuracy_work = read(f'output/{step}/accuracty_work.txt')
        serie2.append((step, accuracy_work[0]))


    plot(serie1, serie2)
