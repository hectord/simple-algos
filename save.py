# -*- coding: utf-8 -*-

import numpy as np
from functools import reduce

def read(filename: str):

    with open(filename, 'r') as f:
        lines = list(f.readlines())

        dimensions = map(int, lines[0].strip().split())

        values = map(float, lines[1].strip().split())

        return np.array(list(values)).reshape(*dimensions)


def write(array, filename):

    with open(filename, 'w') as f:
        base = array.shape
        f.write(' '.join(map(str, base)))
        f.write('\n')

        values = array.reshape(-1)

        total = reduce(lambda x, y: x * y, base)

        for i in range(total):
            f.write('%s' % values[i])

            if i != total - 1:
                f.write(' ')

if __name__ == '__main__':
    array = [
        [1.0, 2.0, 3.0],
        [9, 9, 9]
    ]
    write(np.array(array), 'my.txt')

    print(read('my.txt'))
