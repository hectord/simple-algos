
from typing import Optional
import numpy as np
import itertools


def read_images(count_filter: Optional[int],
                filename='train-images-idx3-ubyte',
                binary=True):
    with open(filename, 'rb') as f:
        res = f.read(2)
        assert res == b'\x00\x00'

        f.read(1)

        dims_count = int.from_bytes(f.read(1), byteorder='big')
        assert dims_count == 3

        dims = []
        for dim in range(dims_count):
            dims.append(int.from_bytes(f.read(4), byteorder='big'))

        count = dims[0]

        for c in range(count):

            image = []
            for i in range(dims[1]):
                line = []
                for g in range(dims[2]):

                    value = int.from_bytes(f.read(1), byteorder='big')

                    if binary:
                        value = 1 if value != 0 else 0
                    else:
                        value = value / 256
                    line.append(value)
                image.append(line)

            yield image

            if c == count_filter:
                break


def read_label(count_filter: Optional[int], filename='train-labels-idx1-ubyte'):

    with open(filename, 'rb') as f:
        res = f.read(2)
        assert res == b'\x00\x00'

        f.read(1)

        dims_count = int.from_bytes(f.read(1), byteorder='big')
        assert dims_count == 1

        dims = []
        for dim in range(dims_count):
            dims.append(int.from_bytes(f.read(4), byteorder='big'))

        count = dims[0]

        for c in range(count):
            yield int.from_bytes(f.read(1), byteorder='big')

            if c == count_filter:
                break


def load_training(count_filter: Optional[int], binary=True):
    Y = np.array(list(read_label(count_filter, 't10k-labels-idx1-ubyte')))

    lines = []
    for a in read_images(count_filter, 't10k-images-idx3-ubyte', binary=binary):
        lines.append(list(itertools.chain(*a)))

    X = np.array(lines).T

    return X, Y


def load_validation(count_filter: Optional[int], binary=True):
    Y = np.array(list(read_label(count_filter, 'train-labels-idx1-ubyte')))

    lines = []
    for a in read_images(count_filter, 'train-images-idx3-ubyte'):
        lines.append(list(itertools.chain(*a)))

    X = np.array(lines).T

    return X, Y

