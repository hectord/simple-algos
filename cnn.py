
from scipy import ndimage
from typing import List

import numpy as np
from load import load_training, load_validation



class CNNStep:

    def forward(self, image):
        raise NotImplementedError


class Convolution(CNNStep):

    def __init__(self, patch):
        self.patch = patch
        assert len(self.patch) == 2

    def forward(self, image):
        assert len(image.shape) == 2

        ret = ndimage.convolve(image, self.patch, mode='constant', cval=0.0)

        return ret


class MaxPooling(CNNStep):

    def __init__(self, size=3):
        self._size = size

    def forward(self, image):

        M, N = image.shape

        S1 = (self._size - M % self._size) % self._size
        S2 = (self._size - N % self._size) % self._size

        image = np.vstack([image, np.zeros((S1, N))])
        image = np.hstack([image, np.zeros((M + S1, S2))])

        M += S1
        N += S2

        K = L = self._size
        MK = M // K
        NL = N // L

        assert MK*K == M
        assert NL*L == N

        maxed = image.reshape(MK, K, NL, L).max(axis=(1, 3))

        return maxed


class ReLULayer(CNNStep):

    def forward(self, image):
        # ReLu
        image = np.array(image, copy=True)
        image[image < 0] = 0
        return image


class Flatten(CNNStep):

    def forward(self, image):
        return image.reshape(-1)


class Dense(CNNStep):

    def __init__(self, neurcount=10, image_size=768):
        self.NEURCOUNT = neurcount
        self.W1 = np.random.random((image_size, self.NEURCOUNT))
        self.B1 = np.random.random((self.NEURCOUNT,))

    def forward(self, flat_image):
        # we need a flat image
        assert len(flat_image.shape) == 1
        assert self.W1.shape[0] == flat_image.size

        ret =  flat_image @ self.W1 + self.B1

        return ret


class NeuralNetwork:

    def __init__(self, layers: List[Convolution]):
        self._layers = layers

    def forward(self, image):

        for filt in self._layers:
            image = filt.forward(image)

        return image
