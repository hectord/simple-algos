
from cnn import Convolution, MaxPooling, ReLULayer, Flatten, Dense, NeuralNetwork
import numpy as np
import unittest


class TestCnn(unittest.TestCase):

    def setUp(self):

        self.filters = [Flatten(),
                        Dense()]
        self.network = NeuralNetwork(self.filters)

    def test_simple_convolution(self):
        patch = np.array([
            [0, 1],
            [0, 1],
        ])
        convolution = Convolution(patch)

        image = np.array([
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
        ])
        ret = convolution.forward(image)

        expected = np.array([
            [0, 2, 1],
            [0, 2, 2],
            [0, 1, 1],
        ])

        self.assertTrue(np.all(expected == ret))

    def test_max_pooling(self):
        maxpooling = MaxPooling(size=2)

        image = np.array([
            [4, 1, 0, 1],
            [2, 1, 2, 2],
            [1, 1, 1, 3],
            [1, 1, 1, 1],
        ])
        ret = maxpooling.forward(image)

        expected = np.array([
            [4, 2],
            [1, 3],
        ])

        self.assertTrue(np.all(expected == ret))

    def test_relu_layer(self):
        relu_layer = ReLULayer()

        image = np.array([
            [ 0, 1, -1],
            [-1, 1,  1],
            [ 0, 1,  1],
        ])
        ret = relu_layer.forward(image)

        expected = np.array([
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
        ])

        self.assertTrue(np.all(expected == ret))

    def test_flatten(self):

        flatten = Flatten()

        image = np.array([
            [ 0, 1],
            [-1, 1],
        ])
        ret = flatten.forward(image)

        expected = np.array([0, 1, -1, 1])

        self.assertTrue(np.all(expected == ret))

    def test_dense(self):
        dense = Dense(neurcount=3, image_size=4)

        dense.W1 = np.array([
            [3.0, 5.0, 1.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        dense.B1 = np.array([1.0, 2.0, 3.0])

        image = np.array([
            [4, 1],
            [2, 1],
        ])

        dense.forward(image.reshape(-1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
