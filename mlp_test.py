# -*- coding: utf-8 -*-

import unittest
import numpy as np
import numpy.testing as npt
from math import e

from mlp import forward_prop, backprop



class TestPerceptron(unittest.TestCase):

    def setUp(self):
        # a col is an input (...)
        self.X = np.array([[0.0, 0.0],
                           [2.0, 1.0],
                           [0.1, 1.0]])

        self.W1 = np.array([[1.0, 0.5, 0.0],
                            [0.5, 0.5, 1.0]])
        self.B1 = np.array([1.0, -1.0])

        # Z1 = W @ X + B
        self.Z1 = np.array([[2.0, -0.5],
                            [2.1,  0.5]])
        # A1 = (Z1[Z1 < 0.0] = 0.0)
        self.A1 = np.array([[2.0, 0.0],
                            [2.1, 0.5]])

        self.W2 = np.array([[2.0, 1.0],
                            [0.0, 1.0]])
        self.B2 = np.array([1.0, 1.0])

        # Z2 = W2 @ A1 + B2
        self.Z2 = np.array([[7.1, 1.5],
                            [3.1, 1.5]])

        # A2 = exp(Z2) / sum
        sum1 = np.exp(7.1) + np.exp(3.1)
        sum2 = np.exp(1.5) + np.exp(1.5)
        self.A2 = np.array([[np.exp(7.1) / sum1, np.exp(1.5) / sum2],
                            [np.exp(3.1) / sum1, np.exp(1.5) / sum2]])

        # expected
        self.Y = np.array([[1, 0]])

        self.A2_expected = np.array([[0.0, 1.0],
                                     [1.0, 0.0]])
        self.dZ2 = self.A2 - self.A2_expected
        self.db2 = 0.5 * np.sum(self.dZ2, axis=1).reshape(2,1)

        self.dW2 = 0.5 * self.dZ2 @ self.A1.T

    def test_forward_prop(self):

        o_Z1, o_A1, o_Z2, o_A2 = forward_prop(self.W1, self.B1,
                                              self.W2, self.B2,
                                              self.X)

        npt.assert_array_almost_equal(self.Z1, o_Z1)
        npt.assert_array_almost_equal(self.A1, o_A1)
        npt.assert_array_almost_equal(self.Z2, o_Z2)
        npt.assert_array_almost_equal(self.A2, o_A2)

    def test_backward_prop(self):

        dW1, db1, dW2, db2 = backprop(self.Z1, self.A1,
                                      self.Z2, self.A2,
                                      self.W2, self.X, self.Y)

        npt.assert_array_almost_equal(self.db2, db2)
        npt.assert_array_almost_equal(self.dW2, dW2)


if __name__ == '__main__':
    unittest.main()
