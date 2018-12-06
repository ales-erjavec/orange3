import unittest

import numpy as np

from Orange.projection import correspondence as corres

class TestCA(unittest.TestCase):
    def test_arpack_solver(self):
        A = np.arange(15).reshape(3, 5)
        A = A / A.sum()
        rsum, csum = A.sum(axis=1), A.sum(axis=0)
        wu, wv = rsum ** -0.5, csum ** -0.5
        B = np.diag(wu) @ (A - np.c_[rsum] * np.r_[csum]) @ np.diag(wv)
        U_, s_, Vh_ = np.linalg.svd(B, full_matrices=True)

        U, s, Vh = corres.ca_sparse(A)

        1 + 1
