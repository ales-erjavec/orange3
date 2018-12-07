import unittest

import numpy as np
import numpy.testing as npt
import scipy.sparse as sp

from Orange.projection import correspondence as corres


class TestCA(unittest.TestCase):
    def test_arpack_solver(self):
        A = np.arange(15).reshape(3, 5)
        A = A / A.sum()
        rsum, csum = A.sum(axis=1), A.sum(axis=0)
        wu, wv = rsum ** -0.5, csum ** -0.5
        B = np.diag(wu) @ (A - np.c_[rsum] * np.r_[csum]) @ np.diag(wv)
        U_, s_, Vh_ = np.linalg.svd(B, full_matrices=False)
        U, s, Vh = corres.ca_sparse(sp.csc_matrix(A))
        k = np.sum(s_ > 1e-9)
        npt.assert_allclose(U_[:, :k], U[:, :k], rtol=0, atol=1e-6)
        npt.assert_allclose(s_[:k], s[:k], rtol=0, atol=1e-6)
        npt.assert_allclose(Vh_.T[:, :k], Vh.T[:, :k], rtol=0, atol=1e-6)

