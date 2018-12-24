from scipy.sparse.linalg import LinearOperator, svds

import unittest

import numpy as np
import numpy.testing as npt
import scipy.sparse as sp

from Orange.data import Table
from Orange.projection import correspondence as corres
from Orange.projection.correspondence import correspondence, cross_tabulate

assert_almost_equal = npt.assert_almost_equal


def reflect(v1, v2):
    """
    Reflect v2 so it matches the sign of `v1` in the first element.
    """
    s1 = np.sign(v1.flat[0])
    s2 = np.sign(v2.flat[0])
    return (v1, -v2) if s1 * s2 == -1 else (v1, v2)


def assert_almost_equal_to_reflection(actual, desired, *args, **kwargs):
    actual, desired = reflect(desired, actual)
    npt.assert_almost_equal(actual, desired, *args, **kwargs)


class TestCA(unittest.TestCase):
    def _test_scale_center_eq(self, A):
        A = A / A.sum()
        s = A.sum(axis=0)
        r = A.sum(axis=1)
        w, v = np.sqrt(1./r,), np.sqrt(1./s)
        U_, s_, Vh_ = corres.scale_center_svd(A, r, s, w, v)
        U, s, Vh = corres.scale_center_svd_arpack(A, r, s, w, v)
        k = np.sum(s_ > 1e-9)
        npt.assert_allclose(s_[:k], s[:k], rtol=0, atol=1e-6)
        npt.assert_allclose(
            np.abs(U_[:, :k]), np.abs(U[:, :k]), rtol=0, atol=1e-6
        )
        npt.assert_allclose(
            np.abs(Vh_[:k]), np.abs(Vh[:k]), rtol=0, atol=1e-6
        )

    def test_scale_center_svd(self):
        sizes = [(2, 2), (3, 5), (4, 5), (10, 20)]
        rs = np.random.RandomState(0)
        for size in sizes:
            A = rs.uniform(size=size)
            self._test_scale_center_eq(A)
            self._test_scale_center_eq(A.T)

    def test_smoke(self):
        # Test on smoke dataset against R package ca.
        smoke = np.array([
            # none light medium heavy
            [    4,    2,     3,     2],  # SM
            [    4,    3,     7,     4],  # JM
            [   25,   10,    12,     4],  # SE
            [   18,   24,    33,    13],  # JE
            [   10,    6,     7,     2],  # SC
        ])
        # > c = ca(smoke)
        # > coo <- plot(ca)
        # > round(coo$rows, 3)
        expected_rows = np.array([
            #  Dim1    Dim2
            [-0.066, -0.194],  # SM
            [ 0.259, -0.243],  # JM
            [-0.381, -0.011],  # SE
            [ 0.233,  0.058],  # JE
            [-0.201,  0.079],  # SC
        ])

        # > round(roo$cols, 3)
        expected_cols = np.array([
            #  Dim1    Dim2
            [-0.393, -0.030],  # none
            [ 0.099,  0.141],  # light
            [ 0.196,  0.007],  # medium
            [ 0.294, -0.198],  # heavy
        ])

        smoke = np.array(smoke, dtype=float)
        ca = correspondence(smoke)
        dims = [0, 1]
        x = np.round(ca.row_principal_coordinates[:, dims], 3)
        y = np.round(ca.col_principal_coordinates[:, dims], 3)
        assert_almost_equal_to_reflection(x[:, 0], expected_rows[:, 0])
        assert_almost_equal_to_reflection(x[:, 1], expected_rows[:, 1])
        assert_almost_equal_to_reflection(y[:, 0], expected_cols[:, 0])
        assert_almost_equal_to_reflection(y[:, 1], expected_cols[:, 1])
        # > round(c$rowcoord, 3)[,1:2]
        expected_rows = np.array([
            #  Dim1    Dim2
            [-0.241, -1.936],  # SM
            [ 0.947, -2.431],  # JM
            [-1.392, -0.107],  # SE
            [ 0.852,  0.577],  # JE
            [-0.735,  0.788],  # SC
        ])
        # > round(c$colcoord, 3)[,1:2]
        expected_cols = np.array([
            #  Dim1    Dim2
            [-1.438, -0.305],  # none
            [ 0.364,  1.409],  # light
            [ 0.718,  0.074],  # medium
            [ 1.074, -1.976],  # heavy
        ])
        x = np.round(ca.row_standard_coordinates[:5, dims], 3)
        y = np.round(ca.col_standard_coordinates[:4, dims], 3)
        assert_almost_equal_to_reflection(x[:, 0], expected_rows[:, 0])
        assert_almost_equal_to_reflection(x[:, 1], expected_rows[:, 1])
        assert_almost_equal_to_reflection(y[:, 0], expected_cols[:, 0])
        assert_almost_equal_to_reflection(y[:, 1], expected_cols[:, 1])
        # > round(c$rowinertia, 3)
        expected = np.array([0.003, 0.012, 0.038, 0.026, 0.006,])

        inertia = np.round(ca.row_inertia, 3)
        assert_almost_equal(inertia, expected)
        # > round(c$colinertia, 3)
        expected = np.array([0.049, 0.007, 0.013, 0.016])
        inertia = np.round(ca.col_inertia, 3)
        assert_almost_equal(inertia, expected)

        assert_almost_equal(ca.inertia_of_axis, ca.svd.s ** 2)

    def test_punctuation(self):
        # Author punctuation
        # (from 'Correspondence Analysis - Herve Abdi Lynne J. Williams')
        data = np.array([
            #period, comma, other
            [  7836,  13112,  6026],  # "Rousseau"
            [ 53655, 102383, 42413],  # "Chateaubriand"
            [115615, 184541, 59226],  # "Hugo"
            [161926, 340479, 62754],  # "Zola"
            [ 38177, 105101, 12670],  # "Proust"
            [ 46371,  58367, 14299],  # "Giraudoux"
        ])
        # Fig 12.
        expected = np.array([
            [ 0.2398,  0.0741],
            [ 0.1895,  0.1071],
            [ 0.1033, -0.0297],
            [-0.0918,  0.0017],
            [-0.2243,  0.0631],
            [ 0.0475, -0.1963],
        ])
        ca = correspondence(data)
        assert_almost_equal(ca.svd.s[:2] ** 2, [0.0178, 0.0056], decimal=3)
        assert_almost_equal_to_reflection(
            ca.rpc[:, 0], expected[:, 0], decimal=4
        )
        assert_almost_equal_to_reflection(
            ca.rpc[:, 1], expected[:, 1], decimal=4
        )


    def test_cross_tabulate(self):
        table = Table("zoo")
        hair = table.domain["hair"]
        type_ = table.domain["type"]
        xtab = cross_tabulate(
            table, table.domain.variables, [table.domain.class_var])
        # assert_array_equal(xtab, [0, 1])
        xtab = cross_tabulate(table, [], [])
        self.assertEqual(xtab.shape, (0, 0))

        xtab = cross_tabulate(table, [hair], [])
        self.assertEqual(xtab.size, 0)
        self.assertEqual(xtab.shape, (len(hair.values), 0))
        xtab = cross_tabulate(table, [], [hair])
        self.assertEqual(xtab.shape, (0, len(hair.values)))
        xtab = cross_tabulate(table, [type_], [hair])
        self.assertEqual(xtab.shape, (len(type_.values), len(hair.values)))
        self.assertEqual(xtab.sum(), len(table))
        self.assertSequenceEqual(
            xtab.sum(axis=1).tolist(), [4, 20, 13, 8, 10, 41, 5])
