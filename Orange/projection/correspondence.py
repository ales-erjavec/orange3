"""

"""

import numpy as np
import scipy.sparse as sp

from scipy.linalg import eigh as lapack_eigh, svd as lapack_svd
from scipy.sparse.linalg import eigsh as arpack_eigh, svds as arpack_svd
from scipy.sparse.linalg import LinearOperator


def wsvd(A, wu=None, wv=None, k=-1, svd_solver="auto", overwrite_a=False):
    """
    Compute a weighted SVD (also called generalized SVD)
    (https://en.wikipedia.org/wiki/Generalized_singular_value_decomposition)

    Factors the matrix `A` as  `A = U @ diag(s) @ Vh` where
    `U.T @ diag(wu) @ U = I` and `Vh @ diag(wv) @ Vh.T = I`

    The `wu` or `wv` are passed as 1d arrays in which case they are taken
    to be elements of a diagonal weighting matrix. If omitted they are
    assumed to be `I`

    Parameters
    ----------
    A : (M, N) array_like

    wu : (M) array_like
        The diagonal elements of `u` (row) weighting matrix.
        Must be non negative.
    wv : (N) array_like
        The diagonal elements of `v` (column) weighing matrix.
        Must be non negative.
    k : int
        Maximum number of returned singular values/vectors. If -1 then all the
        values are returned.

    Returns
    -------
    U : array
    s : array
    Vh : array
    """
    if svd_solver not in {"auto", "arpack", "lapack"}:
        raise ValueError("Invalid 'svd_solver': {}".format(svd_solver))

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A.ndim != 2; ({A.ndim})")
    m, n = A.shape

    if not A.size:
        raise ValueError("A.size == 0")
        # return np.empty((m, 0)), np.empty([],), np.empty((0, m))

    if wu is not None:
        wu = np.asarray(wu)
        if wu.shape != (m,):
            raise ValueError(f"wu.shape != (m,); {wu.shape} != {(m,)}")
    else:
        wu = 1.

    if wv is not None:
        wv = np.asarray(wv)
        if wv.shape != (n,):
            raise ValueError(f"wv.shape != (n,); {wv.shape} != {(n,)}")
    else:
        wv = 1

    wu_sqrt = np.sqrt(wu)
    wv_sqrt = np.sqrt(wv)
    if overwrite_a:
        A_hat = A
    else:
        A_hat = A.copy()
    del A
    # A_hat = diag(sqrt(wu)) @ A @ diag(sqrt(wv))
    A_hat *= wu.reshape((-1, 1))  # broadcast over cols
    A_hat *= wv.reshape((1, -1))  # broadcast over rows

    if svd_solver == "auto":
        if min(n, m) > 200 and 0 < k < 10:
            svd_solver = "arpack"
        else:
            svd_solver = "lapack"

    if svd_solver == "lapack":
        U_hat, s, Vh_hat = lapack_svd(A_hat, full_matrices=False, overwrite_a=True)
        if k > 0:
            U_hat = U_hat[:, :k]
    elif svd_solver == "arpack":
        if k < 0:
            k_ = min(A_hat.shape)
        else:
            k_ = k
        U_hat, s, Vh_hat = arpack_svd(A_hat, k=k_)
    else:
        raise ValueError("Invalid 'svd_solver': {}".format(svd_solver))
    del A_hat
    eps = np.finfo(A.dtype).eps
    assert U_hat.shape[0] == m
    assert Vh_hat.shape[0] == n
    # U = diag(wu_sqrt ** -1) @ U_hat
    wu_sqrt_inv = np.reciprocal(wu_sqrt, where=wu_sqrt >= eps)
    U = np.multiply(U_hat, wu_sqrt_inv.reshape((-1, 1)), out=U_hat)
    wv_sqrt_inv = np.reciprocal(wv_sqrt, where=wv_sqrt >= eps)
    # Vh = (diag(wv_sqrt ** -1) @ Vh_hat.T).T
    Vh = np.multiply(Vh_hat, wv_sqrt_inv.reshape((-1, 1)), out=Vh_hat)
    return U, s, Vh


class _LinOp(LinearOperator):
    """
    diag(w) @ (M - r@s.T) @ diag(v)

     (diag(w) @ M - diag(w) @ r @ s.T) @ diag(v)
    = diag(w) @ M @ diag(v) - wr @ vs.T

    L @ X
    = diag(w) @ M @ diag(v) @ X - wr @ vs.T @ X

    L.T @ X
    X.T @ L
    -------
     X.T  @ diag(w) @  M  @ diag(v) - X.T  @  wr  @  vs.T
    (kxm)   mxm       mxn    nxn      kxm    mx1    1xn
    """
    def __init__(self, M, r, s, w, v):
        assert M
        super().__init__(M.dtype, M.shape)
        self.M = M
        self.r = r
        self.s = s
        self.w = w
        self.v = v
        self.wr = w * r
        self.vs = v * s

    def _matmat(self, X):
        #: >>>
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, m = self.shape
        X = X.reshape(m, -1)

    def _rmatmul(self, X):
        """
        X @ _LinOp
        >>>
        """
        """
        >>> X.T @ diag(w) @ M
        """
    def _rmatvec(self, x):
        return self.rmatmat(x.reshape((-1, 1)))

    def _adjoint(self):
        #
        return LinearOperator(
            shape=self.shape[::-1],
            matmat=self.rmatvec,
            rmatvec=self.matvec,
        )


def ca_sparse(table, k=-1):
    # type: (sp.spmatrix, ...) -> CA
    """
    Solve the simple correspondence analysis problem on a sparse table
    """
    dtype = table.dtype
    total = table.sum()
    m, n = table.shape
    table = table / total  # type: sp.spmatrix

    row_mass = np.asarray(table.sum(axis=1)).ravel()
    col_mass = np.asarray(table.sum(axis=0)).ravel()

    def col_vec(x):
        return x.reshape((-1, 1))

    def row_vec(x):
        return x.reshape((1, -1))

    M = table
    w = np.reciprocal(row_mass)
    v = np.reciprocal(col_mass)
    r = row_mass
    s = col_mass
    wr = w * r
    vs = v * s
    assert w.shape == r.shape == (m,)
    assert v.shape == s.shape == (n,)

    def matmat(X):
        if X.ndim == 1:
            X = col_vec(X)
        _n, k = X.shape
        assert _n == n
        Y1 = X * col_vec(v)
        Y1 = M.dot(Y1); assert Y1.shape == (m, k)
        Y1 *= col_vec(w)
        #: Y1 = diag(w) @ M @ diag(v)
        Y2 = X.T @ col_vec(vs)
        assert Y2.shape == (k, 1)
        Y2 = col_vec(wr) * Y2
        assert Y2.shape == Y1.shape == (m, k)
        return Y2

    def rmatvec(X):
        """
        >>>
        """
        if X.ndim == 1:
            X = col_vec(X)
        m_, k = X.shape
        assert m_ == m
        Y1 = (X.T * row_vec(w))
        # : Y1 @ M -> (M.T @ Y1.T).T
        Y1 = M.T.dot(Y1.T).T
        assert Y1.shape == (k, n)
        Y1 *= row_vec(v)

        Y2 = X.T.dot(col_vec(wr))
        assert Y2.shape == (k, 1)
        Y2 = Y2.dot(row_vec(vs))
        assert Y2.shape == (k, n)
        return Y2
    if k == -1:
        k = min(table.shape)

    op = LinearOperator(table.shape, matmat, matmat=matmat, rmatvec=rmatvec)
    U, D, V = arpack_svd(op, k=min(k, min(table.shape)))
    return U, D, V


class
class CA:
    svd = ...  # type:
    sv = ...   # type:
    def __init__(self, ):
        ...



