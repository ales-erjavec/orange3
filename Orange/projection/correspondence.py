"""
Correspondence analysis
-----------------------

"""
from typing import List, Tuple, Union, Sequence

import Orange

import typing

import numpy as np
import scipy.sparse as sp

from scipy.linalg import svd as lapack_svd
from scipy.sparse.linalg import svds as arpack_svd
from scipy.sparse.linalg import LinearOperator

from Orange.data import DiscreteVariable, Variable, Table, Domain
from Orange.data.util import SharedComputeValue
import Orange.preprocess
from Orange.projection import Projector, Projection
from Orange.statistics import contingency
from Orange.util import Reprable_repr_pretty

Lapack, Arpack, Auto = 'lapack', 'arpack', 'auto'

ndarray = np.ndarray

if typing.TYPE_CHECKING:
    Matrix = typing.Union[ndarray, sp.spmatrix]


def col_vec(x):  # type: (ndarray) -> ndarray
    """Return x as a column vector."""
    return x.reshape((-1, 1))


def row_vec(x):  # type: (ndarray) -> ndarray
    """Return x as a row vector."""
    return x.reshape((1, -1))


if typing:
    @typing.overload
    def correspondence(table: np.ndarray, maxk=-1, solver=Auto) -> 'CA': ...
    @typing.overload
    def correspondence(table: sp.spmatrix, maxk=-1, solver=Auto) -> 'CA': ...


def correspondence(table, maxk=-1, solver=Auto, ):
    """
    Compute simple correspondence analysis

    Parameters
    ----------
    table: (N, M) matrix
        A (N x M) matrix of frequency data.
    maxk : int
        The maximum number of principal components to compute/retain in the
        output. If -1 then the maximum number of components are retained.
    solver : str
        One of 'lapack', 'arpack' or 'auto'. 'auto' chooses the one of the other
        based on the input type and size.
    """
    if isinstance(table, sp.spmatrix):
        table = table.tocsc(copy=False)  # type: sp.csc_matrix
        assert np.all(np.isfinite(table.data))
        assert np.all(table.data >= 0)
        M, N = table.shape
    else:
        table = np.asarray(table)
        assert np.all(table >= 0)
        assert np.all(np.isfinite(table))
        M, N = table.shape

    total = table.sum()
    if total > 0:
        table = table / total

    row_mass = np.asarray(table.sum(axis=1)).ravel()
    col_mass = np.asarray(table.sum(axis=0)).ravel()

    def reciprocal_flush(arr):
        # assert np.all(arr[mask] >= 0)
        eps = np.finfo(arr.dtype).eps
        mask = arr > eps
        out = np.reciprocal(arr, where=mask)
        out[~mask] = 0
        return out

    rw = reciprocal_flush(row_mass)
    w = np.sqrt(rw)
    cw = reciprocal_flush(col_mass)
    v = np.sqrt(cw)

    r = row_mass
    c = col_mass

    assert w.shape == r.shape == (M,)
    assert v.shape == c.shape == (N,)

    if solver == Auto:
        if isinstance(table, sp.spmatrix) or min(N, M) > 200:
            solver = Arpack
        else:
            solver = Lapack

    if isinstance(table, sp.spmatrix):
        if solver not in (Auto, Arpack):
            table = table.todense()

    if maxk < 0:
        maxk = min(table.shape)

    if solver == Arpack:
        maxk = min(maxk, min(table.shape) - 1)
    elif solver == Lapack:
        maxk = min(maxk, *table.shape)

    # solver diag(w) @ (M - r@s.T) @ diag(v) | solver == lapack | arpack
    if solver == Arpack:
        U, s, Vh = scale_center_svd_arpack(table, r, c, w, v, maxk=maxk)
    elif solver == Lapack:
        U, s, Vh = scale_center_svd(table, r, c, w, v, overwrite_a=True)
        U, s, Vh = U[:, :maxk], s[:maxk], Vh[:maxk]
    else:
        assert False
    assert (np.diff(s) <= 0).all()
    eps = np.finfo(s.dtype).eps
    smask = s > np.sqrt(eps)
    U, s, Vh = U[:, smask], s[smask], Vh[smask]
    rowcoord = col_vec(w) * U
    colcoord = col_vec(v) * Vh.T
    return CA(SVD(U, s, Vh), rowcoord, colcoord, row_mass, col_mass)


class SVD(typing.NamedTuple):
    #: The left singular vectors.
    U: np.ndarray
    #: The singular values in descending order.
    s: np.ndarray
    #: The right singular vectors.
    Vh: np.ndarray

    def _repr_pretty_(self, *args, **kwargs):
        return Reprable_repr_pretty(
            type(self).__name__,
            zip(self._fields, self),
            *args, **kwargs
        )


def scale_center_svd(A, r, s, w, v, overwrite_a=False):
    """
    Perform a SVD decomposition of a `diag(w) @ (A - r@s.T) @ diag(v)`
    centered and scaled matrix using *lapack*.

    .. note::
        The centered/weighted matrix is formed in memory.
    --
    Parameters
    ----------
    A : (N, M) matrix
    r : (N) array
    s : (M) array
    w : (N) array
    v : (M) array
    overwrite_a : bool

    Returns
    -------
    svd : SVD
    """
    assert A.shape == (r.size, s.size) == (w.size, v.size)
    if overwrite_a:
        A -= col_vec(r) @ row_vec(s)
    else:
        A = A - col_vec(r) @ row_vec(s)

    A *= col_vec(w)
    A *= row_vec(v)
    U, s, Vh = lapack_svd(A, full_matrices=False, overwrite_a=True)
    assert np.all(np.diff(s) <= 0)
    return SVD(U, s, Vh)


def scale_center_svd_arpack(A, r, s, w, v, maxk=-1):
    """
    Perform a SVD decomposition of a `diag(w) @ (A - r@s.T) @ diag(v)`
    centered and scaled matrix using *arpack*.

    .. note::
        The centered/weighted matrix is never formed in memory.

    Parameters
    ----------
    A : (N, M) matrix
    r : (N) array
    s : (M) array
    w : (N) array
    v : (M) array
    maxk : int
        Number of singular numbers/vectors to return

    Returns
    -------
    svd : SVD
    """
    op = ScaledCenteredLinearOperator(A, r, s, w, v)
    rstate = np.random.RandomState(0xf0042)
    v0 = rstate.uniform(-1, 1, size=min(op.shape))
    if maxk < 0:
        maxk = min(A.shape) - 1
    else:
        maxk = min(maxk, min(A.shape) -1)
    U, s, Vh = arpack_svd(op, k=maxk, v0=v0)
    p = np.argsort(s)[::-1]
    U, s, Vh = U[:, p], s[p], Vh[p]
    return SVD(U, s, Vh)


class ScaledCenteredLinearOperator(LinearOperator):
    """
    A linear operator `diag(w) @ (M - r@s.T) @ diag(v)` where `M` is a
    matrix, `r` and `s` are column vectors, and `w, `v` are the entries of
    a diagonal weighing matrices.

    In other words: center `M` with rank one matrix `r@s.T` and scale the
    resulting matrix with `w` row-wise and with `v` column-wise.

    Parameters
    ----------
    M : (M, N) matrix or a LinearOperator
    r : (M) np.ndarray
    s : (N) np.ndarray
    w : (M) np.ndarray
    v : (N) np.ndarray

    diag(w) @ (M - r@s.T) @ diag(v)
    = (diag(w) @ M - diag(w) @ r @ s.T) @ diag(v)
    = diag(w) @ M @ diag(v) - wr @ vs.T
    ; where wr is w*r and vs is v*s (multiplied element-wise)

    L @ X
    -----
    = diag(w) @  M  @ diag(v) @ X -  wr  @ vs.T @  X
       mxm      mxn    nxn     mxk  mx1    1xn    nxk

    L.T @ X
    X.T @ L
    -------
    = X.T  @ diag(w) @  M  @ diag(v) - X.T  @  wr  @  vs.T
      kxm    mxm       mxn    nxn      kxm    mx1    1xn
    """
    def __init__(self, M, r, c, w, v):
        super().__init__(M.dtype, M.shape)
        assert self.dtype.kind == 'f'
        self.M = M
        self.r = r
        self.c = c
        self.w = w
        self.v = v
        self.wr = w * r
        self.vc = v * c
        assert self.shape == (w.size, v.size) == (r.size, c.size)
        assert w.ndim == v.ndim == r.ndim == v.ndim == 1

    def _matmat(self, X):
        if X.ndim == 1:
            X = col_vec(X)
        m, n = self.shape
        n_, k = X.shape
        assert n_ == n
        Y1 = X * col_vec(self.v)
        Y1 = self.M.dot(Y1)
        assert Y1.shape == (m, k)
        Y1 *= col_vec(self.w)
        #: Y1 = diag(w) @ M @ diag(v)
        Y2 = row_vec(self.vc) @ X
        assert Y2.shape == (1, k)
        Y2 = col_vec(self.wr) @ Y2
        assert Y2.shape == Y1.shape == (m, k)
        return Y1 + Y2

    def _rmatmat(self, X):
        if X.ndim == 1:
            X = col_vec(X)
        m, n = self.shape
        m_, k = X.shape
        assert m_ == m
        Y1 = (X.T * row_vec(self.w))
        # : Y1 @ M -> (M.T @ Y1.T).T
        Y1 = self.M.T.dot(Y1.T).T
        assert Y1.shape == (k, n)
        Y1 *= row_vec(self.v)

        Y2 = X.T.dot(col_vec(self.wr))
        assert Y2.shape == (k, 1)
        Y2 = Y2.dot(row_vec(self.vc))
        assert Y2.shape == (k, n)
        return Y1 - Y2

    def _rmatvec(self, x):
        return self._rmatmat(x.reshape((-1, 1)))

    def _transpose(self):
        """
        (diag(w) @ (M - r@s.T) @ diag(v)).T
        = diag(v).T @ (M - r@s.T).T @ diag(w).T
        = diag(v) @ (M.T - s@r.T) @ diag(w)
        """
        return ScaledCenteredLinearOperator(
            self.M.T, self.c, self.r, self.v, self.w
        )


symmetric, rowprinc, colprinc = 'symetric', 'rowprincipal', 'colprincipal'


class CA:
    #: The solution to the (generalized) SVD problem.
    svd = ...       # type: SVD
    #: The (N, k) array of row standard coordinates
    rowcoord = ...  # type: ndarray
    #: The (M, k) array of column standard coordinates
    colcoord = ...  # type: ndarray
    #: An (N,) array of row point masses
    rowmass = ...   # type: ndarray
    #: An (N,) array of column point masses
    colmass = ...   # type: ndarray
    #: The number of principal dimensions
    k = ...         # type: int

    def __init__(self, svd, rowcoord, colcoord, rowmass, colmass):
        # type: (SVD, ndarray, ndarray, ndarray, ndarray) -> None
        assert rowcoord.ndim == colcoord.ndim == 2
        assert rowcoord.shape[1] == colcoord.shape[1] == svd.s.shape[0]
        assert rowmass.shape == (rowcoord.shape[0],)
        assert colmass.shape == (colcoord.shape[0],)
        k = svd.s.shape[0]
        self.svd = svd
        self.rowcoord = rowcoord
        self.colcoord = colcoord
        self.rowmass = rowmass
        self.colmass = colmass
        self.k = k

    @property
    def row_standard_coordinates(self):  # type:() -> ndarray
        return self.rowcoord

    @property
    def col_standard_coordinates(self):  # type:() -> ndarray
        return self.colcoord

    @property
    def row_principal_coordinates(self):  # type:() -> ndarray
        return self.rowcoord * self.svd.s
    #: Shorthand for `row_principal_coordinates`
    rpc = row_principal_coordinates  # type: ndarray
    row_factors = property(lambda s: s.rpc)

    @property
    def col_principal_coordinates(self):  # type: () -> ndarray
        return self.colcoord * self.svd.s
    #: Shorthand for `col_principal_coordinates`
    cpc = col_principal_coordinates  # type: ndarray
    col_factors = property(lambda s: s.cpc)

    @property
    def row_inertia(self):
        return (col_vec(self.rowmass) * (self.rpc ** 2)).sum(axis=1)

    @property
    def col_inertia(self):
        return (col_vec(self.colmass) * (self.cpc ** 2)).sum(axis=1)

    @property
    def row_inertia_contributions(self):
        """
        Row inertia contribution for each point per each dimension.
        """
        return col_vec(self.rowmass) * (self.rpc ** 2)

    @property
    def col_inertia_contributions(self):
        """
        Column inertia contribution for each point per each dimension.
        """
        return col_vec(self.colmass) * (self.cpc ** 2)

    @property
    def total_inertia(self):
        return np.sum(self.svd.s ** 2)

    @property
    def inertia(self):
        return self.total_inertia

    @property
    def inertia_e_dim(self):
        inertia = self.inertia_of_axis
        return inertia / self.total_inertia

    @property
    def inertia_of_axis(self):
        return self.svd.s ** 2

    @property
    def D(self):
        return self.svd.s

    def plot(self, dim=(0, 1), map=symmetric):
        from matplotlib import pyplot as plt
        from matplotlib import gridspec
        if map == symmetric:
            rowcoords = self.row_principal_coordinates[:, dim]
            colcoords = self.col_principal_coordinates[:, dim]
        elif map == rowprinc:
            rowcoords = self.row_principal_coordinates[:, dim]
            colcoords = self.row_standard_coordinates[:, dim]
        elif map == colprinc:
            rowcoords = self.row_standard_coordinates[:, dim]
            colcoords = self.col_principal_coordinates[:, dim]
        else:
            raise ValueError(map)

        row_labels = ...
        col_labels = ...

        fig = plt.figure()
        coords_conc = np.vstack((rowcoords, colcoords))
        bounds = np.ptp(coords_conc, axis=0)
        if len(dim) == 3:
            if bounds[0] > 1e-9:
                width_ratio = [1., bounds[2] / bounds[0]]
            else:
                width_ratio = [1., 1.]
            grid = gridspec.GridSpec(1, 2, width_ratios=width_ratio)
            ax1 = fig.add_subplot(grid[0])
            ax2 = fig.add_subplot(grid[1], sharey=ax1)
            axes = [(ax1, dim[:2]), (ax2, (dim[2], dim[1]))]
        elif len(dim) == 2:
            ax1 = fig.add_subplot(111)
            ax2 = None
            axes = [(ax1, dim)]
        else:
            assert False

        labeled = set([])

        for ax, (dim1, dim2) in axes:
            ax.plot(*rowcoords.T[[dim1, dim2]], 'o')
            ax.plot(*colcoords.T[[dim1, dim2]], 's')
            ax.set_xlabel(
                f'Component {dim1 + 1} ({self.inertia_e_dim[dim1]:.2%})')
            if dim2 not in labeled:
                ax.set_ylabel(
                    f'Component {dim2 + 1} ({self.inertia_e_dim[dim2]:.2%})')
                labeled.add(dim2)
            ax.grid(True, )
            # ax.axvline([0], linestyle='--', linewidth=1.0, color='k')
            # ax.axhline([0], linestyle='--', linewidth=1.0, color='k')

        fig.show()

    def format_summary(self):
        s = self.svd.s
        inertia_e = self.inertia_e_dim
        inertia_cum = np.cumsum(inertia_e)
        scree = [int(np.floor(20 * inertia))
                 for inertia in inertia_e / inertia_e.sum()]
        scree = ['*' * i for i in scree]
        vals = [f'{v:.2%}' for v in inertia_e]
        cums = [f'{c:.2%}' for c in inertia_cum]
        dims = [f'Dim={i+1}' for i in range(s.size)]
        header = ['dim', "%", "cum%", "screeplot"]
        width = [
            max(map(len, dims + [header[0]])),
            max(map(len, vals + [header[1]])),
            max(map(len, cums + [header[2]])),
            max(map(len, scree + [header[3]])),
        ]

        h = (
            f'{header[0]:<{width[0]}}',
            f'{header[1]:^{width[1]}}',
            f'{header[2]:^{width[2]}}',
            f'{header[3]:<{width[3]}}',
        )
        rows = [
            (
                f'{dim+1:<{width[0]}}',
                f'{val:>{width[1]}}',
                f'{cum:>{width[2]}}',
                f'{scree:<{width[3]}}'
            )
            for dim, val, cum, scree in zip(range(s.size,), vals, cums, scree)
        ]
        return "\n".join(map(" ".join, [h] + rows))

    def __repr__(self):
        return self.format_summary()


class MCA(CA):
    def __init__(self, *args, adjusted_inertia=None, **kwargs):
        super().__init__(*args, **kwargs)
        if adjusted_inertia is None:
            raise TypeError("Missing required parameter 'adjusted_inertia'")
        self.adjusted_inertia = adjusted_inertia

    @property
    def inertia_e_dim(self):
        return self.svd.s ** 2 / self.adjusted_inertia

    @property
    def total_inertia(self):
        return self.adjusted_inertia


def multiple_correspondence(B, counts, maxk=-1, solver=Auto):
    # type: (ndarray, Sequence[int]) -> MCA
    """
    Compute the multiple correspondence analysis on a *Burt* table.

    Parameters
    ----------
    B : (N, N) Matrix
        A burt table.
    counts : Sequence[int]
        The counts of levels/categories for each category. `sum(counts)`
        must equal `N`

    Returns
    -------
    solution: MCA
        A MCA solution

    References
    ----------
    .. [1] Oleg Nenadić and Michael Greenacre,  Computation of Multiple
       Correspondence Analysis, with code in R
    """
    if not isinstance(B, sp.spmatrix):
        B = np.asarray(B)
    N, N_ = B.shape
    assert N == N_
    Js = np.asarray(counts)
    assert Js.ndim == 1
    Q = Js.size  # number of groups
    assert np.all(Js > 0)
    J = np.sum(Js)
    assert J == N

    ca = correspondence(B, maxk=maxk, solver=solver)

    if Q <= 1:
        # Single variable. Maybe warn? Error?
        return MCA(ca.svd, ca.rowcoord, ca.colcoord, ca.rowmass, ca.colmass,
                   adjusted_inertia=ca.total_inertia)

    # compute the adjusted inertias
    total = ca.inertia
    sv = ca.svd.s

    adjusted_total = Q / (Q - 1) * (total - (J - Q) / (Q ** 2))
    mask = sv > 1. / Q
    svadj = np.where(
        mask,
        (Q / (Q - 1)) * (sv - 1/Q),
        0.0
    )
    # print("sv      ", sv.round(4)[:5])
    # print("adjusted", svadj.round(4)[:5])
    # print("ev      ", (svadj**2).round(4)[:5])
    # print("percenr ", (svadj**2) / adjusted_total)

    mca = MCA(
        SVD(ca.svd.U, svadj, ca.svd.Vh),
        ca.rowcoord, ca.colcoord,
        ca.rowmass, ca.colmass,
        adjusted_inertia=adjusted_total
    )
    # print(ca.inertia_e_dim)
    return mca


correspondence.mca = multiple_correspondence


if typing.TYPE_CHECKING:
    VariableSpec = Union[int, str, Variable]
    Contingency = Union[contingency.Discrete, contingency.Continuous]


def compute_contingencies(table, col_vars, row_var):
    # type: (Table, List[VariableSpec], VariableSpec) -> List[Contingency]
    domain = table.domain
    cont = table._compute_contingency(col_vars, row_var)
    contingencies = [
        contingency.get_contingency(
            arr, domain[colvar], domain[row_var], col_unk, row_unk, unk
        )
        for colvar, (arr, col_unk, row_unk, unk) in zip(col_vars, cont)
    ]
    return contingencies


def cross_tabulate(
        data,     # type: Orange.data.Table
        rowvars,  # type: List[DiscreteVariable]
        colvars   # type: List[DiscreteVariable]
):  # type: (...) -> np.ndarray
    """
    Cross tabulate a set of variables against each other.
    """
    nrow = sum(len(v.values) for v in rowvars)
    ncol = sum(len(v.values) for v in colvars)
    out = np.zeros((nrow, ncol), dtype=float)
    if out.size == 0:
        return out
    istart = iend = 0
    domain = data.domain
    colvar_indices = list(map(domain.index, colvars))
    for rowvar in rowvars:
        rowvar_ind = domain.index(rowvar)
        iend += len(rowvar.values)
        contingencies = compute_contingencies(data, colvar_indices, rowvar_ind)
        out[istart:iend] = np.hstack([np.asarray(c) for c in contingencies])
        istart = iend
    assert iend == nrow
    return out


def burt_table(
        data,       # type: Orange.data.Table
        variables,  # type: List[DiscreteVariable]
):  # type: (...) -> Tuple[List[Tuple[DiscreteVariable, str]], np.ndarray]
    """
    Construct a 'Burt table' (all values cross-tabulation) for `variables`.

    Return and ordered list of (variable, value) pairs and a
    numpy.ndarray contingency

    Parameters
    ----------
    data : Orange.data.Table
    variables : List[DiscreteVariable]

    Returns
    -------
    values : List[Tuple[DiscreteVariable, str]]
        A list
    table : (K, K) np.array
        Cross tabulation for all variable,value pairs
    """
    values = [(var, value) for var in variables for value in var.values]

    table = np.zeros((len(values), len(values)))
    counts = [len(attr.values) for attr in variables]
    offsets = np.r_[0, np.cumsum(counts)]

    for i in range(len(variables)):
        cxt = cross_tabulate(data, [variables[i]], variables[i:])
        assert cxt.shape == (counts[i], table.shape[1] - offsets[i])
        start1, end1 = offsets[i], offsets[i] + counts[i]
        table[start1:end1, start1:] = cxt
        table[start1:, start1:end1] = cxt.T
    assert np.all(table == table.T)
    return values, table


class CATransform(SharedComputeValue):
    def __init__(self, variables, factors, _indicator_domain=None):
        super().__init__(compute_shared=self._compute_shared)
        self.variables = tuple(variables)
        self.input_domain = Orange.data.Domain(variables)
        if _indicator_domain is None:
            c = Orange.preprocess.continuize.DomainContinuizer(
                multinomial_treatment=Orange.preprocess.Continuize.Indicators
            )
            _indicator_domain = c(self.input_domain)
        self._indicator_domain = _indicator_domain

        self.factors = factors
        if not len(self._indicator_domain) == self.factors.size:
            raise ValueError("Wrong factors size for the domain")

    def _compute_shared(self, data):
        inst = isinstance(data, Orange.data.Instance)
        if inst:
            data = Orange.data.Table([data])
        return data.transform(self._indicator_domain)

    def compute(self, data, shared_data):
        inst = isinstance(data, Orange.data.Instance)
        assert shared_data.domain.variables == self._indicator_domain.variables
        q = len(self.input_domain.variables)
        TX = np.dot(shared_data.X, self.factors / q)
        return TX[0] if inst else TX

    @classmethod
    def create_transformed(cls, namefmt, sourcevars, factors):
        # input domain
        domain = Orange.data.Domain(sourcevars)
        c = Orange.preprocess.continuize.DomainContinuizer(
            multinomial_treatment=Orange.preprocess.Continuize.Indicators
        )
        indicator_domain = c(domain)
        return [
            Orange.data.ContinuousVariable(
                namefmt.format(i + 1),
                compute_value=CATransform(sourcevars, factors[:, i],
                                          _indicator_domain=indicator_domain)
            )
            for i in range(factors.shape[1])
        ]


class MCA_(Projector):
    name = 'MCA'
    fit = None

    def __init__(self, n_components=2, solver=Auto):
        self.n_components = n_components
        self.solver = solver

    def __call__(self, data):
        # type: (Table) -> CAProjectionModel
        domain = data.domain
        catvars = [var for var in domain.attributes if var.is_discrete]
        if not catvars:
            raise ValueError("no categorical vars")
            return MCAProjectionModel(
                MCA(...), [], Domain([], domain.class_vars, domain.metas)
            )
        counts = [len(v.values) for v in catvars]
        if not sum(counts):
            raise ValueError("no categories")
            return MCAProjectionModel(
                MCA(...), [], Domain([], domain.class_vars, domain.metas)
            )

        B = burt_table(data, catvars)
        mca = multiple_correspondence(
            B, counts, solver=self.solver, maxk=self.n_components
        )
        cavars = CATransform.create_transformed("MCA{}", catvars, mca.cpc)
        tdomain = Domain(cavars, [], [])
        return CAProjectionModel(mca, catvars, tdomain)


class MCAProjectionModel(Projection):
    def __init__(self, mca, variables):
        # type: (MCA, List[DiscreteVariable]) -> None
        self.mca = mca
        #: The input variables
        self.variables = variables
        #: The transformed output domain
        self.domain = Domain(
            CATransform.create_transformed("MCA{}", variables, mca.rpc)
        )

    def __call__(self, data):
        return data.transform(self.domain)


class CA_(Projector):
    name = "CA"
    fit = None

    def __init__(self, n_components=2, solver=Auto):
        self.n_components = n_components
        self.solver = solver

    def __call__(self, data, rowvars=None, colvars=None):
        domain = data.domain
        if rowvars is None:
            rowvars = [v for v in domain.attributes if v.is_discrete]
        if colvars is None:
            colvars = [v for v in domain.class_vars if v.is_discrete]
        T = cross_tabulate(data, rowvars, colvars)
        ca = correspondence(T, self.n_components, solver=self.solver)
        return CAProjectionModel(ca, rowvars, colvars)


class CAProjectionModel(Projection):
    def __init__(self, ca, rowvars, colvars):
        # type: (CA, List[DiscreteVariable], List[DiscreteVariable]) -> None
        self.ca = ca
        #: The row variables
        self.rowvars = rowvars
        #: The transformed output domain
        self.colvars = colvars
        self.domain = Domain(
            CATransform.create_transformed("CA Row {}", rowvars, ca.rpc),
            CATransform.create_transformed("CA Col {}", colvars, ca.cpc),
        )

    def __call__(self, data):
        return data.transform(self.domain)
