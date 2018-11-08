import sys
from concurrent.futures import Future
from typing import Optional

import numpy as np
import scipy.spatial.distance

from AnyQt.QtWidgets import QFormLayout, QApplication
from AnyQt.QtCore import Qt, QTimer, QObject, QThread
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import pyqtgraph as pg

from Orange.data import ContinuousVariable, Domain, Table, Variable
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.projection.manifold import torgerson, MDS

from Orange.widgets import gui, settings, report
from Orange.widgets.settings import SettingProvider
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.visualize.owscatterplotgraph import (
    OWScatterPlotBase, OWProjectionWidget
)
from Orange.widgets.utils import concurrent as qconcurrent
from Orange.widgets.widget import Msg, OWWidget, Input, Output
from Orange.widgets.utils.annotated_data import (
    ANNOTATED_DATA_SIGNAL_NAME, create_annotated_table, create_groups_table,
    get_unique_names
)

ndarray = np.ndarray


def stress(X, distD):
    assert X.shape[0] == distD.shape[0] == distD.shape[1]
    D1_c = scipy.spatial.distance.pdist(X, metric="euclidean")
    D1 = scipy.spatial.distance.squareform(D1_c, checks=False)
    delta = D1 - distD
    delta_sq = np.square(delta, out=delta)
    return delta_sq.sum(axis=0) / 2


import scipy.linalg.blas as blas


def check_symmetric_packed(ar, name):
    # type: (np.ndarray, str) -> Tuple[np.ndarray, int]
    """
    Check that the array `ar` is a symmetric packed matrix.

    Raise an error if not.

    Parameters
    ----------
    ar : np.ndarray
        Array to check
    name : str
        Parameter name to use in the error message.

    Returns
    -------

    Raises
    ------
    ValueError
    """
    m, = ar.shape
    N = int(np.floor(np.sqrt(m * 2)))
    if N * (N + 1) != 2 * m:
        raise ValueError(
            "'{name}' is not in packed format (N * (N + 1) != 2 * {m} "
            "for all N)"
            .format(name=name, m=m))
    if not ar.flags.c_contiguous:
        raise ValueError("'{name}' array is not contiguous".format(name=name))
    return ar, N


def sym_matrix_sum_packed(ap, lower=False):
    # type: (np.ndarray, bool) -> np.ndarray
    """
    Return the sum of elements of the symmetric matrix in packed format over a
    single dimension (sum over rows/columns).

    Parameters
    ----------
    ap : np.ndarray

    lower : bool
        Is the `ap` the lower or upper.

    Returns
    -------

    """
    ap, N = check_symmetric_packed(ap, name="ap")
    ones = np.ones(N, dtype=ap.dtype)
    return spmv(ap, ones, lower=lower)


def spmv(ap, x, alpha=1.0, beta=0.0, y=None, lower=False):
    # type: (ndarray, ndarray, float, float, Optional[ndarray], bool) -> ndarray
    """
    Symmetric packed matrix vector multiplication.

    y = alpha * ap @ x + beta * y

    Example
    -------
    >>> a = np.array([1, 0, 0, 2, 0, 1], dtype=float)
    >>> spmv(a, np.array([1., 2., 3.], ))
    array([ 1.,  4.,  3.])
    """
    # Exposed in scipy 1.0.0
    ap, N = check_symmetric_packed(ap, name="ap")
    if ap.dtype.char == "d":
        spmv = blas.dspmv
    elif ap.dtype.char == "f":
        spmv = blas.sspmv
    elif ap.dtype.char == "D":
        spmv = blas.zspmv
    else:
        raise TypeError("Unsupported dtype: {}".format(ap.dtype))

    if alpha is None:
        alpha = 1.0
    # lower in fortran (row major) is upper in C and vice versa
    lower_f = not lower

    if y is None:
        assert beta == 0.0
        y = np.empty(N, ap.dtype)
        overwrite_y = True
    else:
        overwrite_y = False
    out = spmv(N, alpha, ap, x, beta=beta, y=y,
               lower=int(lower_f), overwrite_y=overwrite_y)
    assert (not overwrite_y) or out.ctypes.data == y.ctypes.data
    return out


def stress_packed(d1, d2):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Return the stress between two distance matrices `d1` and `d2` in symmetric
    packed format.

    Parameters
    ----------
    d1 : ((N + 1) * N // 2) np.ndarray
    d2 : ((N + 1) * N // 2) np.ndarray

    Returns
    -------
    stress : float

    """
    assert d1.shape == d2.shape
    m, = d1.shape
    N = int(np.floor(np.sqrt(m * 2)))
    if N * (N + 1) != m * 2:
        raise ValueError("Input is not a matrix in symmetric packed form")
    ss = scipy.spatial.distance.cdist(
        d1.reshape((1, m)), d2.reshape(1, m),
        metric="sqeuclidean"
    )
    return ss / (N ** 2)


def graph_laplacian_packed(WP, lower=False, overwrite_wp=False):
    # type: (np.ndarray, bool, bool) -> np.ndarray
    """
    Return a graph laplacian of the adjacency matrix `WP` in symmetric
    packed storage.

    Parameters
    ----------
    WP : ((N + 1) * N // 2, ) np.ndarray
        An symmetric adjacency matrix in packed storage (blas sp format)
    lower: bool
        Is `wp` in lower or upper packed format.
    overwrite_wp:
        If `True` then WP will be overwritten.

    Returns
    -------
    Lwp : ndarray
        A graph laplacian of WP in symmetric packed storage.
    """
    WP, N = check_symmetric_packed(WP, name="WP")
    if overwrite_wp:
        Lw = np.negative(WP, out=WP)
    else:
        Lw = -WP
    indices = np.empty(N, dtype=np.intp)
    indices[0] = 0
    if lower:
        np.cumsum(np.arange(2, N + 1, 1), out=indices[1:])
    else:
        np.cumsum(np.arange(N, 1, -1), out=indices[1:])
    Lw[indices] = 0
    Lw[indices] = -sym_matrix_sum_packed(Lw, lower=lower)
    return Lw


def smacof_update_matrix_packed(dist, delta, weights=None, out=None):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        B = np.divide(delta, dist, out=out)
        if weights is not None:
            B = np.multiply(B, weights, out=B)
    finitemask = np.isfinite(B)
    np.logical_not(finitemask, out=finitemask)
    B[finitemask] = 0
    del finitemask
    B = graph_laplacian_packed(B, overwrite_wp=True)
    return B


def condensed_to_sym_p(a):
    # type: (np.ndarray) -> np.ndarray
    """
    Extend a hollow condensed symmetric matrix (scipy.distance format) to
    symmetric packed format (blas).

    Parameters
    ----------
    a : (N - 1) * N // 2 array

    Returns
    -------
    a : ((N + 1) * N // 2) array
    """
    a, N_ = check_symmetric_packed(a, name="a")
    N = N_ + 1
    mask = condensed_to_sym_p_mask(N)
    out = np.zeros_like(mask, dtype=a.dtype)
    out[mask] = a
    return out


def condensed_to_sym_p_mask(N):
    # type: (int) -> np.ndarray
    assert N > 0
    mask = np.ones(((N + 1) * N) // 2, dtype=bool)
    mask[0] = False
    indices = np.arange(N, 1, -1)
    np.cumsum(indices, out=indices)
    mask[indices] = False
    return mask


def smacof_step(X, dissimilarity, delta=None, overwrite_delta=False):
    # type: (np.ndarray, np.ndarray, Optional[np.ndarray], bool) -> np.ndarray
    """
    Run a single SMACOF update step.

    Parameters
    ----------
    X : (N, k) ndarray
        The current point configuration
    dissimilarity : ((N + 1) * N // 2,) ndarray
        The optimal (desired) point dissimilarity in symmetric packed storage
        (blas sp format)
    delta : ((N + 1) * N // 2,) ndarray, optional
        Precomputed pairwise distances between points in X in symmetric packed
        storage.
    overwrite_delta : bool
        If `True` (and `delta` is supplied) then `delta` array is reused and
        overwritten.

    Returns
    -------
    X : (N, k) ndarray
        The updated point embeddings.
    References
    ----------
    Jan de Leeuw - Applications of Convex Analysis to Multidimensional Scaling
    """
    N, _ = X.shape
    assert dissimilarity.shape == ((N * (N + 1)) // 2, ), \
           "{} != {}".format(dissimilarity.shape, N)

    if delta is None:
        delta = scipy.spatial.distance.pdist(X, metric="euclidean")
        # from hollow condensed -> symmetric packed format
        delta = condensed_to_sym_p(delta)
        B_p_out = delta
    elif overwrite_delta and delta is not None:
        B_p_out = delta
    else:
        B_p_out = None
    B_p = smacof_update_matrix_packed(delta, dissimilarity, out=B_p_out)
    assert B_p.shape == delta.shape

    # X_out = 1. / N * B @ X
    X_out = np.empty_like(X)
    for i in range(X.shape[1]):
        X_out[:, i] = spmv(B_p, X[:, i], alpha=1. / N)
    return X_out


def smacof_iter(diss, embedding, max_iter=300, rtol=1e-5):
    """
    Return an iterator over successive SMACOF improved point embeddings.
    """
    done = False
    iterations_done = 0
    stress = np.finfo(np.float).max
    N, dim = embedding.shape
    sym_p_mask = condensed_to_sym_p_mask(N)
    delta_c = scipy.spatial.distance.pdist(embedding)
    delta = condensed_to_sym_p(delta_c)
    while not done:
        embedding_new = smacof_step(
            embedding, diss, delta, overwrite_delta=delta is not None)
        iterations_done += 1
        # need scipy 1.0.0 for out
        delta_c = scipy.spatial.distance.pdist(embedding_new, out=delta_c)
        delta[sym_p_mask] = delta_c
        stress_new = stress_packed(diss, delta)

        if iterations_done >= max_iter:
            done = True
        elif np.isclose(stress, stress_new, rtol=rtol, atol=0.0):
            done = True
        elif not np.isfinite(stress_new):
            raise RuntimeError("Non-finite stress value. Aborting")
        stress = stress_new
        embedding = embedding_new
        yield embedding, stress, iterations_done / max_iter


#: Maximum number of displayed closest pairs.
MAX_N_PAIRS = 10000


class OWMDSGraph(OWScatterPlotBase):
    #: Percentage of all pairs displayed (ranges from 0 to 20)
    connected_pairs = settings.Setting(5)

    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self.pairs_curve = None
        self.draw_pairs = True
        self._similar_pairs = None
        self.effective_matrix = None

    def set_effective_matrix(self, effective_matrix):
        self.effective_matrix = effective_matrix

    def pause_drawing_pairs(self):
        self.draw_pairs = False

    def resume_drawing_pairs(self):
        self.draw_pairs = True
        self.update_pairs(True)

    def update_coordinates(self):
        super().update_coordinates()
        self.update_pairs(reconnect=False)

    def update_pairs(self, reconnect):
        if self.pairs_curve:
            self.plot_widget.removeItem(self.pairs_curve)
        if not self.draw_pairs or self.connected_pairs == 0 \
                or self.effective_matrix is None \
                or self.scatterplot_item is None:
            return
        emb_x, emb_y = self.scatterplot_item.getData()
        if self._similar_pairs is None or reconnect:
            # This code requires storing lower triangle of X (n x n / 2
            # doubles), n x n / 2 * 2 indices to X, n x n / 2 indices for
            # argsort result. If this becomes an issue, it can be reduced to
            # n x n argsort indices by argsorting the entire X. Then we
            # take the first n + 2 * p indices. We compute their coordinates
            # i, j in the original matrix. We keep those for which i < j.
            # n + 2 * p will suffice to exclude the diagonal (i = j). If the
            # number of those for which i < j is smaller than p, we instead
            # take i > j. Among those that remain, we take the first p.
            # Assuming that MDS can't show so many points that memory could
            # become an issue, I preferred using simpler code.
            m = self.effective_matrix
            n = len(m)
            p = min(n * (n - 1) // 2 * self.connected_pairs // 100,
                    MAX_N_PAIRS * self.connected_pairs // 20)
            indcs = np.triu_indices(n, 1)
            sorted = np.argsort(m[indcs])[:p]
            self._similar_pairs = fpairs = np.empty(2 * p, dtype=int)
            fpairs[::2] = indcs[0][sorted]
            fpairs[1::2] = indcs[1][sorted]
        emb_x_pairs = emb_x[self._similar_pairs].reshape((-1, 2))
        emb_y_pairs = emb_y[self._similar_pairs].reshape((-1, 2))

        # Filter out zero distance lines (in embedding coords).
        # Null (zero length) line causes bad rendering artifacts
        # in Qt when using the raster graphics system (see gh-issue: 1668).
        (x1, x2), (y1, y2) = (emb_x_pairs.T, emb_y_pairs.T)
        pairs_mask = ~(np.isclose(x1, x2) & np.isclose(y1, y2))
        emb_x_pairs = emb_x_pairs[pairs_mask, :]
        emb_y_pairs = emb_y_pairs[pairs_mask, :]
        self.pairs_curve = pg.PlotCurveItem(
            emb_x_pairs.ravel(), emb_y_pairs.ravel(),
            pen=pg.mkPen(0.8, width=2, cosmetic=True),
            connect="pairs", antialias=True)
        self.plot_widget.addItem(self.pairs_curve)


class OWMDS(OWProjectionWidget):
    name = "MDS"
    description = "Two-dimensional data projection by multidimensional " \
                  "scaling constructed from a distance matrix."
    icon = "icons/MDS.svg"
    keywords = ["multidimensional scaling", "multi dimensional scaling"]

    class Inputs:
        data = Input("Data", Table, default=True)
        distances = Input("Distances", DistMatrix)
        data_subset = Input("Data Subset", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    settings_version = 3

    #: Initialization type
    PCA, Random, Jitter = 0, 1, 2

    #: Refresh rate
    RefreshRate = [
        ("Every iteration", 1),
        ("Every 5 steps", 5),
        ("Every 10 steps", 10),
        ("Every 25 steps", 25),
        ("Every 50 steps", 50),
        ("None", -1)
    ]

    #: Runtime state
    Running, Finished, Waiting = 1, 2, 3

    settingsHandler = settings.DomainContextHandler()

    max_iter = settings.Setting(300)
    initialization = settings.Setting(PCA)
    refresh_rate = settings.Setting(3)

    auto_commit = settings.Setting(True)

    graph = SettingProvider(OWMDSGraph)
    graph_name = "graph.plot_widget.plotItem"

    class Error(OWWidget.Error):
        not_enough_rows = Msg("Input data needs at least 2 rows")
        matrix_too_small = Msg("Input matrix must be at least 2x2")
        no_attributes = Msg("Data has no attributes")
        mismatching_dimensions = \
            Msg("Data and distances dimensions do not match.")
        out_of_memory = Msg("Out of memory")
        optimization_error = Msg("Error during optimization\n{}")

    def __init__(self):
        super().__init__()
        #: Input dissimilarity matrix
        self.matrix = None  # type: Optional[DistMatrix]
        #: Input subset data table
        self.subset_data = None  # type: Optional[Table]
        #: Data table from the `self.matrix.row_items` (if present)
        self.matrix_data = None  # type: Optional[Table]
        #: Input data table
        self.signal_data = None

        self._subset_mask = None  # type: Optional[np.ndarray]
        self._invalidated = False
        self.effective_matrix = None

        self.__smacof_task = None

        self.__update_loop = None
        # timer for scheduling updates
        self.__timer = QTimer(self, singleShot=True, interval=0)
        # self.__timer.timeout.connect(self.__next_step)
        self.__state = OWMDS.Waiting
        self.__in_next_step = False

        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = OWMDSGraph(self, box)
        self.graph.pause_drawing_pairs()
        box.layout().addWidget(self.graph.plot_widget)
        self.plot = self.graph.plot_widget
        g = self.graph.gui

        box = gui.vBox(self.controlArea, box=True)
        self.runbutton = gui.button(box, self, "Run optimization",
                                    callback=self._toggle_run)
        gui.comboBox(box, self, "refresh_rate", label="Refresh: ",
                     orientation=Qt.Horizontal,
                     items=[t for t, _ in OWMDS.RefreshRate],
                     callback=self.__invalidate_refresh)
        hbox = gui.hBox(box, margin=0)
        gui.button(hbox, self, "PCA", callback=self.do_PCA)
        gui.button(hbox, self, "Randomize", callback=self.do_random)
        gui.button(hbox, self, "Jitter", callback=self.do_jitter)

        g.point_properties_box(self.controlArea)
        box = g.effects_box(self.controlArea)
        g.add_control(box, gui.hSlider, "Show similar pairs:",
            master=self.graph, value="connected_pairs",
            minValue=0, maxValue=20, createLabel=False,
            callback=self._on_connected_changed
        )
        g.plot_properties_box(self.controlArea)

        self.size_model = g.points_models[2]
        self.size_model.order = g.points_models[2].order[:1] + ("Stress", ) + \
                                g.points_models[2].order[1:]

        self.controlArea.layout().addStretch(100)
        self.graph.box_zoom_select(self.controlArea)
        gui.auto_commit(self.controlArea, self, "auto_commit",
                        "Send Selection", "Send Automatically")

        self._initialize()
        self._executor = qconcurrent.ThreadExecutor(self)

    def selection_changed(self):
        self.commit()

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        """Set the input dataset.

        Parameters
        ----------
        data : Optional[Table]
        """
        if data is not None and len(data) < 2:
            self.Error.not_enough_rows()
            data = None
        else:
            self.Error.not_enough_rows.clear()

        self.signal_data = data

        if self.matrix is not None and data is not None and \
                len(self.matrix) == len(data):
            self.closeContext()
            self.data = data
            self.openContext(data)
        else:
            self._invalidated = True

    @Inputs.distances
    def set_disimilarity(self, matrix):
        """Set the dissimilarity (distance) matrix.

        Parameters
        ----------
        matrix : Optional[Orange.misc.DistMatrix]
        """

        if matrix is not None and len(matrix) < 2:
            self.Error.matrix_too_small()
            matrix = None
        else:
            self.Error.matrix_too_small.clear()

        self.matrix = matrix
        self.matrix_data = matrix.row_items if matrix is not None else None
        self._invalidated = True

    @Inputs.data_subset
    def set_subset_data(self, subset_data):
        """Set a subset of `data` input to highlight in the plot.

        Parameters
        ----------
        subset_data: Optional[Table]
        """
        self.subset_data = subset_data
        # invalidate the pen/brush when the subset is changed
        self._subset_mask = None  # type: Optional[np.ndarray]
        self.controls.graph.alpha_value.setEnabled(subset_data is None)

    def _clear(self):
        self.graph.set_effective_matrix(None)
        self.__set_smacof_task(None)
        self.__state = OWMDS.Waiting

    def _initialize(self):
        # clear everything
        self.closeContext()
        self._clear()
        self.Error.clear()
        self.effective_matrix = None
        self.embedding = None

        # if no data nor matrix is present reset plot
        if self.signal_data is None and self.matrix_data is None:
            self.data = None
            self.init_attr_values()
            return

        if self.signal_data is not None and self.matrix is not None and \
                len(self.signal_data) != len(self.matrix):
            self.Error.mismatching_dimensions()
            self._update_plot()
            return

        if self.signal_data is not None:
            self.data = self.signal_data
        elif self.matrix_data is not None:
            self.data = self.matrix_data

        if self.matrix is not None:
            self.effective_matrix = self.matrix
            if self.matrix.axis == 0 and self.data is self.matrix_data:
                self.data = None
        elif self.data.domain.attributes:
            preprocessed_data = MDS().preprocess(self.data)
            self.effective_matrix = Euclidean(preprocessed_data)
        else:
            self.Error.no_attributes()
            return

        self.init_attr_values()
        self.openContext(self.data)
        self.graph.set_effective_matrix(self.effective_matrix)

    def _toggle_run(self):
        if self.__state == OWMDS.Running:
            self.stop()
            self._invalidate_output()
        else:
            self.start()

    def start(self):
        if self.__state == OWMDS.Running:
            assert self.__smacof_task is not None
            return
        elif self.__state == OWMDS.Finished:
            # Resume/continue from a previous run
            assert self.__smacof_task is None
            self.__start()
        elif self.__state == OWMDS.Waiting and \
                self.effective_matrix is not None:
            self.__start()

    def stop(self):
        if self.__state == OWMDS.Running:
            assert self.__smacof_task is not None
            self.__set_smacof_task(None)

    def __start(self):
        self.graph.pause_drawing_pairs()
        X = self.effective_matrix
        embedding = self.embedding

        if embedding is None:
            if self.initialization == OWMDS.PCA:
                embedding = torgerson(X)
            else:
                embedding = np.random.random(size=(X.shape[0], 2))

        dist_packed = X[np.triu_indices(X.shape[0])]

        class State(QObject):
            updateReady = Signal()
            stop = False
            #: The current updated embedings, stress and % of  (max) iterations
            #: done
            current = (None, None, None)

        state = State()
        smacof_loop = smacof_iter(dist_packed, embedding, self.max_iter)

        def run():
            done = False
            X_new = None
            while not done:
                try:
                    X_new, stress, pdone = next(smacof_loop)
                except StopIteration:
                    done = True
                else:
                    state.current = (X_new, stress, pdone)
                    state.updateReady.emit()
                if state.stop:
                    smacof_loop.close()
                    raise KeyboardInterrupt
            assert X_new is not None
            return X_new
        f = self._executor.submit(run)

        state.updateTimer = QTimer(interval=0, singleShot=True)
        state.updateTimer.timeout.connect(self.__update_current_embedding)
        state.updateReady.connect(state.updateTimer.start)
        state.f = f
        state.watcher = qconcurrent.FutureWatcher(f)
        state.watcher.done.connect(self.__finish)
        self.__set_smacof_task(state)

    def __set_smacof_task(self, task):
        if self.__smacof_task is not None:
            self.__smacof_task.updateTimer.timeout.disconnect(
                self.__update_current_embedding)
            self.__smacof_task.watcher.done.disconnect(self.__finish)
            self.__smacof_task.stop = True
            # wait until completed and update the last updated X
            if self.__smacof_task.f.exception() is None:
                pass
                # self.embedding = self.__smacof_task.current[0]
            self.__smacof_task = None

        self.__smacof_task = task
        if task is not None:
            self.setBlocking(True)
            self.progressBarInit()
            self.setStatusMessage("Running...")
            self.runbutton.setText("Stop")
            self.__state = OWMDS.Running
        else:
            self.setBlocking(False)
            self.progressBarFinished()
            self.setStatusMessage("")
            self.runbutton.setText("Start")
            self.__state = OWMDS.Waiting

    def __update_current_embedding(self):
        assert self.thread() is QThread.currentThread()
        X, stress, progress = self.__smacof_task.current
        self.embedding = X
        self.progressBarSet(np.round(progress * 100, 1))
        self._update_plot()

    def __finish(self, f):
        # type: (Future[np.ndarray]) -> None
        self.setBlocking(False)
        self.progressBarFinished()
        self.setStatusMessage("")
        self.runbutton.setText("Start")
        self.__state = OWMDS.Finished
        assert f.done()
        assert self.__smacof_task is not None
        assert self.__smacof_task.f is f
        self.__smacof_task = None
        try:
            embedding = f.result()
        except MemoryError:
            self.Error.out_of_memory()
            self.graph.resume_drawing_pairs()
        except Exception as exc:  # pylint: disable=broad-except
            self.Error.optimization_error(str(exc), exc_info=True)
            self.graph.resume_drawing_pairs()
        else:
            self.embedding = embedding
            self.graph.resume_drawing_pairs()
            self.graph.update_coordinates()
            self.unconditional_commit()
            self._update_plot()

    def do_PCA(self):
        self.__invalidate_embedding(self.PCA)

    def do_random(self):
        self.__invalidate_embedding(self.Random)

    def do_jitter(self):
        self.__invalidate_embedding(self.Jitter)

    def __invalidate_embedding(self, initialization=PCA):
        def jitter_coord(part):
            span = np.max(part) - np.min(part)
            part += np.random.uniform(-span / 20, span / 20, len(part))

        # reset/invalidate the MDS embedding, to the default initialization
        # (Random or PCA), restarting the optimization if necessary.
        if self.embedding is None:
            return
        state = self.__state
        # if self.__update_loop is not None:
        #     self.__set_update_loop(None)
        if self.__smacof_task is not None:
            self.__set_smacof_task(None)

        X = self.effective_matrix

        if initialization == OWMDS.PCA:
            self.embedding = torgerson(X)
        elif initialization == OWMDS.Random:
            self.embedding = np.random.rand(len(X), 2)
        else:
            jitter_coord(self.embedding[:, 0])
            jitter_coord(self.embedding[:, 1])

        self._update_plot()

        # restart the optimization if it was interrupted.
        if state == OWMDS.Running:
            self.__start()

    def __invalidate_refresh(self):
        state = self.__state

        # if self.__update_loop is not None:
        #     self.__set_update_loop(None)
        if self.__smacof_task is not None:
            self.__set_smacof_task(None)

        # restart the optimization if it was interrupted.
        # TODO: decrease the max iteration count by the already
        # completed iterations count.
        if state == OWMDS.Running:
            self.__start()

    def handleNewSignals(self):
        if self._invalidated:
            self.graph.pause_drawing_pairs()
            self._invalidated = False
            self._initialize()
            self.start()

        self._update_plot()
        self.unconditional_commit()

    def _invalidate_output(self):
        self.commit()

    def _on_connected_changed(self):
        self.graph.set_effective_matrix(self.effective_matrix)
        self.graph.update_pairs(reconnect=True)

    def _update_plot(self):
        self.graph.reset_graph()
        if self.embedding is not None:
            self.graph.update_pairs(reconnect=True)

    def get_size_data(self):
        if self.attr_size == "Stress":
            return stress(self.embedding, self.effective_matrix)
        else:
            return super().get_size_data()

    def get_coordinates_data(self):
        return self.embedding.T if self.embedding is not None else (None, None)

    def get_subset_mask(self):
        if self.data is not None and self.subset_data is not None:
            return np.in1d(self.data.ids, self.subset_data.ids)

    def commit(self):
        if self.embedding is not None:
            names = get_unique_names([v.name for v in self.data.domain.variables],
                                     ["mds-x", "mds-y"])
            domain = Domain([ContinuousVariable(names[0]),
                             ContinuousVariable(names[1])])
            output = embedding = Table.from_numpy(
                domain, self.embedding[:, :2]
            )
        else:
            output = embedding = None

        if self.embedding is not None and self.data is not None:
            domain = self.data.domain
            domain = Domain(domain.attributes, domain.class_vars,
                            domain.metas + embedding.domain.attributes)
            output = self.data.transform(domain)
            output.metas[:, -2:] = embedding.X

        selection = self.graph.get_selection()
        if output is not None and len(selection) > 0:
            selected = output[selection]
        else:
            selected = None
        if self.graph.selection is not None and np.max(self.graph.selection) > 1:
            annotated = create_groups_table(output, self.graph.selection)
        else:
            annotated = create_annotated_table(output, selection)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self.graph.clear()
        self.__set_smacof_task(None)

    def send_report(self):
        if self.data is None:
            return

        def name(var):
            return var.name if isinstance(var, Variable) else var

        caption = report.render_items_vert((
            ("Color", name(self.attr_color)),
            ("Label", name(self.attr_label)),
            ("Shape", name(self.attr_shape)),
            ("Size", name(self.attr_size)),
            ("Jittering", self.graph.jitter_size != 0 and "{} %".format(
                self.graph.jitter_size))))
        self.report_plot()
        if caption:
            self.report_caption(caption)

    @classmethod
    def migrate_settings(cls, settings_, version):
        if version < 2:
            settings_graph = {}
            for old, new in (("label_only_selected", "label_only_selected"),
                             ("symbol_opacity", "alpha_value"),
                             ("symbol_size", "point_width"),
                             ("jitter", "jitter_size")):
                settings_graph[new] = settings_[old]
            settings_["graph"] = settings_graph
            settings_["auto_commit"] = settings_["autocommit"]

        if version < 3:
            if "connected_pairs" in settings_:
                connected_pairs = settings_["connected_pairs"]
                settings_["graph"]["connected_pairs"] = connected_pairs

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            domain = context.ordered_domain
            n_domain = [t for t in context.ordered_domain if t[1] == 2]
            c_domain = [t for t in context.ordered_domain if t[1] == 1]
            context_values = {}
            for _, old_val, new_val in ((domain, "color_value", "attr_color"),
                                        (c_domain, "shape_value", "attr_shape"),
                                        (n_domain, "size_value", "attr_size"),
                                        (domain, "label_value", "attr_label")):
                tmp = context.values[old_val]
                if tmp[1] >= 0:
                    context_values[new_val] = (tmp[0], tmp[1] + 100)
                elif tmp[0] != "Stress":
                    context_values[new_val] = None
                else:
                    context_values[new_val] = tmp
            context.values = context_values

        if version < 3 and "graph" in context.values:
            values = context.values
            values["attr_color"] = values["graph"]["attr_color"]
            values["attr_size"] = values["graph"]["attr_size"]
            values["attr_shape"] = values["graph"]["attr_shape"]
            values["attr_label"] = values["graph"]["attr_label"]


def main(argv=None):
    if argv is None:
        argv = sys.argv
    import gc
    app = QApplication(list(argv))
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "iris"

    data = Table(filename)
    w = OWMDS()
    w.set_data(data)
    w.set_subset_data(data[np.random.choice(len(data), 10)])
    w.handleNewSignals()

    w.show()
    w.raise_()
    rval = app.exec_()

    w.set_subset_data(None)
    w.set_data(None)
    w.handleNewSignals()

    w.saveSettings()
    w.onDeleteWidget()
    w.deleteLater()
    del w
    gc.collect()
    app.processEvents()
    return rval

if __name__ == "__main__":
    sys.exit(main())
