"""
Distributions
-------------

A widget for plotting variable distributions.

"""
import sys
import collections

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import numpy
import pyqtgraph as pg

import Orange.data
from Orange.statistics import distribution, contingency
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, colorpalette
from Orange.widgets.widget import InputSignal
from Orange.widgets.visualize.owlinearprojection import LegendItem, ScatterPlotItem


def selected_index(view):
    """Return the selected integer `index` (row) in the view.

    If no index is selected return -1

    `view` must be in single selection mode.
    """
    indices = view.selectedIndexes()
    assert len(indices) < 2, "View must be in single selection mode"
    if indices:
        return indices[0].row()
    else:
        return -1


class DistributionBarItem(pg.GraphicsObject):
    def __init__(self, geometry, dist, colors):
        super().__init__()
        self.geometry = geometry
        self.dist = dist
        self.colors = colors
        self.__picture = None

    def paint(self, painter, options, widget):
        if self.__picture is None:
            self.__paint()
        painter.drawPicture(0, 0, self.__picture)

    def boundingRect(self):
        return self.geometry

    def __paint(self):
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        pen = QtGui.QPen(QtGui.QBrush(Qt.white), 0.5)
        pen.setCosmetic(True)
        painter.setPen(pen)

        geom = self.geometry
        x, y = geom.x(), geom.y()
        w, h = geom.width(), geom.height()
        for d, c in zip(self.dist, self.colors):
            if d == 0:
                continue
            painter.setBrush(QtGui.QBrush(c.lighter()))
            painter.drawRect(QtCore.QRectF(x, y, w, d * h))
            y += d * h
        painter.end()

        self.__picture = picture


class OWDistributions(widget.OWWidget):
    name = "Distributions"
    description = "Display value distributions of a data feature in a graph."
    icon = "icons/Distribution.svg"
    priority = 100
    inputs = [InputSignal("Data", Orange.data.Table, "set_data",
                          doc="Set the input data set")]

    settingsHandler = settings.DomainContextHandler()
    #: Selected variable index
    variable_idx = settings.ContextSetting(-1)
    #: Selected group variable
    groupvar_idx = settings.ContextSetting(0)

    Hist, ASH, Kernel = 0, 1, 2
    #: Continuous variable density estimation method
    cont_est_type = settings.Setting(ASH)
    relative_freq = settings.Setting(False)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None

        self.distributions = None
        self.contingencies = None
        self.var = self.cvar = None
        varbox = gui.widgetBox(self.controlArea, "Variable")

        self.varmodel = itemmodels.VariableListModel()
        self.groupvarmodel = itemmodels.VariableListModel()

        self.varview = QtGui.QListView(
            selectionMode=QtGui.QListView.SingleSelection)
        self.varview.setSizePolicy(
            QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.varview.setModel(self.varmodel)
        self.varview.setSelectionModel(
            itemmodels.ListSingleSelectionModel(self.varmodel))
        self.varview.selectionModel().selectionChanged.connect(
            self._on_variable_idx_changed)
        varbox.layout().addWidget(self.varview)
        gui.separator(varbox, 8, 8)
        gui.comboBox(
            varbox, self, "cont_est_type", label="Show continuous variables by",
            valueType=int,
            items=["Histograms", "Average shifted histograms",
                   "Kernel density estimators"],
            callback=self._on_cont_est_type_changed)

        box = gui.widgetBox(self.controlArea, "Group by")
        self.groupvarview = QtGui.QListView(
            selectionMode=QtGui.QListView.SingleSelection)
        self.groupvarview.setFixedHeight(100)
        self.groupvarview.setSizePolicy(
            QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        self.groupvarview.setModel(self.groupvarmodel)
        self.groupvarview.selectionModel().selectionChanged.connect(
            self._on_groupvar_idx_changed)
        box.layout().addWidget(self.groupvarview)
        self.cb_rel_freq = gui.checkBox(
            box, self, "relative_freq", "Show relative frequencies",
            callback=self._on_relative_freq_changed)

        plotview = pg.PlotWidget(background=None)
        self.mainArea.layout().addWidget(plotview)
        w = QtGui.QLabel()
        w.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        self.mainArea.layout().addWidget(w, Qt.AlignCenter)

        self.plot = pg.PlotItem()
#         self.plot.getViewBox().setMouseEnabled(False, False)
        self.plot.getViewBox().setMenuEnabled(False)
        plotview.setCentralItem(self.plot)

        pen = QtGui.QPen(self.palette().color(QtGui.QPalette.Text))
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setPen(pen)

        self._legend = LegendItem()
        self._legend.setParentItem(self.plot.getViewBox())
        self._legend.hide()
        self._legend.anchor((1, 0), (1, 0))

    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.data = data
        if self.data is not None:
            domain = self.data.domain
            self.varmodel[:] = list(domain)
            self.groupvarmodel[:] = \
                ["(None)"] + [var for var in domain if var.is_discrete]
            if domain.has_discrete_class:
                self.groupvar_idx = \
                    list(self.groupvarmodel).index(domain.class_var)
            self.openContext(domain)
            self.variable_idx = min(max(self.variable_idx, 0),
                                    len(self.varmodel) - 1)
            self.groupvar_idx = min(max(self.groupvar_idx, 0),
                                    len(self.groupvarmodel) - 1)
            itemmodels.select_row(self.groupvarview, self.groupvar_idx)
            itemmodels.select_row(self.varview, self.variable_idx)
            self._setup()

    def clear(self):
        self.plot.clear()
        self.varmodel[:] = []
        self.groupvarmodel[:] = []
        self.variable_idx = -1
        self.groupvar_idx = 0
        self._legend.clear()
        self._legend.hide()

    def _setup(self):
        self.plot.clear()
        self._legend.clear()
        self._legend.hide()

        varidx = self.variable_idx
        self.var = self.cvar = None
        if varidx >= 0:
            self.var = self.varmodel[varidx]
        if self.groupvar_idx > 0:
            self.cvar = self.groupvarmodel[self.groupvar_idx]
        self.set_left_axis_name()
        self.enable_disable_rel_freq()
        if self.var is None:
            return
        if self.cvar:
            self.contingencies = \
                contingency.get_contingency(self.data, self.var, self.cvar)
            self.display_contingency()
        else:
            self.distributions = \
                distribution.get_distribution(self.data, self.var)
            self.display_distribution()

    def _density_estimator(self):
        if self.cont_est_type == OWDistributions.Hist:
            def hist(dist):
                dist = numpy.asarray(dist)
                X, W = dist
                nbins = hist_nbins_scott(X, W)
                h, edges = numpy.histogram(X, bins=max(10, nbins), weights=W)
                return edges, h
            return hist
        elif self.cont_est_type == OWDistributions.ASH:
            def ash(dist):
                dist = numpy.asarray(dist)
                X, W = dist
                nbins = hist_nbins_scott(X, W)
                return ASH_for_distribution(dist, bins=nbins, m=5)
            return ash
        elif self.cont_est_type == OWDistributions.Kernel:
            def rkernel(dist):
                dist = numpy.asarray(dist)
                X, W = dist
                bw = kde_bw_silverman(X, weights=W)
                return rect_kernel_curve(dist, bandwidth=bw)
            return rkernel

    def _conditional_density_estimator(self):
        if self.cont_est_type == OWDistributions.Hist:
            def hist(cont):
                nbins = hist_nbins_scott(cont.values, cont.counts.sum(axis=0))
                return histogram_for_contingency(cont, bins=max(10, nbins))
            return hist
        elif self.cont_est_type == OWDistributions.ASH:
            def ash(cont):
                nbins = hist_nbins_scott(cont.values, cont.counts.sum(axis=0))
                amin, amax = cont.values.min(), cont.values.max()
                basebins = histogram_bins([amin, amax], nbins)
                return ASH_for_contingency(cont, bins=basebins, m=5)
            return ash
        elif self.cont_est_type == OWDistributions.Kernel:
            def rkernel(cont):
                bw = kde_bw_silverman(
                    cont.values, weights=cont.counts.sum(axis=0))
                curves = [rect_kernel_curve(dist, bandwidth=bw)
                          for dist in cont]
                W = cont.counts.sum(axis=1)
                W /= W.sum() if W.sum() > 0 else 1
                return [(X, Y * w) for (X, Y), w in zip(curves, W)]

            return rkernel
        else:
            assert False

    def display_distribution(self):
        dist = self.distributions
        var = self.var
        assert len(dist) > 0
        self.plot.clear()

        bottomaxis = self.plot.getAxis("bottom")
        bottomaxis.setLabel(var.name)

        self.set_left_axis_name()
        if var and var.is_continuous:
            bottomaxis.setTicks(None)
            curve_est = self._density_estimator()
            edges, curve = curve_est(dist)
            if len(edges):
                item = pg.PlotCurveItem()
                item.setData(edges, curve, antialias=True, stepMode=True,
                             fillLevel=0, brush=QtGui.QBrush(Qt.gray),
                             pen=QtGui.QColor(Qt.white))
                self.plot.addItem(item)
        else:
            bottomaxis.setTicks([list(enumerate(var.values))])
            for i, w in enumerate(dist):
                geom = QtCore.QRectF(i - 0.33, 0, 0.66, w)
                item = DistributionBarItem(geom, [1.0],
                                           [QtGui.QColor(128, 128, 128)])
                self.plot.addItem(item)

    def _on_relative_freq_changed(self):
        self.set_left_axis_name()
        if self.cvar and self.cvar.is_discrete:
            self.display_contingency()
        else:
            self.display_distribution()

    def display_contingency(self):
        """
        Set the contingency to display.
        """
        cont = self.contingencies
        var, cvar = self.var, self.cvar
        assert len(cont) > 0
        self.plot.clear()
        self._legend.clear()

        bottomaxis = self.plot.getAxis("bottom")
        bottomaxis.setLabel(var.name)

        palette = colorpalette.ColorPaletteGenerator(len(cvar.values))
        colors = [palette[i] for i in range(len(cvar.values))]

        def isvalid(edges, values):
            """Is the curve valid (defines at least one patch)."""
            return len(edges) > 0

        if var and var.is_continuous:
            bottomaxis.setTicks(None)

            curve_est = self._conditional_density_estimator()
            curves = curve_est(cont)

            # Compute the cumulative curves (stacked on top of each other),
            # but preserve the invalid ones.
            cum_curves = []
            cumulative_curve = [], []
            for X, Y  in curves:
                if isvalid(*cumulative_curve) and isvalid(X, Y):
                    cumulative_curve = sum_rect_curve(X, Y, *cumulative_curve)
                    cum_curves.append(cumulative_curve)
                elif isvalid(X, Y):
                    cumulative_curve = X, Y
                    cum_curves.append(cumulative_curve)
                else:
                    # X, Y is not valid. Preserve it in the list.
                    cum_curves.append(([], []))

            assert len(cum_curves) == len(cvar.values)

            # plot the cumulative curves 'back to front'.
            for (X, Y), color in reversed(list(zip(cum_curves, colors))):
                if isvalid(X, Y):
                    item = pg.PlotCurveItem()
                    pen = QtGui.QPen(QtGui.QBrush(Qt.white), 0.5)
                    pen.setCosmetic(True)
                    item.setData(X, Y, antialias=True, stepMode=True,
                                 fillLevel=0, brush=QtGui.QBrush(color.lighter()),
                                 pen=pen)
                    self.plot.addItem(item)

        elif var and var.is_discrete:
            bottomaxis.setTicks([list(enumerate(var.values))])

            cont = numpy.array(cont)
            for i, (value, dist) in enumerate(zip(var.values, cont.T)):
                dsum = sum(dist)
                geom = QtCore.QRectF(i - 0.333, 0, 0.666,
                                     100 if self.relative_freq else dsum)
                item = DistributionBarItem(geom, dist / dsum, colors)
                self.plot.addItem(item)

        for color, name in zip(colors, cvar.values):
            self._legend.addItem(
                ScatterPlotItem(pen=color, brush=color, size=10, shape="s"),
                name
            )
        self._legend.show()

    def set_left_axis_name(self):
        label = "Frequency"
        if self.var and self.var.is_continuous and \
                self.cont_est_type == OWDistributions.Kernel:
            label = "Density"
        elif self.var and self.var.is_discrete and self.cvar and \
                self.relative_freq:
            label = "Relative frequency"

        self.plot.getAxis("left").setLabel(label)

    def enable_disable_rel_freq(self):
        self.cb_rel_freq.setDisabled(
            self.var is None or self.cvar is None or self.var.is_continuous)

    def _on_variable_idx_changed(self):
        self.variable_idx = selected_index(self.varview)
        self._setup()

    def _on_groupvar_idx_changed(self):
        self.groupvar_idx = selected_index(self.groupvarview)
        self._setup()

    def _on_cont_est_type_changed(self):
        self.set_left_axis_name()
        if self.data is not None:
            self._setup()

    def onDeleteWidget(self):
        self.plot.clear()
        super().onDeleteWidget()


def dist_sum(D1, D2):
    """
    A sum of two continuous distributions.
    """
    X1, W1 = D1
    X2, W2 = D2
    X = numpy.r_[X1, X2]
    W = numpy.r_[W1, W2]
    sort_ind = numpy.argsort(X)
    X, W = X[sort_ind], W[sort_ind]

    unique, uniq_index = numpy.unique(X, return_index=True)
    spans = numpy.diff(numpy.r_[uniq_index, len(X)])
    W = [numpy.sum(W[start:start + span])
         for start, span in zip(uniq_index, spans)]
    W = numpy.array(W)
    assert W.shape[0] == unique.shape[0]
    return unique, W


def histogram_bins(a, bins=10):
    _, edges = numpy.histogram(a, bins)
    return edges


def histogram_for_distribution(dist, bins=10):
    dist = numpy.asarray(dist)
    X, W = dist

    if X.size == 0:
        return numpy.array([], dtype=float), numpy.array([], dtype=float)

    counts, edges = numpy.histogram(X, bins, weights=W)
    return edges, counts


def histogram_for_contingency(cont, bins=10):
    bins = histogram_bins([cont.values.min(), cont.values.max()], bins=bins)
    return [histogram_for_distribution(dist, bins) for dist in cont]


def ASH_for_distribution(dist, bins=10, m=5):
    dist = numpy.asarray(dist)
    X, W = dist
    if X.size == 0:
        return numpy.array([], dtype=float), numpy.array([], dtype=float)

    if numpy.isscalar(bins):
        bins = histogram_bins(X, bins)

    hist, bin_edges = ash(X, bins=bins, m=m, weights=W)
    return bin_edges, hist


def ASH_for_contingency(cont, bins=10, m=5):
    if numpy.isscalar(bins):
        bins = histogram_bins(cont.values, bins)

    return [ASH_for_distribution(dist, bins=bins, m=m) for dist in cont]


def ash_smooth(hist, m=5):
    hist = numpy.asarray(hist)
    if m < 0 or m % 2 != 1:
        raise ValueError("m % 2 != 1")

    kernel = triangular_kernel(m)
    kernel /= kernel.sum()
    if hist.size < kernel.size:
        raise ValueError("hist.size < m")

    return numpy.convolve(hist, kernel, mode="same")


def ash(a, bins=10, m=5, weights=None):
    """
    Compute the average shifted histogram.

    Parameters
    ----------
    a : array-like
        Input data array.
    bins : int or array-like
        Base histogram bins. If array-like then it MUST contain
        equi-width bin edges.
    m : int
        Smoothing parameter.
    weights : array-like
        An array of weights of the same shape as `a`
    """
    bins = histogram_bins(a, bins)
    ash_bins = ash_resample_bins(bins, m=m)
    hist, bin_edges = numpy.histogram(a, ash_bins, weights=weights)
    ash = ash_smooth(hist, m=m)
    return ash, bin_edges


def ash_sample_bins(amin, amax, h, m):
    if m % 2 != 1:
        raise ValueError("m % 2 != 1")

    k = m // 2

    if amin == amax:
        # This is bad, but consistent with numpy.histogram
        amin, amax = amin - 0.5, amax + 0.5

    delta = h / m
    offset = k * delta
    nbins = max(numpy.ceil((amax - amin + 2 * offset) / delta), m)
    bins = numpy.linspace(amin - offset, amax + offset, nbins + 1,
                          endpoint=True)
    return bins


def ash_resample_bins(basebins, m):
    assert m % 2 == 1

    basebins = numpy.asarray(basebins, dtype=float)
    binwidths = numpy.diff(basebins, n=1)

    if numpy.any(numpy.abs(numpy.diff(binwidths)) > 1e-9):
        raise ValueError

    h = numpy.mean(binwidths)
    amin, amax = basebins[0], basebins[-1]
    return ash_sample_bins(amin, amax, h, m)


def triangular_kernel(n):
    if n % 2 != 1:
        raise ValueError("n % 2 != 1")

    a = numpy.linspace(-1, 1, n + 2, endpoint=True)[1:-1]
    return numpy.clip(1 - numpy.abs(a), 0, 1)


def rect_kernel_curve(dist, bandwidth=None):
    """
    Return a rectangular kernel density curve for `dist`.
    """
    dist = numpy.array(dist)
    if dist.size == 0:
        return numpy.array([], dtype=float), numpy.array([], dtype=float)

    X, W = dist

    if bandwidth is None:
        # Silverman's rule of thumb.
        bandwidth = kde_bw_silverman(X, W)

    bottom_edges = X - bandwidth / 2
    top_edges = X + bandwidth / 2

    edges = numpy.hstack((bottom_edges, top_edges))
    edge_weights = numpy.hstack((W, -W))

    sort_ind = numpy.argsort(edges)
    edges = edges[sort_ind]
    edge_weights = edge_weights[sort_ind]

    # NOTE: The final cumulative sum element is 0
    curve = numpy.cumsum(edge_weights)[:-1]
    curve /= numpy.sum(W) * bandwidth
    return edges, curve


def IQR(a, weights=None):
    """
    Interquartile range of `a`.
    """
    q1, q3 = weighted_quantiles(a, [0.25, 0.75], weights=weights)
    return q3 - q1


def kde_bw_silverman(a, weights=None):
    """
    Silverman's rule of thumb for kernel bandwidth selection.
    """
    A = weighted_std(a, weights=weights)
    iqr = IQR(a, weights=weights)
    if iqr > 0:
        A = min(A, iqr / 1.34)

    return 0.9 * A * (a.size ** -0.2)


def sum_rect_curve(Xa, Ya, Xb, Yb):
    """
    Sum two curves (i.e. stack one over the other).
    """
    X = numpy.r_[Xa, Xb]
    Y = numpy.r_[Ya, 0, Yb, 0]
    assert X.shape == Y.shape

    dY = numpy.r_[Y[0], numpy.diff(Y)]
    sort_ind = numpy.argsort(X)
    X = X[sort_ind]
    dY = dY[sort_ind]

    unique, uniq_index = numpy.unique(X, return_index=True)
    spans = numpy.diff(numpy.r_[uniq_index, len(X)])
    dY = [numpy.sum(dY[start:start + span])
          for start, span in zip(uniq_index, spans)]
    dY = numpy.array(dY)
    assert dY.shape[0] == unique.shape[0]
    # NOTE: The final cumulative sum element is 0
    Y = numpy.cumsum(dY)[:-1]

    return unique, Y


def hist_nbins_scott(a, weights=None):
    """
    Scott's normal reference rule for histogram bin count.
    """
    if weights is not None:
        std = weighted_std(a, weights=weights)
    else:
        std = numpy.std(a)
    n = a.size if weights is None else numpy.sum(weights)
    h = 3.5 * std * (n ** (- 1 / 3))

    if h > 0:
        return int(numpy.ceil((a.max() - a.min()) / h))
    else:
        return 1


def hist_nbins_FD(a, weights=None):
    """
    Freedman-Diaconis's choice for histogram bin count.
    """
    q1, q3 = weighted_quantiles(a, weights=weights)
    irq = q3 - q1
    n = a.size if weights is None else numpy.sum(weights)
    h = 2 * irq * (n ** (- 1 / 3))

    if h > 0:
        return int(numpy.ceil((a.max() - a.min()) / h))
    else:
        return 1


def weighted_std(a, axis=None, weights=None, ddof=0):
    mean = numpy.average(a, axis=axis, weights=weights)

    if axis is not None:
        shape = shape_reduce_keep_dims(a.shape, axis)
        mean = mean.reshape(shape)

    sq_diff = numpy.power(a - mean, 2)
    mean_sq_diff, wsum = numpy.average(
        sq_diff, axis=axis, weights=weights, returned=True
    )

    if ddof != 0:
        mean_sq_diff *= wsum / (wsum - ddof)

    return numpy.sqrt(mean_sq_diff)


def weighted_quantiles(a, prob=[0.25, 0.5, 0.75], alphap=0.4, betap=0.4,
                       axis=None, weights=None):
    a = numpy.asarray(a)
    prob = numpy.asarray(prob)

    sort_ind = numpy.argsort(a, axis)
    a = a[sort_ind]

    if weights is None:
        weights = numpy.ones_like(a)
    else:
        weights = numpy.asarray(weights)
        weights = weights[sort_ind]

    n = numpy.sum(weights)
    k = numpy.cumsum(weights, axis)

    # plotting positions for the known n knots
    pk = (k - alphap * weights) / (n + 1 - alphap * weights - betap * weights)

#     m = alphap + prob * (1 - alphap - betap)

    return numpy.interp(prob, pk, a, left=a[0], right=a[-1])


def shape_reduce_keep_dims(shape, axis):
    if shape is None:
        return ()

    shape = list(shape)
    if isinstance(axis, collections.Sequence):
        for ax in axis:
            shape[ax] = 1
    else:
        shape[axis] = 1
    return tuple(shape)


def main(argv=None):
    import gc
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QtGui.QApplication(argv)
    w = OWDistributions()
    w.show()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "heart_disease"
    data = Orange.data.Table(filename)
    w.set_data(data)
    w.handleNewSignals()
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.saveSettings()
    w.deleteLater()
    del w
    app.processEvents()
    gc.collect()
    return rval


if __name__ == "__main__":
    sys.exit(main())
