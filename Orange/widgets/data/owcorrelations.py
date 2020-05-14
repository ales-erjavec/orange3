"""
Correlations widget
"""
import numbers
from enum import IntEnum
from functools import partial
from operator import attrgetter
from types import SimpleNamespace
from itertools import combinations, groupby, chain
from typing import Optional

import numpy as np
from scipy.stats import spearmanr, distributions
from scipy import special

from sklearn.cluster import KMeans

from AnyQt.QtCore import Qt, QItemSelectionModel, QItemSelection, \
    QSize, pyqtSignal as Signal, QModelIndex, QSortFilterProxyModel
from AnyQt.QtGui import QStandardItem, QColor
from AnyQt.QtWidgets import QHeaderView, QTableView, QLineEdit, QPushButton, \
    QStyleOptionViewItem

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.distance.distance import _corrcoef2, _spearmanr2
from Orange.preprocess import SklImpute, Normalize, Remove
from Orange.statistics.util import FDR
from Orange.widgets import gui
from Orange.widgets.gui import TableBarItem
from Orange.widgets.settings import Setting, ContextSetting, \
    DomainContextHandler
from Orange.widgets.utils import vartype, itemmodels
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.tablemodeladapter import TableModelDispatcher
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.widget import OWWidget, AttributeList, Msg

NAN = 2
SIZE_LIMIT = 1000000


class CorrelationType(IntEnum):
    """
    Correlation type enumerator. Possible correlations: Pearson, Spearman.
    """
    PEARSON, SPEARMAN = 0, 1

    @staticmethod
    def items():
        """
        Texts for correlation types. Can be used in gui controls (eg. combobox).
        """
        return ["Pearson correlation", "Spearman correlation"]


def pearsonr(x: np.ndarray):
    n, p = x.shape
    r = np.corrcoef(x, rowvar=False)
    return pearsonr_pval(r, n - 2)


def pearsonr_pval(r, df):
    # from stats.pearsonr
    r1_mask = np.abs(r) < 1.0
    with np.errstate(divide="ignore"):
        t_squared = r ** 2 * (df / ((1.0 - r) * (1.0 + r)))
    prob = special.betainc(0.5 * df, 0.5, df / (df + t_squared), where=r1_mask)
    prob[~r1_mask] = 0.0
    return r, prob


def pearsonr2(x: np.ndarray, y: np.ndarray):
    x_n, p1 = x.shape
    y_n, p2 = y.shape
    if x_n != y_n:
        raise ValueError(f"{x.shape[0]} != {y.shape[0]}")
    n = x_n
    r = _corrcoef2(x, y)
    return pearsonr_pval(r, n - 2)


def spearman2(x: np.ndarray, y: np.ndarray):
    x_n, p1 = x.shape
    y_n, p2 = y.shape
    if x_n != y_n:
        raise ValueError(f"{x.shape[0]} != {y.shape[0]}")
    n = x_n
    r = _spearmanr2(x, y)
    return spearman_pval(r, n - 2)


def spearman_pval(r, df):
    # from stats.spearmanr
    r1_mask = np.abs(r) < 1.0
    with np.errstate(divide="ignore"):
        t = r * np.sqrt(df / ((r + 1.0) * (1.0 - r)), where=r1_mask)
    prob = np.full_like(r, np.nan)
    prob[~r1_mask] = 0.0
    prob[r1_mask] = 2 * distributions.t.sf(np.abs(t[r1_mask]), df)
    return prob


def spearman_pval_1(r, df):
    # from stats.spearmanr
    r1_mask = np.abs(r) < 1.0
    with np.errstate(divide="ignore"):
        t_squared = r ** 2 * (df / ((r + 1.0) * (1.0 - r)))
    prob = special.betainc(0.5 * df, 0.5, df / (df + t_squared), where=r1_mask)
    prob[~r1_mask] = 0.0
    return prob


NEGATIVE_COLOR = QColor(70, 190, 250)
POSITIVE_COLOR = QColor(170, 242, 43)

CorrelationVarPair = Qt.UserRole + 2
CorrelationPValRole = Qt.UserRole + 3


class OWCorrelations(OWWidget):
    name = "Correlations"
    description = "Compute all pairwise attribute correlations."
    icon = "icons/Correlations.svg"
    priority = 1106

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)
        features = Output("Features", AttributeList)
        correlations = Output("Correlations", Table)

    want_control_area = False

    correlation_type: int

    settings_version = 3
    settingsHandler = DomainContextHandler()
    selection = ContextSetting([])
    feature = ContextSetting(None)
    correlation_type = Setting(0)

    class Information(OWWidget.Information):
        removed_cons_feat = Msg("Constant features have been removed.")

    class Warning(OWWidget.Warning):
        not_enough_vars = Msg("At least two continuous features are needed.")
        not_enough_inst = Msg("At least two instances are needed.")

    def __init__(self):
        super().__init__()
        self.data = None  # type: Optional[Table]
        self.cont_data = None  # type: Optional[Table]
        self.model = None  # type: Optional[TableModelDispatcher]

        self.proxy_model = QSortFilterProxyModel(
            self, filterCaseSensitivity=False)

        # GUI
        box = gui.vBox(self.mainArea)
        self.correlation_combo = gui.comboBox(
            box, self, "correlation_type", items=CorrelationType.items(),
            orientation=Qt.Horizontal, callback=self._correlation_combo_changed
        )

        self.feature_model = DomainModel(
            order=DomainModel.ATTRIBUTES, separators=False,
            placeholder="(All combinations)", valid_types=ContinuousVariable)
        gui.comboBox(
            box, self, "feature", callback=self._feature_combo_changed,
            model=self.feature_model
        )

        gui.separator(box)
        self.filter_line = QLineEdit(
            placeholderText="Filter...",
        )
        self.filter_line.textChanged.connect(
            self.proxy_model.setFilterFixedString)
        self.rank_table = QTableView(
            selectionBehavior=QTableView.SelectRows,
            selectionMode=QTableView.SingleSelection,
            showGrid=False,
            editTriggers=QTableView.NoEditTriggers
        )
        header = self.rank_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.hide()
        self.rank_table.setItemDelegateForColumn(0, TableBarItem(self.rank_table))
        self.rank_table.setModel(self.proxy_model)
        self.rank_table.selectionModel().selectionChanged.connect(
            self._selection_changed)
        box.layout().addWidget(self.filter_line)
        box.layout().addWidget(self.rank_table)

        # button_box = gui.hBox(self.mainArea)
        button = QPushButton("Run", default=True)
        box.layout().addWidget(button)

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

    def sizeHint(self):
        return super().sizeHint().expandedTo(QSize(350, 400))

    def _correlation_combo_changed(self):
        self.apply()

    def _feature_combo_changed(self):
        self.apply()

    def _selection_changed(self, *args):
        selmodel = self.rank_table.selectionModel()
        indices = [idx.row() for idx in selmodel.selectedRows(0)]
        model = self.proxy_model

        def vpair(row):
            idx1 = model.index(row, 1)
            idx2 = model.index(row, 2)
            v1, v2 = idx1.data(Qt.UserRole), idx2.data(Qt.UserRole)
            return [v1, v2]

        if indices:
            self.selection = vpair(indices[0])
        else:
            self.selection = []
        self.commit()

    def _select(self):
        model = self.vizrank.rank_table.model()
        if not model.rowCount():
            return
        selection = QItemSelection()

        # This flag is needed because data in the model could be
        # filtered by a feature and therefore selection could not be found
        selection_in_model = False
        if self.selection:
            sel_names = sorted(var.name for var in self.selection)
            for i in range(model.rowCount()):
                # pylint: disable=protected-access
                names = sorted(x.name for x in model.data(
                    model.index(i, 0), CorrelationVarPair))
                if names == sel_names:
                    selection.select(model.index(i, 0),
                                     model.index(i, model.columnCount() - 1))
                    selection_in_model = True
                    break
        if not selection_in_model:
            selection.select(model.index(0, 0),
                             model.index(0, model.columnCount() - 1))
        self.vizrank.rank_table.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.clear_messages()
        self.data = data
        self.cont_data = None
        self.selection = []
        if data is not None:
            if len(data) < 2:
                self.Warning.not_enough_inst()
            else:
                domain = data.domain
                cont_vars = [a for a in domain.class_vars + domain.metas +
                             domain.attributes if a.is_continuous]
                cont_data = Table.from_table(Domain(cont_vars), data)
                remover = Remove(Remove.RemoveConstant)
                cont_data = remover(cont_data)
                if remover.attr_results["removed"]:
                    self.Information.removed_cons_feat()
                if len(cont_data.domain.attributes) < 2:
                    self.Warning.not_enough_vars()
                else:
                    self.cont_data = SklImpute()(cont_data)
            self.info.set_input_summary(len(data),
                                        format_summary_details(data))
        else:
            self.info.set_input_summary(self.info.NoInput)
        self.set_feature_model()
        self.openContext(self.cont_data)
        self.apply()

    def set_feature_model(self):
        self.feature_model.set_domain(self.cont_data and self.cont_data.domain)
        data = self.data
        if self.cont_data and data.domain.has_continuous_class:
            self.feature = self.cont_data.domain[data.domain.class_var.name]
        else:
            self.feature = None

    def apply(self):
        if self.cont_data is not None:
            self.run()
        else:
            self.commit()

    def run(self):
        data = self.cont_data
        if self.feature is None:
            if self.correlation_type == CorrelationType.PEARSON:
                r, pval = pearsonr(data.X)
            else:
                r, pval = spearmanr(data.X)

        # take upper triangular part of r and pval
        indices = np.triu_indices_from(r, 1)
        r, pval = r[indices], pval[indices]
        sorter = np.argsort(np.abs(r))[::-1]
        vars_ = np.array(data.domain.attributes, dtype=object)
        pairs = np.full((r.size, 2), None, dtype=object)
        start = 0
        for i in range(len(vars_) - 1):
            pairs[start: start + len(vars_) - i - 1, 0] = vars_[i]
            pairs[start: start + len(vars_) - i - 1, 1] = vars_[i + 1:]
            start += len(vars_) - i - 1
        r = r[sorter]
        pval = pval[sorter]
        pairs = pairs[sorter]

        cols = [r, pairs[:, 0], pairs[:, 1]]

        converters = ["{:+.3f}".format, lambda v: v.name, lambda v: v.name]
        converters_tip = [float, lambda v: v.name, lambda v: v.name]

        def display_data(row, col, converters=converters):
            c = converters[col]
            return c(cols[col][row])

        model = TableModelDispatcher(
            shape=(len(pairs), len(cols)),
            data_dispatch={
                Qt.DisplayRole: display_data,
                Qt.ToolTipRole: partial(display_data, converters=converters_tip),
                Qt.UserRole: lambda row, col: cols[col][row],
                gui.TableBarItem.BarRole: lambda row, col: abs(r[row]),
                gui.TableBarItem.BarColorRole:
                    lambda row, col: POSITIVE_COLOR if r[row] > 0 else NEGATIVE_COLOR,
                CorrelationPValRole: lambda row, col: pval[row],
                CorrelationVarPair: lambda row, col: list(pairs[row]),
            },
        )
        self.proxy_model.setSourceModel(model)
        self.model = model

    def commit(self):
        self.Outputs.data.send(self.data)
        summary = len(self.data) if self.data else self.info.NoOutput
        details = format_summary_details(self.data) if self.data else ""
        self.info.set_output_summary(summary, details)

        if self.data is None or self.cont_data is None:
            self.Outputs.features.send(None)
            self.Outputs.correlations.send(None)
            return

        attrs = [ContinuousVariable("Correlation"), ContinuousVariable("FDR")]
        metas = [StringVariable("Feature 1"), StringVariable("Feature 2")]
        domain = Domain(attrs, metas=metas)
        model = self.model
        x = np.array([[float(model.data(model.index(row, 0), role))
                       for role in (Qt.DisplayRole, CorrelationPValRole)]
                      for row in range(model.rowCount())])
        x[:, 1] = FDR(list(x[:, 1]))
        # pylint: disable=protected-access
        m = np.array([[a.name for a in model.data(model.index(row, 0),
                                                  CorrelationVarPair)]
                      for row in range(model.rowCount())], dtype=object)
        corr_table = Table(domain, x, metas=m)
        corr_table.name = "Correlations"

        # data has been imputed; send original attributes
        self.Outputs.features.send(AttributeList(
            [self.data.domain[var.name] for var in self.selection]))
        self.Outputs.correlations.send(corr_table)

    def send_report(self):
        self.report_table(CorrelationType.items()[self.correlation_type],
                          self.rank_table)

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            sel = context.values["selection"]
            context.values["selection"] = [(var.name, vartype(var))
                                           for var in sel[0]]
        if version < 3:
            sel = context.values["selection"]
            context.values["selection"] = ([(name, vtype + 100)
                                            for name, vtype in sel], -3)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCorrelations).run(Table("iris"))
