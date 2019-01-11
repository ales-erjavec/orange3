from Orange.widgets.unsupervised.utils.plottools import PlotToolBox, PlotPinchZoomTool
from xml.sax.saxutils import escape

import sys
import enum

from itertools import groupby, chain, repeat
from operator import itemgetter
from types import SimpleNamespace
from collections import OrderedDict

import typing
from typing import (
    List, Tuple, Sequence, Optional, Iterable, Dict, Union, Callable,
)

import numpy as np

from AnyQt.QtWidgets import (
    QListView, QApplication, QComboBox, QGraphicsSceneHelpEvent,
    QToolTip, QGridLayout, QLabel, QCheckBox, QSizePolicy, QStackedWidget,
    QTreeView, QGraphicsObject, QGraphicsScene, QGraphicsView, QSlider,
    QGroupBox, QHBoxLayout
)
from AnyQt.QtGui import (
    QBrush, QColor, QPen, QPalette, QFont, QPainter
)
from AnyQt.QtCore import (
    Qt, QRectF, QLineF, QTimer, QPointF
)
import pyqtgraph as pg

import Orange.data
from Orange.data import (
    Table, Domain, ContinuousVariable, StringVariable, DiscreteVariable
)
from Orange.projection.correspondence import (
    cross_tabulate, burt_table, correspondence, CA, MCA, CATransform
)
from Orange.widgets import widget, gui, settings

from Orange.widgets.utils import (
    itemmodels, colorbrewer, qcolor_alpha, qfont_adjust_size, unique,
    enum_lookup
)
from Orange.widgets.utils.itemmodels import create_list_model, iter_model
from Orange.widgets.utils.spinbox import DoubleSpinBoxWithSlider
from Orange.widgets.utils.combobox import EnumComboBox
from Orange.widgets.utils.graphicsitems import StaticTextItem
from Orange.widgets.visualize.utils.plotutils import (
    HelpEventDelegate, AnchorItem
)
from Orange.widgets.visualize.owscatterplotgraph import (
    ScatterPlotItem, LegendItem as _LegendItem
)
from Orange.widgets.unsupervised.utils import (
    MappedColumnProxyModel, AnalysisRoleView, EnumItemDelegate
)

from Orange.widgets.evaluate.owrocanalysis import once
from Orange.widgets.widget import Input, Output
from Orange.widgets.settings import Setting



ColorSchemes = colorbrewer.colorSchemes["qualitative"]

# TODO: Need to create shapes have the same area.
Shapes = np.array(["o", "s", "t", "p", "h"])


class VariableRole(enum.Enum):
    None_ = "None"
    ActiveRow = "Row Role"
    ActiveColumn = "Column Role"


AnalysisRole = next(gui.OrangeUserRole)


class CAVariableListModel(itemmodels.VariableListModel):
    #: Default item data for specified roles
    DefaultData = {
        AnalysisRole: VariableRole.None_,
    }

    def data(self, index, role=Qt.DisplayRole):
        r = super().data(index, role)
        if r is None and role in self.DefaultData:
            return self.DefaultData[role]
        else:
            return r

    def flags(self, index):
        if index.isValid():
            flags = super().flags(index)
            v = self.data(index, Qt.EditRole)
            if isinstance(v, DiscreteVariable) and len(v.values) < 2:
                # disable the variable in the view if it has insufficient number
                # of values
                flags = flags & ~(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            return flags
        else:
            return super().flags(index)


class VariableRoleDelegate(EnumItemDelegate):
    def displayText(self, value, locale):
        """Reimplemented."""
        if isinstance(value, VariableRole):
            # Use empty text for None_ role
            return "" if value == VariableRole.None_ else value.value
        else:
            return super().displayText(value, locale)


class CATypes(enum.Enum):
    CA = "CA"
    MCA = "Multiple CA"


class MapType(enum.Enum):
    Symmetric = "Symmetric"
    RowPrincipal = "Row principal"
    ColPrincipal = "Column principal"


Symmetric, RowPrincipal, ColPrincipal = MapType


class CAData:
    ca = ...        # type: Union[CA, MCA]
    rowitems = ...  # type: Sequence[Tuple[DiscreteVariable, str]]
    colitems = ...  # type: Sequence[Tuple[DiscreteVariable, str]]
    rownames = ...  # type: List[str]
    colnames = ...  # type: List[str]

    def __init__(self, ca, rowitems, colitems, rownames=None, colnames=None):
        self.ca = ca
        self.rowitems = rowitems
        self.colitems = colitems
        fmt = "{}:{}"
        if rownames is None:
            rownames = [fmt.format(v.name, value) for v, value in rowitems]
        if colnames is None:
            colnames = [fmt.format(v.name, value) for v, value in colitems]
        self.rownames = rownames
        self.colnames = colnames


MapTypesItems = [
    {
        Qt.DisplayRole: "Symmetric",
        Qt.ToolTipRole: "Both row and column points are plotted in principal "
                        "coordinates.",
        Qt.UserRole: Symmetric,
    }, {
        Qt.DisplayRole: "Column principal",
        Qt.ToolTipRole: "Column points are in principal coordinates, "
                        "row points in standard coordinates.",
        Qt.UserRole: ColPrincipal,
    }, {
        Qt.DisplayRole: "Row principal",
        Qt.ToolTipRole: "Row points are in principal coordinates, "
                        "column points in standard coordinates.",
        Qt.UserRole: RowPrincipal
    }
]

PointSizeItems = [
    {
        Qt.DisplayRole: "Same",
        Qt.ToolTipRole: "All point are the same size",
        Qt.UserRole: "same",
    }, {
        Qt.DisplayRole: "Mass",
        Qt.ToolTipRole: "The point size corresponds to their mass",
        Qt.UserRole: "mass",
    }, {
        Qt.DisplayRole: "Inertia",
        Qt.ToolTipRole: "The point size corresponds the percentage of "
                        "inertia explained in the shown dimensions",
        Qt.UserRole: "contrib-abs"
    }, {
        Qt.DisplayRole: "Rel. Inertia",
        Qt.ToolTipRole: "The point size corresponds the percentage of "
                        "inertia explained over <b>all</b> dimensions",
        Qt.UserRole: "contrib-rel"
    }
]

PointLabelItems = [
    {
        Qt.DisplayRole: "None",
        Qt.ToolTip: "No labels are displayed",
        Qt.UserRole: "none",
    }, {
        Qt.DisplayRole: "Short",
        Qt.ToolTip: "Only category values are displayed",
        Qt.UserRole: "names-short",
    }, {
        Qt.DisplayRole: "Full",
        Qt.ToolTip: "Both category name and values are displayed",
        Qt.UserRole: "names",
    },
]


# The Madness and the Damage Done
class GraphicsScene(pg.GraphicsScene):
    # Restore the base QGraphicsScene itemAt, items, ...
    # (https://github.com/pyqtgraph/pyqtgraph/pull/801)

    def itemAt(self, *args, **kwargs):
        return QGraphicsScene.itemAt(self, *args, **kwargs)

    def items(self, *args, **kwargs):
        return QGraphicsScene.items(self, *args, **kwargs)

    def selectedItems(self):
        return QGraphicsScene.selectedItems(self)

    def collidingItems(self, *args, **kwargs):
        return QGraphicsScene.collidingItems(self, *args, **kwargs)

    def focusItem(self):
        return QGraphicsScene.focusItem(self)

    def mouseGrabberItem(self):
        return QGraphicsScene.mouseGrabberItem(self)


class GraphicsView(pg.GraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # replace the scene and all its contents
        self.setCentralItem(None)
        scene = self.scene()
        self.sceneObj = GraphicsScene(parent=self)
        self.setScene(self.sceneObj)
        self.setCentralItem(pg.PlotItem())
        scene.clear()
        scene.setParent(None)

    def wheelEvent(self, event):
        QGraphicsView.wheelEvent(self, event)


class OWCorrespondenceAnalysis(widget.OWWidget):
    name = "Correspondence Analysis"
    description = "Correspondence analysis for categorical multivariate data."
    icon = "icons/CorrespondenceAnalysis.svg"
    keywords = ["ca", "mca", "multiple ca"]

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True)
        contingency = Input("Contingency", Orange.data.Table)

    class Outputs:
        coordinates = Output("Coordinates", Table)
        transformed_data = Output("Transformed data", Orange.data.Table)

    settingsHandler = settings.DomainContextHandler()

    selected_var_indices = settings.ContextSetting([])

    auto_commit = Setting(True)
    _ca_setup_selection = settings.ContextSetting({})  # type: Dict[str, str]
    _map_type = settings.Setting("Symmetric")  # type: str
    _ca_type = settings.Setting("CA")          # type: str
    show_as_arrows = settings.Setting(False)   # type: bool
    inertia_limit_row = settings.Setting(1.0)  # type: float
    inertia_limit_col = settings.Setting(1.0)  # type: float
    row_point_size = settings.Setting("same")  # type: str
    col_point_size = settings.Setting("same")  # type: str
    col_point_label = settings.Setting("label")  # type: str
    row_point_label = settings.Setting("label")  # type: str
    opacity = settings.Setting(255)            # type: int
    base_point_size = settings.Setting(12)     # type: int

    graph_name = "plot"

    class Error(widget.OWWidget.Error):
        #: ..
        empty_data = widget.Msg("Empty dataset")
        #: The 'data' input has no categorical columns
        no_disc_vars = widget.Msg("No categorical data")
        #: The rank 0 solution (constant or rank 1 contingency table)
        null_solution = widget.Msg("Zero dimensional solution.")
        #: Lin alg error (SVD convergence,)
        svd_convergence = widget.Msg("SVD Convergence error:\n")

    class Warning(widget.OWWidget.Warning):
        xor_inputs = widget.Msg(
            "Both 'Data' and 'Contingency' are on the input. Using "
            "'Contingency'."
        )

    def __init__(self):
        super().__init__()
        self.data = None
        #: Contingency table from the "set_contingency" input.
        #: If set it overrides the self.data
        self.contingency_table = None
        #: The computed solution
        self.cadata = None  # type: Optional[CAData]

        self.component_x = 0
        self.component_y = 1

        self.ca_type = enum_lookup(CATypes, self._ca_type, CATypes.CA)
        self.map_type = enum_lookup(MapType, self._map_type, MapType.Symmetric)

        # GUI
        self.controlArea.layout().setSpacing(-1)  # reset spacing

        self.ca_type_cb = cb = EnumComboBox(
            toolTip="Select analysis type"
        )
        cb.addItem("Simple correspondence analysis", userData=CATypes.CA)
        cb.addItem("Multiple correspondence analysis", userData=CATypes.MCA)
        cb.setCurrentValue(self.ca_type)
        cb.currentValueChanged.connect(self.set_analysis_type)
        self.controlArea.layout().addWidget(cb)

        self.varlist = CAVariableListModel(parent=self)

        self.active_stack_widget = QStackedWidget()
        self.controlArea.layout().addWidget(self.active_stack_widget)
        # Simple CA row/column selection
        box = gui.vBox(None, "Variables", addSpace=False, addToLayout=False)

        def make_list_view(sourcemodel):
            view = AnalysisRoleView(
                selectionMode=QListView.ExtendedSelection,
                selectionBehavior=QTreeView.SelectRows,
                uniformRowHeights=True,
                rootIsDecorated=False,
                allColumnsShowFocus=True,
                sizePolicy=QSizePolicy(QSizePolicy.Preferred,
                                       QSizePolicy.Ignored),
                editTriggers=QTreeView.EditKeyPressed | QTreeView.DoubleClicked,
                editRole=Qt.EditRole,
            )
            view.setItemDelegateForColumn(1, VariableRoleDelegate(view))
            model = MappedColumnProxyModel(
                parent=self,
                mappedRoles={
                    AnalysisRole: AnalysisRole,
                    Qt.EditRole: AnalysisRole,
                    Qt.DisplayRole: AnalysisRole,
                }
            )
            model.setSourceModel(sourcemodel)
            view.setModel(model)
            statesmodel = create_list_model([
                {Qt.DisplayRole: state.value, Qt.UserRole: state}
                for state in VariableRole
            ])
            statesmodel.setParent(view)
            view.setStateModel(statesmodel)
            view.setColumnWidth(1, view.sizeHintForColumn(1))
            return view

        def changed(tl, br, roles=[]):
            if roles and AnalysisRole not in roles:
                return
            if self.ca_type != CATypes.CA:
                return
            self._ca_var_changed()

        self.varlist.dataChanged.connect(changed)
        self.ca_varview_row = make_list_view(self.varlist)
        box.layout().addWidget(self.ca_varview_row)
        self.ca_varview_row.resizeColumnToContents(1)
        self.active_stack_widget.addWidget(box)

        box = gui.vBox(None, "Variables", addSpace=False, addToLayout=False)
        self.mca_varview = view = QListView(
            selectionMode=QListView.MultiSelection,
            uniformItemSizes=True,
        )
        self.active_stack_widget.addWidget(box)
        self.active_stack_widget.setCurrentIndex(
            0 if self.ca_type == CATypes.CA else 1
        )
        view.setModel(self.varlist)
        view.selectionModel().selectionChanged.connect(self._mca_var_changed)
        box.layout().addWidget(view)

        grid = QGridLayout(objectName="pr-axis-grid", spacing=5)

        frame = QGroupBox("Principal Components")
        # frame.setContentsMargins(0, 0, 0, 0)
        frame.setLayout(grid)
        self.controlArea.layout().addWidget(frame)
        frame.ensurePolished()
        print(frame.getContentsMargins())
        # gui.widgetBox(self.controlArea, "Principal Components",
        #               orientation=grid, addSpace=False)

        self.axis_x_cb = gui.comboBox(
            None, self, "component_x", callback=self._component_changed,
            contentsLength=5, maximumContentsLength=5,
            addToLayout=False
        )
        self.axis_y_cb = gui.comboBox(
            None, self, "component_y", callback=self._component_changed,
            contentsLength=5, maximumContentsLength=5,
            addToLayout=False
        )

        def small_label(*args, **kwargs):
            label = QLabel(*args, **kwargs)
            font = qfont_adjust_size(label.font(), -2)
            label.setFont(font)
            return label

        self.infotext_x = small_label("N/A")
        self.infotext_y = small_label("N/A")
        self.infotext_sum = small_label("N/A")
        self.infotext_total = small_label("Total inertia: N/A")

        sp = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        grid.addWidget(QLabel("Axes"), 0, 1)
        grid.addWidget(QLabel("Explained inertia"), 0, 2)
        grid.addWidget(QLabel("X:", sizePolicy=sp), 1, 0)
        grid.addWidget(self.axis_x_cb, 1, 1)
        grid.addWidget(QLabel("Y:", sizePolicy=sp), 2, 0)
        grid.addWidget(self.axis_y_cb, 2, 1)
        grid.addWidget(self.infotext_x, 1, 2)
        grid.addWidget(self.infotext_y, 2, 2)
        grid.addWidget(small_label("\N{N-ARY SUMMATION}="), 3, 1, Qt.AlignRight)
        grid.addWidget(self.infotext_sum, 3, 2)
        grid.addWidget(self.infotext_total, 3, 0, 1, 2, Qt.AlignLeft)

        grid = QGridLayout(spacing=5)
        frame = QGroupBox("Plot")
        frame.setLayout(grid)
        self.controlArea.layout().addWidget(frame)
        # box = gui.widgetBox(self.controlArea, "Plot", addSpace=False,
        #                     orientation=grid)
        # map type selection combo box
        self.map_type_cb = catype_cb = EnumComboBox()
        catype_model = create_list_model(MapTypesItems)
        catype_model.setParent(catype_cb)
        catype_cb.setModel(catype_model)
        catype_cb.setCurrentValue(self.map_type)
        catype_cb.currentValueChanged.connect(self.set_map_type)

        self.arrow_cb = QCheckBox(
            "Standard coords. as arrows", checked=self.show_as_arrows,
            enabled=self.map_type != MapType.Symmetric
        )
        self.arrow_cb.toggled.connect(self.set_display_arrows)

        MTypeRow = 0
        grid.addWidget(QLabel("Map type:", sizePolicy=sp), MTypeRow, 0)
        grid.addWidget(catype_cb, MTypeRow, 1, 1, 2, Qt.AlignLeft)
        ArrowRow = 1
        grid.addWidget(self.arrow_cb, ArrowRow, 1, 1, 2)
        PSizeRow = 3

        sizesmodel = create_list_model(PointSizeItems)
        sizesmodel.setParent(self)
        self.row_point_size_cb = EnumComboBox(
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            minimumContentsLength=5,
        )
        self.col_point_size_cb = EnumComboBox(
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            minimumContentsLength=5,
        )
        self.row_point_size_cb.setModel(sizesmodel)
        self.row_point_size_cb.setCurrentValue(self.row_point_size)
        self.row_point_size_cb.currentValueChanged.connect(
            self.set_row_point_size_property)

        self.col_point_size_cb.setModel(sizesmodel)
        self.col_point_size_cb.setCurrentValue(self.col_point_size)
        self.col_point_size_cb.currentValueChanged.connect(
            self.set_col_point_size_property)

        labelsmodel = create_list_model(PointLabelItems)
        self.row_point_label_cb = EnumComboBox(
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            minimumContentsLength=5,
        )
        self.row_point_label_cb.setModel(labelsmodel)
        self.row_point_label_cb.currentValueChanged.connect(
            self.set_row_label_property)

        self.col_point_label_cb = EnumComboBox(
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            minimumContentsLength=5,
        )
        self.col_point_label_cb.setModel(labelsmodel)
        self.col_point_label_cb.setCurrentValue(self.col_point_label)
        self.col_point_label_cb.currentValueChanged.connect(
            self.set_col_label_property)

        grid.addWidget(QLabel("Point size:", sizePolicy=sp), PSizeRow, 0)
        grid.addWidget(self.row_point_size_cb, PSizeRow, 1)
        grid.addWidget(self.col_point_size_cb, PSizeRow, 2)
        grid.addWidget(small_label("Rows"), PSizeRow - 1, 1)
        grid.addWidget(small_label("Columns"), PSizeRow - 1, 2)
        self.base_point_size_slider = QSlider(
            orientation=Qt.Horizontal, minimum=3, maximum=16,
            value=self.base_point_size
        )
        self.base_point_size_slider.valueChanged.connect(
            self.set_base_point_size)
        grid.addWidget(self.base_point_size_slider, PSizeRow + 1, 1, 1, 2)

        LabelRow = PSizeRow + 2
        grid.addWidget(QLabel("Label:", ), LabelRow, 0)
        grid.addWidget(self.row_point_label_cb, LabelRow, 1)
        grid.addWidget(self.col_point_label_cb, LabelRow, 2)

        LabelLimitRow = LabelRow + 1
        self.inertia_limit_row_sb = DoubleSpinBoxWithSlider(
            value=self.inertia_limit_row, minimum=0., maximum=25.,
            singleStep=0.1, decimals=1, suffix="%", keyboardTracking=False,
            toolTip="Filter displayed row labels based on percentage of "
                    "explained inertia of the point"
        )
        self.inertia_limit_row_sb.valueChanged[float].connect(
            self.set_inertia_limit_row
        )
        self.inertia_limit_col_sb = DoubleSpinBoxWithSlider(
            value=self.inertia_limit_col, minimum=0., maximum=50.,
            singleStep=0.1, decimals=1, suffix="%", keyboardTracking=False,
            toolTip="Filter displayed column labels based on percentage of "
                    "explained inertia of the point"
        )
        self.inertia_limit_col_sb.valueChanged[float].connect(
            self.set_inertia_limit_col
        )
        grid.addWidget(QLabel("Label limit:"), LabelLimitRow, 0)
        grid.addWidget(self.inertia_limit_row_sb, LabelLimitRow, 1)
        grid.addWidget(self.inertia_limit_col_sb, LabelLimitRow, 2)

        gui.auto_send(self.controlArea, self, "auto_commit")

        gui.rubber(self.controlArea)

        # Setup the plot view
        self.plot = CAPlotItem()
        self.plotview = GraphicsView()
        self.plotview.setRenderHint(QPainter.Antialiasing)
        self.plotview.setAntialiasing(True)
        self.plotview.setCentralItem(self.plot)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setAspectLocked(True)

        axis = self.plot.getAxis("left")
        axis.setGrid(50)
        axis = self.plot.getAxis("bottom")
        axis.setGrid(50)

        self.plot.setRange(QRectF(-1, 1, 2, 2))
        self.mainArea.layout().setContentsMargins(0, 0, 0, 0)
        self.mainArea.layout().addWidget(self.plotview)

        bl = QHBoxLayout()
        bl.setContentsMargins(4, 4, 4, 4)
        toolbox = QGroupBox("Zoom/Select", )
        toolbox.setLayout(bl)
        tb = self.plot.findChild(PlotToolBox)  # type: PlotToolBox
        bl.addWidget(tb.button(PlotToolBox.ZoomIn))
        bl.addWidget(tb.button(PlotToolBox.ZoomOut))
        bl.addWidget(tb.button(PlotToolBox.ZoomReset))
        bl.addSpacing(5)
        bl.addWidget(tb.button(PlotToolBox.SelectTool))
        bl.addWidget(tb.button(PlotToolBox.ZoomTool))
        bl.addWidget(tb.button(PlotToolBox.PanTool))
        bl.addStretch(10)
        toolbox.setLayout(bl)
        self.controlArea.layout().addWidget(toolbox)

        gui.rubber(self.controlArea)

        self.__update_timer = QTimer(self, interval=0, singleShot=True)
        self.__update_timer.timeout.connect(self._update_CA)

    def storeSpecificSettings(self):
        self._sync_settings()
        super().storeSpecificSettings()

    def saveSettings(self):
        self._sync_settings()
        super().saveSettings()

    def _sync_settings(self):
        model = self.varlist
        selection = {}
        for index in iter_model(model):
            state = index.data(AnalysisRole)
            var = index.data(Qt.EditRole)
            if isinstance(state, VariableRole) \
                    and state is not VariableRole.None_ \
                    and isinstance(var, Orange.data.Variable):
                selection[var.name] = state.name
        self._ca_setup_selection = selection
        self._map_type = self.map_type.name
        self._ca_type = self.ca_type.name

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.clear()
        self.Error.clear()

        if data is not None and not len(data):
            self.Error.empty_data()
            data = None

        self.data = data
        if data is not None:
            self.varlist[:] = [var for var in data.domain.variables
                               if var.is_discrete]
            if not len(self.varlist):
                self.Error.no_disc_vars()
                self.data = None
            else:
                self.selected_var_indices = [0, 1][:len(self.varlist)]
                self.component_x, self.component_y = 0, 1
                self.openContext(data)
                self._restore_selection()
        self._invalidate()

    @Inputs.contingency
    def set_contingency(self, table):
        if table is None and self.contingency_table is None:
            return

        self.closeContext()
        self.clear()

        def table_row_labels(table):
            metas = table.domain.metas
            # find a first string meta column
            rownames = None
            for m in metas:
                if m.is_string:
                    rownames = m
            if rownames is not None:
                coldata, _ = table.get_column_view(rownames)
                return coldata
            else:
                return np.array([str(i + 1) for i in range(len(table))],
                                dtype=object)
        if table is not None:
            cdomain = table.domain
            rowlabels = table_row_labels(table)
            variables = [
                SimpleNamespace(
                    name="Rows", values=rowlabels),
                SimpleNamespace(
                    name="Columns", values=[v.name for v in cdomain.attributes]),
            ]
            self.contingency_table = SimpleNamespace(
                data=table,
                rowvars=[variables[0]],
                colvars=[variables[1]],

            )
        else:
            self.contingency_table = None
        self._invalidate()

    def handleNewSignals(self):
        if self.contingency_table is not None and self.data is not None:
            self.Warning.xor_inputs()
            self.ca_type_cb.setEnabled(False)
            self.set_analysis_type(CATypes.CA)
        else:
            self.ca_type_cb.setEnabled(True)
            self.Warning.clear()

        if self.__update_timer.isActive():
            # _updateCA will do a output data commit.
            self._update_CA()
        else:
            # clear outputs
            self.Outputs.transformed_data.send(None)

    def commit(self):
        output_table = None
        if self.ca is not None:
            sel_vars = self.selected_vars()
            if len(sel_vars) == 2:
                rf = np.vstack((self.ca.row_factors, self.ca.col_factors))
            else:
                rf = self.ca.row_factors
            vars_data = [(val.name, var) for val in sel_vars for var in val.values]
            output_table = Table(
                Domain([ContinuousVariable(f"Component {i + 1}")
                        for i in range(rf.shape[1])],
                       metas=[StringVariable("Variable"),
                              StringVariable("Value")]),
                rf, metas=vars_data
            )
        self.Outputs.coordinates.send(output_table)

    def clear(self):
        self.cadata = None
        self.plot.clear()
        self.varlist[:] = []

    def selected_vars(self):
        rows = sorted(
            ind.row() for ind in self.mca_varview.selectionModel().selectedRows())
        return [self.varlist[i] for i in rows]

    def selected_mca_vars(self):
        model = self.varlist
        active = []
        for i in range(model.rowCount()):
            idx = model.index(i, 0)
            var = model.data(idx, Qt.EditRole)
            role = model.data(idx, AnalysisRole)
            if isinstance(var, DiscreteVariable):
                if role == VariableRole.ActiveRow:
                    active.append(var)
        return active

    def selected_ca_vars(self):
        """
        Return the current simple CA variable role selection as a list of
        row variables an column variables.
        """
        rowvars, colvars = [], []
        model = self.varlist
        for i in range(self.varlist.rowCount()):
            idx = model.index(i, 0)
            item = model.data(idx, Qt.EditRole)
            if not isinstance(item, DiscreteVariable):
                continue

            arole = model.data(idx, AnalysisRole)
            if arole == VariableRole.ActiveRow:
                rowvars.append(item)
            elif arole == VariableRole.ActiveColumn:
                colvars.append(item)

        return rowvars, colvars

    def _restore_selection(self):
        def restore(view, indices):
            with itemmodels.signal_blocking(view.selectionModel()):
                itemmodels.select_rows(view, indices)
        if self.selected_var_indices:
            restore(self.mca_varview, self.selected_var_indices)
        elif len(self.varlist):
            restore(self.varlist, [i for i in range(min(2, len(self.varlist)))])
        if len(self.varlist) >= 2:
            model = self.varlist
            ca_setup = self._ca_setup_selection
            if not ca_setup:
                ca_setup = {v.name: r for v, r in zip(self.varlist, ['ActiveRow', 'ActiveColumn'])}
            byname = {v.name: i for i, v in enumerate(self.varlist)}
            for name, role in ca_setup.items():
                if name in byname:
                    model.setData(
                        model.index(byname[name], 0),
                        enum_lookup(VariableRole, role),
                        AnalysisRole,
                    )

    def _p_axes(self):
        # type: () -> Tuple[int, int]
        # return the current selected displayed principal axes
        return self.component_x, self.component_y

    def _mca_var_changed(self):
        self.selected_var_indices = sorted(
            ind.row() for ind in self.mca_varview.selectionModel().selectedRows())
        if self.ca_type == CATypes.MCA:
            self._invalidate()

    def _ca_var_changed(self):
        if self.ca_type == CATypes.CA:
            self._invalidate()

    def _component_changed(self):
        if self.cadata is not None:
            self._setup_plot()
            self._update_info()

    def _invalidate(self):
        """
        Invalidate the current computed results and schedule an update.
        """
        self.__update_timer.start()

    def set_analysis_type(self, ca_type):
        # type: (CATypes) -> None
        """
        Set the current CA analysis type.

        Ie. switch between simple or multiple CA
        """
        if self.ca_type != ca_type:
            idx = self.ca_type_cb.findData(ca_type, Qt.UserRole)
            assert idx != -1
            self.ca_type = ca_type
            self.ca_type_cb.setCurrentIndex(idx)
            self.active_stack_widget.setCurrentIndex(idx)
            self.cadata = None
            self._invalidate()

    def set_map_type(self, mtype):
        # type: (MapType) -> None
        """
        Set displayed map type
        """
        if self.map_type != mtype:
            self.map_type = mtype
            self.map_type_cb.setCurrentValue(mtype)
            self.arrow_cb.setEnabled(mtype != MapType.Symmetric)
            self._setup_plot()

    def set_display_arrows(self, show):
        # type: (bool) -> None
        if self.show_as_arrows != show:
            self.show_as_arrows = show
            self.arrow_cb.setChecked(show)
            if self.map_type != MapType.Symmetric:
                self._setup_plot()

    def set_inertia_limit_row(self, limit):
        """
        Set the label display threshold for row points.

        (in percent of explained inertia units).
        """
        sb = self.inertia_limit_row_sb
        limit = round(limit, sb.decimals())
        if self.inertia_limit_row != limit:
            self.inertia_limit_row = limit
            sb.setValue(limit)
            item = self.plot.rowitem()
            if item is not None:
                item.set_inertia_threshold(limit / 100.0)

    def set_inertia_limit_col(self, limit):
        """
        Set the label display threshold for column points.

        (in percent of explained inertia units).
        """
        sb = self.inertia_limit_col_sb
        limit = round(limit, sb.decimals())
        if self.inertia_limit_col != limit:
            self.inertia_limit_col = limit
            sb.setValue(limit)
            item = self.plot.colitem()
            if item is not None:
                item.set_inertia_threshold(limit / 100)

    def set_row_point_size_property(self, which):
        self.row_point_size = which
        item = self.plot.rowitem()
        if item is not None:
            item.set_size_property(which)

    def set_col_point_size_property(self, which):
        self.col_point_size = which
        item = self.plot.colitem()
        if item is not None:
            item.set_size_property(which)

    def set_base_point_size(self, size):
        if size != self.base_point_size:
            self.base_point_size = size
            for item in [self.plot.colitem(), self.plot.rowitem()]:
                if item is not None:
                    item.set_base_point_size(size)

    def set_row_label_property(self, which):
        # type: (str) -> None
        """
        Set the plot row point labels source property
        """
        if which != self.row_point_label:
            self.row_point_label = which
            item = self.plot.rowitem()
            if item is not None:
                item.set_label_property(which)

    def set_col_label_property(self, which):
        # type: (str) -> None
        if which != self.col_point_label:
            self.col_point_label = which
            item = self.plot.colitem()
            if item is not None:
                item.set_label_property(which)

    def _update_CA(self):
        # Recompute the CA solution based on current settings and set it for
        # display
        self.__update_timer.stop()
        ctable = None
        # type of analysis to perform
        catype = None
        rowitems, colitems = [], []
        rownames = None
        colnames = None

        def expand_var_pairs(vars):
            return [(v, val) for v in vars for val in v.values]

        if self.contingency_table is not None:
            # simple ca on a contingency table input
            ctable = self.contingency_table.data.X
            rowitems = expand_var_pairs(self.contingency_table.rowvars)
            colitems = expand_var_pairs(self.contingency_table.colvars)
            rownames = [value for _, value in rowitems]
            colnames = [value for _, value in colitems]
            catype = CATypes.CA
        elif self.data is not None:
            catype = self.ca_type
            if catype == CATypes.MCA:
                ca_vars = self.selected_vars()
                rowvars = colvars = ca_vars
                _items, ctable = burt_table(self.data, ca_vars)
                rowitems = colitems = _items
            elif catype == CATypes.CA:
                rowvars, colvars = self.selected_ca_vars()
                if rowvars and colvars:
                    ctable = cross_tabulate(self.data, rowvars, colvars)
                    rowitems = expand_var_pairs(rowvars)
                    colitems = expand_var_pairs(colvars)
            else:
                assert False

        if ctable is not None and ctable.size:
            if catype == CATypes.CA:
                ca = correspondence(ctable, )
            else:
                counts = [len(v.values) for v in rowvars]
                # ca = multiple_correspondence(ctable, counts)
                ca = correspondence.mca(ctable, counts)
        else:
            ca = None
        self._set_ca_solution(ca, rowitems=rowitems, colitems=colitems,
                              rownames=rownames, colnames=colnames)
        self.commit()

    def _set_ca_solution(self, ca, rowitems=None, colitems=None,
                         rownames=None, colnames=None):
        self.axis_x_cb.clear()
        self.axis_y_cb.clear()

        if ca is not None and ca.k == 0:
            self.Error.null_solution()
        else:
            self.Error.null_solution.clear()
        if ca is None:
            self.cadata = None
            self.plot.clear()
            return
        else:
            self.cadata = CAData(
                ca=ca,
                rowitems=rowitems, colitems=colitems,
                rownames=rownames, colnames=colnames,
            )
        sv = ca.D
        nd = int(np.sum(sv > np.sqrt(np.finfo(sv.dtype).eps)))
        axlabels = [str(i + 1) for i in range(nd)]
        self.axis_x_cb.addItems(axlabels)
        self.axis_y_cb.addItems(axlabels)
        dim1, dim2 = self.component_x, self.component_y
        if dim1 >= nd:
            dim1 = nd - 1
        if dim2 >= nd:
            dim2 = nd - 1

        if nd > 1 and dim1 == dim2:
            if dim1 > 0:
                dim1 = 0
            else:
                dim2 = 1

        self.component_x = dim1
        self.component_y = dim2

        self.axis_x_cb.setCurrentIndex(self.component_x)
        self.axis_y_cb.setCurrentIndex(self.component_y)
        self._setup_plot()
        self._update_info()
        self.commit()

    def _setup_plot(self):
        self.plot.clear()
        cadata = self.cadata

        if cadata is None or not cadata.ca.k:
            return
        ca = cadata.ca
        map_type = self.map_type
        dim = self._p_axes()  # displayed principal axes
        std_coords_arrows = self.show_as_arrows

        if map_type in (MapType.ColPrincipal, MapType.RowPrincipal) \
                and std_coords_arrows:
            map_type = (
                _MapType.ColPrincipalArrow
                if map_type == MapType.ColPrincipal else
                _MapType.RowPrincipalArrow
            )

        if dim[0] == dim[1]:
            dim = dim[:1]

        if isinstance(ca, MCA):
            self.plot.plotMCA(cadata, dim=dim, maptype=map_type)
        else:
            self.plot.plotCA(cadata, dim=dim, maptype=map_type)

        for item, limit, size, label in zip(
                [self.plot.rowitem(), self.plot.colitem()],
                [self.inertia_limit_row, self.inertia_limit_col],
                [self.row_point_size, self.col_point_size],
                [self.row_point_label, self.col_point_label]):
            if item is not None:
                item.set_inertia_threshold(limit)
                item.set_base_point_size(self.base_point_size)
                item.set_size_property(size)
                item.set_label_property(label)

    def _update_info(self):
        # update the info text labels in the GUI control area
        if self.cadata is None or self.cadata.ca.k == 0:
            self.infotext_x.setText("N/A")
            self.infotext_y.setText("N/A")
            self.infotext_sum.setText("N/A")
            self.infotext_total.setText("Total inertia: N/A")
        else:
            inertia = self.cadata.ca.inertia_e_dim * 100.0
            ax1, ax2 = self._p_axes()

            fmt = "{:.2f}%"
            self.infotext_x.setText(fmt.format(inertia[ax1]))
            if ax1 != ax2:
                self.infotext_y.setText(fmt.format(inertia[ax2]))
                total = sum(inertia[[ax1, ax2]])
            else:
                self.infotext_y.setText("...")
                total = inertia[ax1]
            self.infotext_sum.setText(fmt.format(total))
            self.infotext_total.setText(
                "Total inertia: " +
                ("{:.2f}".format(self.cadata.ca.inertia)).rstrip("0")
            )

    def send_report(self):
        if self.data is None:
            return

        vars = self.selected_vars()
        if not vars:
            return

        items = OrderedDict()
        items["Data instances"] = len(self.data)
        if len(vars) == 1:
            items["Selected variable"] = vars[0]
        else:
            items["Selected variables"] = "{} and {}".format(
                ", ".join(var.name for var in vars[:-1]), vars[-1].name)
        self.report_items(items)
        self.report_plot()

    def commit(self):
        if self.contingency_table is not None:
            return
        cadata, data = self.cadata, self.data

        if cadata is not None and cadata.ca.k > 0:
            ca = cadata.ca
            rowvars = [g for g, _ in group_items(cadata.rowitems)]
            colvars = [g for g, _ in group_items(cadata.colitems)]
            assert all(v in data.domain.variables for v in colvars + rowvars)
            nd = np.sum(ca.D > np.sqrt(np.finfo(ca.D.dtype).eps))
            tvars = CATransform.create_transformed(
                "CA component {}", colvars, ca.col_factors[:, :nd]
            )
            # To add row components or not to add? That is the question.
            tdata = data.transform(
                Orange.data.Domain(tvars, [], data.domain.metas)
            )
            self.Outputs.transformed_data.send(tdata)

    def onDeleteWidget(self):
        self.plot.clear()
        self.data = None
        self.contingency_table = None
        self.__update_timer.stop()
        super().onDeleteWidget()


class GraphicsGroup(pg.GraphicsObject):
    """
    An empty pg.GraphicsObject to be used to groups items.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlag(QGraphicsObject.ItemHasNoContents, True)

    def boundingRect(self):
        return QRectF()

    def paint(self, painter, option, widget=None):
        pass


class LegendItem(_LegendItem):
    class Entry:
        def __init__(self, name, sample=None):
            # type: (str, Optional[ScatterPlotItem]) -> None
            self.name = name
            self.sample = sample

    def addSeparator(self, name=""):
        sample = ScatterPlotItem(
            x=[], y=[], brush=QBrush(Qt.NoBrush), pen=QPen(Qt.NoPen)
        )
        self.addItem(sample, name)


class _MapType(enum.Enum):
    RowPrincipalArrow = "RowPrincipalArrow"
    ColPrincipalArrow = "ColPrincipalArrow"


class CAPlotItem(pg.PlotItem):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # route help events to this item
        self.__helpDelegate = HelpEventDelegate(self.helpEvent, self)
        self.vb.setAspectLocked(True)
        self.vb.grabGesture(Qt.PinchGesture, )
        self.cadata = None  # type: Optional[CAData]
        self._legend = LegendItem()
        self._legend.setParentItem(self.vb)
        self._legend.anchor((1, 0), (1, 0), offset=(-5, 5))
        self._legend.hide()
        self.__rowitem = None  # type: Optional[DepictItem]
        self.__colitem = None  # type: Optional[DepictItem]

        # zoomfit = QAction(
        #     "Fit in view", self, objectName="action-zoom-fit",
        #     shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0),
        #
        # )
        # zoomout = QAction(
        #     "Zoom out", self, objectName="action-zoom-out",
        #     shortcut=QKeySequence.ZoomOut
        # )
        # zoomin = QAction(
        #     "Zoom in tool", self, objectName="action-zoom-in",
        #     shortcut=QKeySequence.ZoomIn
        # )
        # zoomfit.triggered.connect(self.zoomToFit)
        # zoomout.triggered.connect(self.zoomOut)
        # zoomin.triggered.connect(self.zoomIn)
        # grp = QActionGroup(
        #     self, objectName="actiongroup-view-tool", exclusive=True,
        # )
        # zoom_to_rect = QAction(
        #     "Zoom to", self, objectName="action-zoom-to-rect", checkable=True,
        #     shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_1),
        # )
        # pan = QAction(
        #     "Pan view", self, objectName="action-pan-view", checkable=True,
        #     shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_2)
        # )
        #
        # grp.addAction(zoom_to_rect)
        # grp.addAction(pan)
        # pan.setChecked(True)

        toolbox = PlotToolBox(self)
        toolbox.setViewBox(self.vb)
        zoomout = toolbox.standardAction(PlotToolBox.ZoomOut)
        zoomin = toolbox.standardAction(PlotToolBox.ZoomIn)
        zoomfit = toolbox.standardAction(PlotToolBox.ZoomReset)
        pantool = toolbox.standardAction(PlotToolBox.PanTool)
        zoomout.triggered.connect(self.zoomOut)
        zoomin.triggered.connect(self.zoomIn)
        zoomfit.triggered.connect(self.zoomToFit)
        # selecttool = toolbox.standardAction(PlotToolBox.SelectTool)
        self.addActions([zoomfit, zoomout, zoomin, pantool])
        gs = PlotPinchZoomTool(self)
        gs.setViewBox(self.vb)
        if parent is not None:
            self.setParentItem(parent)

    def itemChange(self, change, value):
        """Reimplemented."""
        # Grab help events from the scene.
        if change == QGraphicsObject.ItemSceneChange:
            scene = self.scene()
            if scene is not None:
                scene.removeEventFilter(self.__helpDelegate)
        elif change == QGraphicsObject.ItemSceneHasChanged:
            scene = self.scene()
            if scene is not None:
                scene.installEventFilter(self.__helpDelegate)
        return super().itemChange(change, value)

    def rowitem(self):
        return self.__rowitem

    def colitem(self):
        return self.__colitem

    def clear(self):
        self._legend.clear()
        self._legend.hide()
        self.__rowitem = None
        self.__colitem = None
        super().clear()

    def zoomIn(self):
        vb = self.getViewBox()
        vb.scaleBy((1.25 ** -1, 1.25 ** -1))

    def zoomOut(self):
        vb = self.getViewBox()
        vb.scaleBy((1.25, 1.25))

    def zoomToFit(self):
        self.autoRange()

    def helpEvent(self, event):
        # type: (QGraphicsSceneHelpEvent) -> bool
        """
        Parameters
        ----------
        event : QGraphicsSceneHelpEvent

        Returns
        -------
        handled : bool
            True if the event was handled and should not propagate.
        """
        if not self.sceneBoundingRect().contains(event.scenePos()):
            return False

        vb = self.getViewBox()
        group = vb.childGroup
        # hitarea in scene coordinates (also map through the view?)
        hitarea = QRectF(-5, -5, 10, 10).translated(event.scenePos())
        # ... in object local coordinates
        hitarea = group.mapFromScene(hitarea).boundingRect()
        items = [("Rows", self.__rowitem),
                 ("Columns", self.__colitem)]
        items = [(which, item) for which, item in items if
                 item is not None and
                 item.scpitem is not None]
        hititems = []
        for which, item in items:
            px, py = item.scpitem.getData()
            tx = (hitarea.left() <= px) & (px < hitarea.right())
            ty = (hitarea.top() <= py) & (py < hitarea.bottom())
            test = tx & ty
            idcs = np.flatnonzero(test)
            if idcs.size:
                hititems.append((which, item, idcs))
        if not hititems:
            return False
        # we possibly have two item types (rows and columns)
        # render  as
        # Row(s)
        # ------
        # ...      Mass   Inertia
        # name     0.2       .. %

        parts = []
        header = ["", "Mass", "Inertia"]
        header = [["<th>{}</th>".format(header) for header in header]]
        for which, item, indices in hititems:
            # mass, inertia
            rows = [[escape(item.names[i]), 0,  # item.mass[i],
                     100 * item.inertia_e[i]]
                    for i in indices]
            formats = ["<th align='right'>{}</th>",
                       "<td align='right'>{:.3g}</td>",
                       "<td align='right'>{:.2f}%</td>"]
            rows = [[fmt.format(value) for fmt, value in zip(formats, row)]
                    for row in rows]
            tip = ("<table>\n" +
                   "\n".join("<tr>{}</tr>".format("".join(row))
                             for row in header + rows) +
                   "</table>")
            parts.append((which, tip))
        if len(parts) > 1:
            tooltip = "".join("<h4>{}</h4><span>{}</span>".format(header, contents)
                              for header, contents in parts)
        elif len(parts) == 1:
            tooltip = parts[0][1]
        else:
            tooltip = ""
        if tooltip:
            style = "<style>table { float: right; }; </style>"
            QToolTip.showText(event.screenPos(), style + tooltip, event.widget())
            return True
        else:
            return False

    def _color_scheme(self):
        base = self.palette().color(QPalette.Base)  # type: QColor
        if base.isValid() and base.lightnessF() > 0.4:
            sch = ColorSchemes["Dark2"]
        else:
            sch = ColorSchemes["Set3"]
        sch = sch.copy()
        if 2 not in sch:
            sch[2] = sch[3][:2]
            sch[1] = sch[3][:1]
        return sch

    def plotCA(self, cadata, dim=(0, 1), maptype=Symmetric, *,
               row_color=None, col_color=None):
        # type: (CAData, ...) -> None
        colorscheme = self._color_scheme()
        small_font = qfont_adjust_size(self.font(), -2)
        colors = [QColor(*c) for c in colorscheme[max(colorscheme.keys())]]
        ca = cadata.ca
        if len(dim) == 2 and dim[0] == dim[1]:
            dim = (dim[0],)

        if maptype == Symmetric:
            rowcoords = ca.row_factors
            colcoords = ca.col_factors
        elif maptype in (RowPrincipal, _MapType.RowPrincipalArrow):
            rowcoords = ca.row_factors
            colcoords = ca.col_standard_coordinates
        elif maptype in (ColPrincipal, _MapType.ColPrincipalArrow):
            rowcoords = ca.row_standard_coordinates
            colcoords = ca.col_factors
        else:
            assert False

        row_groups = group_items(cadata.rowitems,)
        col_groups = group_items(cadata.colitems,)
        # row/column point inertias per each dimension
        column_inertia_ = ca.col_inertia_contributions
        row_inertia_ = ca.row_inertia_contributions
        # row/column point inertia in the depicted dimensions
        row_inertia = row_inertia_[:, dim].sum(axis=1)
        col_inertia = column_inertia_[:, dim].sum(axis=1)
        # ratio of inertia explained in the displayed principal dimensions
        row_inertia_e = row_inertia / row_inertia.sum()
        col_inertia_e = col_inertia / col_inertia.sum()

        def contrib_rel(inertias):
            return inertias[:, dim].sum(axis=1) / inertias.sum(axis=1)

        def contrib_abs(inertias, ):
            return inertias[:, dim].sum(axis=1) / (ca.D[list(dim)] ** 2).sum()

        def short_names(items):
            return [v for _, v in items]

        row_size_depict_data = {
            "same": once(lambda: np.full_like(row_inertia_e, 1.0)),
            "mass": once(lambda: ca.rowmass),
            "contrib-rel": once(lambda: contrib_rel(row_inertia_)),
            "contrib-abs": once(lambda: contrib_abs(row_inertia_)),
            "inertia": lambda: row_inertia_e,
            "inertia-relative": once(lambda: ca.row_inertia),
            "names": lambda: cadata.rownames,
            "names-short": once(lambda: [v for _, v in cadata.rowitems])
        }
        col_size_depict_data = {
            "same": once(lambda: np.full_like(col_inertia_e, 1.0)),
            "mass": once(lambda: ca.colmass),
            "contrib-rel": once(lambda: contrib_rel(column_inertia_)),
            "contrib-abs": once(lambda: contrib_abs(column_inertia_)),
            "inertia": lambda: col_inertia_e,
            "inertia-relative": once(lambda: ca.col_inertia),
            "names": lambda: cadata.colnames,
            "names-short": once(lambda: [v for _, v in cadata.colitems])

        }

        symbol_groups = False
        # color by row/column points
        row_group_sizes = [len(items) for _, items in row_groups]
        col_group_sizes = [len(items) for _, items in col_groups]

        if (len(row_groups) > 1 or len(col_groups) > 1) and \
                len(row_groups) + len(col_groups) < len(Shapes):
            row_symbols = tiled_seq_indices(row_group_sizes)
            col_symbols = tiled_seq_indices(col_group_sizes, start=len(row_groups))

            row_symbols = Shapes[row_symbols]
            col_symbols = Shapes[col_symbols]
            symbol_groups = True
        else:
            row_symbols, col_symbols = Shapes[0], Shapes[1]

        if row_color is None:
            row_color = colors[0]
        if col_color is None:
            col_color = colors[1]
        rowcoords_x, rowcoords_y = extract_coords2D(rowcoords, dim)
        rowscpitem = ScatterPlotItem(
            rowcoords_x, rowcoords_y,
            brush=QBrush(row_color), pen=pg.mkPen(qcolor_alpha(row_color, 75)),
            symbol=row_symbols,
        )
        rowscpitem.setObjectName("ca-row-points")
        rowscpitem.setProperty("-items", cadata.rowitems)

        colcoords_x, colcoords_y = extract_coords2D(colcoords, dim)
        colscpitem = ScatterPlotItem(
            colcoords_x, colcoords_y,
            brush=QBrush(col_color), pen=pg.mkPen(qcolor_alpha(col_color, 75)),
            symbol=col_symbols,
        )
        colscpitem.setObjectName("ca-col-points")
        colscpitem.setProperty("-items", cadata.colitems)

        collabels = plot_labels(
            colcoords_x, colcoords_y, cadata.colnames, font=small_font,
        )
        rowlabels = plot_labels(
            rowcoords_x, rowcoords_y, cadata.rownames, font=small_font,
        )

        collabels.setParentItem(colscpitem)
        rowlabels.setParentItem(rowscpitem)

        if maptype == _MapType.RowPrincipalArrow:
            col_arrows = plot_arrows(
                colcoords_x, colcoords_y, labels=cadata.colnames,
                pen=pg.mkPen(col_color), font=small_font
            )
            col_depict_item = col_arrows
        else:
            col_arrows = None
            col_depict_item = colscpitem
        if maptype == _MapType.ColPrincipalArrow:
            row_arrows = plot_arrows(
                rowcoords_x, rowcoords_y, labels=cadata.rownames,
                pen=pg.mkPen(row_color), font=small_font
            )
            row_depict_item = row_arrows
        else:
            row_arrows = None
            row_depict_item = rowscpitem

        rowitem = DepictItem(
            rowscpitem, cadata.rownames, rowlabels, row_arrows, row_inertia_e,
            depict_coords=row_size_depict_data,
        )
        colitem = DepictItem(
            colscpitem, cadata.colnames, collabels, col_arrows, col_inertia_e,
            depict_coords=col_size_depict_data,
        )
        rowitem.set_size_property(None)
        colitem.set_size_property(None)

        # format legend
        if not symbol_groups and len(row_groups) == len(col_groups) == 1:
            # both row/column are single group/variable
            legenditems = [
                LegendItem.Entry(row_groups[0][0].name, rowscpitem),
                LegendItem.Entry(col_groups[0][0].name, colscpitem),
            ]
        elif not symbol_groups:
            legenditems = [
                LegendItem.Entry("Rows", rowscpitem),
                LegendItem.Entry("Columns", colscpitem)
            ]
        else:
            # build a hierarchical legend
            #             [NAME]
            #   (shape)   [name]
            legenditems = [
                LegendItem.Entry("Rows"),
            ] + [
                LegendItem.Entry(
                    grp.name, ScatterPlotItem(brush=QBrush(row_color), symbol=s)
                )
                for (grp, _), s in zip(row_groups, unique(row_symbols))
            ] + [
                LegendItem.Entry("Columns")
            ] + [
                LegendItem.Entry(
                    grp.name, ScatterPlotItem(brush=QBrush(col_color), symbol=s)
                )
                for (grp, _), s in zip(col_groups, unique(col_symbols))
            ]
        for item in legenditems:
            if item.sample is not None:
                self._legend.addItem(item.sample, item.name)
            else:
                self._legend.addSeparator(item.name)
        self._legend.show()

        self.__colitem = colitem
        self.__rowitem = rowitem

        self.addItem(row_depict_item)
        self.addItem(col_depict_item)
        self.autoRange()

        inertia = ca.inertia_of_axis
        inertia_e = inertia / inertia.sum()
        ax = self.getAxis("bottom")
        ax.setLabel("Component {} ({:.1f}%)"
                    .format(dim[0] + 1, 100 * inertia_e[dim[0]]))
        ax = self.getAxis("left")
        if len(dim) == 2 and dim[0] != dim[1]:
            ax.setLabel("Component {} ({:.1f}%)"
                        .format(dim[1] + 1, 100 * inertia_e[dim[1]]))
            ax.show()
        else:
            ax.hide()

    def plotMCA(self, cadata, dim=(0, 1), maptype=Symmetric):
        # type: (CAData, Tuple[int, int], ...) -> None
        if len(dim) == 2 and dim[0] == dim[1]:
            dim = (dim[0],)

        colorscheme = self._color_scheme()
        colors = [QColor(*c) for c in colorscheme[max(colorscheme.keys())]]

        ca = cadata.ca
        groups = group_items(cadata.rowitems)

        small_font = qfont_adjust_size(self.font(), -2)
        if maptype == MapType.Symmetric:
            rowcoords = ca.row_factors
        elif maptype in (MapType.RowPrincipal, _MapType.RowPrincipalArrow):
            rowcoords = ca.row_factors
        elif maptype in (MapType.ColPrincipal, _MapType.ColPrincipalArrow):
            rowcoords = ca.row_standard_coordinates
        else:
            assert False

        inertia_ = ca.row_inertia_contributions
        inertia = inertia_[:, dim].sum(axis=1)
        inertia_e = inertia / inertia.sum()
        sizes = size_ratio_scale(inertia_e)

        if len(groups) < max(colorscheme.keys()):
            indices = tiled_seq_indices((len(items) for _, items in groups))
            color_table = [QColor(*c) for c in colorscheme[len(groups)]]
            brush_table = [QBrush(c) for c in color_table]
            pen_table = [QPen(b.color().darker(120)) for b in brush_table]
            # basecolors = [color_table[i] for i in indices]
            brush = [brush_table[i] for i in indices]
            pen = [pen_table[i] for i in indices]
            # brush = [QBrush(c) for c in basecolors]
            # pen = [QPen(b.color().darker(120)) for b in brush]
        else:
            brush = pg.mkBrush(colors[0])
            pen = pg.mkPen(colors[0])

        rowcoords_x, rowcoords_y = extract_coords2D(rowcoords, dim)
        rowscpitem = ScatterPlotItem(
            rowcoords_x, rowcoords_y,
            brush=brush, pen=pen, size=sizes,
        )
        rowscpitem.setProperty("-items", cadata.rowitems)
        rowlabels = plot_labels(
            rowcoords_x, rowcoords_y, cadata.rownames, font=small_font,
        )
        rowlabels.setParentItem(rowscpitem)
        size_props = {
            "same": once(lambda: np.full_like(inertia, 1.0)),
            "mass": once(lambda: ca.rowmass),
            "contrib-abs": once(lambda: inertia_[:, dim].sum(axis=1) / (ca.inertia_of_axis[list(dim)].sum())),
            "contrib-rel": once(lambda: inertia_[:, dim].sum(axis=1) / inertia_.sum(axis=1)),
            "inertia": lambda: inertia_e,
            "inertia-relative": once(lambda: inertia / ca.row_inertia),
            "names": lambda: cadata.rownames,
            "names-short": lambda: [v for _, v in cadata.rowitems],
        }
        self.__rowitem = DepictItem(
            rowscpitem, cadata.rownames, rowlabels, None, inertia_e,
            depict_coords=size_props
        )
        legenditems = []
        if len(groups) < len(colors):
            for (var, _), color in zip(groups, colors):
                legenditems += [
                    LegendItem.Entry(
                        var.name, ScatterPlotItem(brush=QBrush(color))
                    )
                ]
        for item in legenditems:
            self._legend.addItem(item.sample, item.name)
        self._legend.setVisible(bool(legenditems))

        self.addItem(rowscpitem)

        inertia_e_dim = ca.inertia_e_dim
        ax = self.getAxis("bottom")
        ax.setLabel("Component {} ({:.1f}%)"
                    .format(dim[0] + 1, 100 * inertia_e_dim[dim[0]]))
        ax = self.getAxis("left")
        if len(dim) == 2 and dim[0] != dim[1]:
            ax.setLabel("Component {} ({:.1f}%)"
                        .format(dim[1] + 1, 100 * inertia_e_dim[dim[1]]))
            ax.show()
        else:
            ax.hide()
        self.vb.autoRange()


# NOTE: Rename to DepictionDelegate


class DepictItem:
    scpitem = ...     # type: ScatterPlotItem
    names = ...       # type: List[str]
    labelsitem = ...  # type: LabelGroup
    arrowsitem = ...  # type: ArrowGroup
    inertia_e = ...   # type: np.ndarray
    inertia_e_theshold = 0.00

    def __init__(self, scpitem, names, labelsitem, arrowitem, inertia_e,
                 inertia_e_threshold=0.1,
                 depict_coords={}  # type: Dict[str, Callable[[], np.ndarray]]
    ):
        # type: (ScatterPlotItem, LabelGroup, ArrowGroup, np.ndarray, ...) -> None
        self.scpitem = scpitem
        self.names = names
        self.labelsitem = labelsitem
        self.arrowsitem = arrowitem
        self.inertia_e = inertia_e
        self.inertia_e_theshold = inertia_e_threshold
        self._depict_coords = depict_coords.copy()
        self.base_point_size = 12
        self.size_property_name = None
        self.label_property_name = None

    def set_inertia_threshold(self, limit):
        if self.inertia_e_theshold == limit:
            return
        self.inertia_e_theshold = limit
        if self.labelsitem is not None:
            labelitems = self.labelsitem.items
        else:
            labelitems = []
        mask = self.inertia_e > limit
        count = np.count_nonzero(mask)
        enable_rotate = count < 300

        if self.arrowsitem is not None:
            arrowitems = self.arrowsitem.items
        else:
            arrowitems = []
        if arrowitems:
            assert len(arrowitems) == mask.size
            for item, visible in zip(arrowitems, mask):
                item.setLabelVisible(visible)
                item.setAutoRotateLabel(enable_rotate)
        if labelitems:
            assert len(labelitems) == mask.size
            for item, visible in zip(labelitems, mask):
                item.setVisible(visible)

    def set_base_point_size(self, basesize):
        if basesize != self.base_point_size:
            self.base_point_size = basesize
            self._update_sizes()

    def set_size_property(self, name):
        # type: (Optional[str]) -> None
        """
        Set size property
        """
        if self.size_property_name != name:
            self.size_property_name = name
            self._update_sizes()

    def _update_sizes(self, ):
        data = self.size_data_for_property(self.size_property_name)
        if data is not None:
            sizes = size_ratio_scale(data, base=self.base_point_size)
            self.scpitem.setSize(sizes)
            sizes = sizes + 5
        else:
            self.scpitem.setSize(self.base_point_size)
            sizes = repeat(self.base_point_size + 5)

        if self.arrowsitem is not None:
            for item, size in zip(self.arrowsitem.items, sizes):
                item._arrow.setStyle(headLen=size)
                item.setPen(item.pen())

    def size_data_for_property(self, property):
        thunk = self._depict_coords.get(property, lambda: None)
        return thunk()

    def color_data_for_property(self, property):
        thunk = self._depict_coords.get(property, lambda: None)
        data = thunk()
        if data.dtype.kind == 'i':
            pass

    def set_label_property(self, name):
        if self.label_property_name != name:
            self.label_property_name = name
            self._update_labels()

    def label_data_for_property(self, name):
        thunk = self._depict_coords.get(name, lambda: None)
        return thunk()

    def _update_labels(self):
        if self.labelsitem is None and self.arrowsitem is None:
            return
        data = self.label_data_for_property(self.label_property_name)
        if data is None:
            data = [""] * len(self.labelsitem.items)
        assert len(data) == len(self.labelsitem.items)
        if self.labelsitem is not None:
            for item, data_ in zip(self.labelsitem.items, data):
                item.setText(str(data_))

        if self.arrowsitem is not None:
            for item, data_ in zip(self.arrowsitem.items, data):
                item.setText(str(data_))


def extract_coords2D(data, dim):
    if len(dim) == 2 and dim[0] == dim[1]:
        dim = (dim[0],)
    if len(dim) == 1:
        x = data[:, dim[0]]
        return x, np.zeros_like(x)
    elif len(dim) == 2:
        return data[:, dim[0]], data[:, dim[1]]
    else:
        assert False


def tiled_seq_indices(counts, start=0):
    # type: (Iterable[int], int) -> List[int]
    """
    Create tiled sequential indices
    """
    indices = chain.from_iterable(
        ([i] * count for i, count in enumerate(counts, start))
    )
    return list(indices)


def scale_max(arr):
    return arr / arr.max()


def size_ratio_scale(arr, base=12, min=3):
    return np.round((base - min) * scale_max(arr) + min)


if typing.TYPE_CHECKING:
    K = typing.TypeVar("K")
    V = typing.TypeVar("V")


def group_items(items):
    # type: (Iterable[Tuple[K, V]]) -> List[Tuple[K, List[Tuple[K, V]]]]
    return [(k, list(v)) for k, v in groupby(items, itemgetter(0))]


def plot_arrows(x, y, labels, font=None, pen=None, ):
    # type: (np.ndarray, np.ndarray, List[str], QFont, QPen) -> ArrowGroup
    """
    Parameters
    ----------
    x : (N,) ndarray
    y : (N,) ndarray
    labels : List[str]
    font : QFont
    pen : QPen

    Returns
    -------
    group : ArrowGroup
    """
    if pen is None:
        pen = pg.mkPen(Qt.black, width=1)
    arrows = [
        AnchorItem(
            line=QLineF(0, 0, x, y), pen=pen, font=font, text=label,
        )
        for x, y, label in zip(x, y, labels)
    ]
    return ArrowGroup(items=arrows)


class ArrowGroup(GraphicsGroup):
    """
    A `pg.GraphicsObject` grouping a list of AnchorItems.
    """
    # disable label rotation past this many AnchorItems
    MaximumRotatedLabels = 300

    def __init__(self, parent=None, items=[], **kwargs):
        super().__init__(None, *kwargs)
        self.items = list(items)  # type: List[AnchorItem]
        if len(items) > self.MaximumRotatedLabels:
            pass

        autorotate = len(items) <= self.MaximumRotatedLabels
        for item in self.items:
            item.setAutoRotateLabel(autorotate)
            item.setParentItem(self)

        if parent is not None:
            self.setParentItem(parent)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        # type: (int, float, ...) -> Tuple[float, float]
        """
        dataBounds protocol implementation for pg.ViewBox auto scaling.
        """
        def rect(p1, p2):  # type: (QPointF, QPointF) -> QRectF
            r = QRectF(p1.x(), p1.y(), 0, 0)
            r.setBottomRight(p2)
            return r.normalized()
        brect = QRectF()

        for it in self.items:
            if it.isVisible():
                line = it.line()
                brect |= rect(line.p1(), line.p2())
        if ax == 0:
            return brect.left(), brect.right()
        else:
            return brect.top(), brect.bottom()


def plot_labels(x, y, labels, font=None, color=None):
    # type: (np.ndarray, np.ndarray,  List[str], QFont, QColor) -> LabelGroup
    """
    Parameters
    ----------
    x : (N,) ndarray
    y : (N,) ndarray
    labels : List[str]
    font : Optional[QFont]
    color : Optional[QColor]

    Returns
    -------
    group: LabelGroup
    """
    items = []
    for x, y, text in zip(x, y, labels):
        item = StaticTextItem(
            pos=QPointF(x, y), text=text, font=font, anchor=(0.5, -0.2)
        )
        item.setFlag(QGraphicsObject.ItemIgnoresTransformations)
        if color is not None:
            p = item.palette()
            p.setColor(QPalette.Text, color)
            item.setPalette(p)
        items.append(item)
    return LabelGroup(items=items)


class LabelGroup(GraphicsGroup):
    """
    A `pg.GraphicsObject` grouping a list of StaticTextItems.
    """
    def __init__(self, parent=None, items=[], **kwargs):
        super().__init__(None, **kwargs)
        self.items = list(items)  # type: List[StaticTextItem]
        for item in self.items:
            item.setParentItem(self)

        if parent is not None:
            self.setParentItem(parent)


def main(argv=None):  # pragma: no cover
    if argv is None:
        argv = sys.argv
    app = QApplication(argv)
    argv = app.arguments()
    argv = list(argv)
    filename = None
    contingency = None

    while len(argv) > 1:
        arg = argv.pop(1)
        if arg in ("-c", "--contingency"):
            contingency = True
        elif arg.startswith("-"):
            raise SystemExit("invalid arg: " + arg)
        else:
            filename = arg
            break
    if filename is None:
        filename = "zoo"
    if contingency is None:
        contingency = False

    data = Orange.data.Table(filename)

    w = OWCorrespondenceAnalysis()
    if contingency:
        w.set_contingency(data)
    else:
        w.set_data(data)

    w.handleNewSignals()
    w.show()
    w.raise_()
    rval = app.exec_()
    w.set_data(None)
    w.set_contingency(None)
    w.handleNewSignals()
    w.saveSettings()
    w.onDeleteWidget()
    return rval


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

