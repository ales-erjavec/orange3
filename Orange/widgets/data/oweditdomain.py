"""
Edit Domain
-----------

A widget for manual editing of a domain's attributes.

"""
import warnings

from xml.sax.saxutils import escape
from itertools import repeat, chain
from functools import singledispatch, partial
from typing import (
    Tuple, List, Any, Optional, Union, Dict, Sequence, Iterable,
    Callable, TypeVar, Mapping, Hashable,
)

import numpy as np
import pandas as pd
from AnyQt.QtWidgets import (
    QListView, QVBoxLayout, QFormLayout, QLineEdit, QGroupBox,
    QStyledItemDelegate, QStyleOptionViewItem, QSizePolicy, QPushButton,
    QSplitter,
)
from AnyQt.QtCore import Qt, QSize, QModelIndex
from AnyQt.QtCore import pyqtSlot as Slot

import Orange.data

from Orange.preprocess.transformation import Transformation, Identity, Lookup
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import (
    itemmodels, unique_everseen, ftry, datatype, disconnected
)


from Orange.widgets.data.utils.variableeditor import (
    AsCategorical, AsContinuous, AsTime, AsString, StrpTime, Rename,
    MassVariablesEditor, Categorical, CategoricalVector, Real, Time, String,
    RealVector, TimeVector, StringVector, CategoriesMapping, Variable,
    make_dict_mapper, DataVectorTypes, variable_icon, VariableTypes,
    ReinterpretTransformTypes, Annotate, Transform, Unlink, ModifyAnnotations,
    apply_mapping_transform, ftry, categorize_unique, DictMissingConst,
    categorical_from_vector, ReinterpretTransform
)
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output


ndarray = np.ndarray  # pylint: disable=invalid-name
MArray = np.ma.MaskedArray
DType = Union[np.dtype, type]

V = TypeVar("V", bound=Orange.data.Variable)  # pylint: disable=invalid-name
H = TypeVar("H", bound=Hashable)


def deconstruct(obj):
    # type: (datatype) -> Tuple[str, Tuple[Any, ...]]
    """
    Deconstruct a tuple subclass to its class name and its contents.

    Parameters
    ----------
    obj : A tuple

    Returns
    -------
    value: Tuple[str, Tuple[Any, ...]]
    """
    return datatype.deconstruct(obj)


def reconstruct(tname, args, types=None):
    # type: (str, Tuple[Any, ...], Mapping[str, type]) -> Tuple[Any, ...]
    """
    Reconstruct a tuple subclass (inverse of deconstruct).

    Parameters
    ----------
    tname : str
        Type name
    args : Tuple[Any, ...]

    Returns
    -------
    rval: Tuple[Any, ...]
    """
    if types is None:
        types = globals()
    try:
        constructor = types[tname]
    except KeyError:
        raise NameError(tname)
    return constructor(*args)


#: ItemDataRole storing the data vector transform
#: (`List[Union[ReinterpretTransform, Transform]]`)
TransformRole = Qt.UserRole + 42


class VariableEditDelegate(QStyledItemDelegate):
    ReinterpretNames = {
        AsCategorical: "categorical", AsContinuous: "numeric",
        AsString: "string", AsTime: "time"
    }

    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        item = index.data(Qt.EditRole)
        var = tr = None
        if isinstance(item, DataVectorTypes):
            var = item.vtype
            option.icon = variable_icon(var)
        if isinstance(item, VariableTypes):
            var = item
            option.icon = variable_icon(item)
        elif isinstance(item, Orange.data.Variable):
            var = item
            option.icon = gui.attributeIconDict[var]

        transform = index.data(TransformRole)
        if not isinstance(transform, list):
            transform = []

        if transform and isinstance(transform[0], ReinterpretTransformTypes):
            option.icon = variable_icon(transform[0])

        if not option.icon.isNull():
            option.features |= QStyleOptionViewItem.HasDecoration

        if var is not None:
            text = var.name
            for tr in transform:
                if isinstance(tr, Rename):
                    text = ("{} \N{RIGHTWARDS ARROW} {}"
                            .format(var.name, tr.name))
            for tr in transform:
                if isinstance(tr, ReinterpretTransformTypes):
                    text += f" (reinterpreted as " \
                            f"{self.ReinterpretNames[type(tr)]})"
            option.text = text
        if transform:
            # mark as changed (maybe also change color, add text, ...)
            option.font.setItalic(True)


# Item model for edited variables (Variable). Define a display role to be the
# source variable name. This is used only in keyboard search. The display is
# otherwise completely handled by a delegate.
class VariableListModel(itemmodels.PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        # type: (QModelIndex, Qt.ItemDataRole) -> Any
        row = index.row()
        if not index.isValid() or not 0 <= row < self.rowCount():
            return None
        if role == Qt.DisplayRole:
            item = self[row]
            if isinstance(item, VariableTypes):
                return item.name
            if isinstance(item, DataVectorTypes):
                return item.vtype.name
        return super().data(index, role)


class OWEditDomain(widget.OWWidget):
    name = "Edit Domain"
    description = "Rename variables, edit categories and variable annotations."
    icon = "icons/EditDomain.svg"
    priority = 3125
    keywords = ["rename", "drop", "reorder", "order"]

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        data = Output("Data", Orange.data.Table)

    class Error(widget.OWWidget.Error):
        duplicate_var_name = widget.Msg("A variable name is duplicated.")

    settingsHandler = settings.DomainContextHandler()
    settings_version = 2

    _domain_change_store = settings.ContextSetting({})
    _selected_item = settings.ContextSetting(None)  # type: Optional[Tuple[str, int]]
    _merge_dialog_settings = settings.ContextSetting({})
    output_table_name = settings.ContextSetting("")
    _saved_splitter_state: bytes = settings.Setting(b'')

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.data = None  # type: Optional[Orange.data.Table]
        self._invalidated = False

        main = gui.hBox(self.controlArea, spacing=6)
        box = gui.vBox(None, "Variables", objectName="variables-group-box")
        self._edit_splitter = splitter = QSplitter(
            Qt.Horizontal,
            childrenCollapsible=False,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.MinimumExpanding),
            objectName="edit-splitter"
        )
        main.layout().addWidget(splitter)
        splitter.addWidget(box)

        self.variables_model = VariableListModel(parent=self)
        self.variables_view = QListView(
            selectionMode=QListView.ExtendedSelection,
            uniformItemSizes=True,
        )
        self.variables_view.setItemDelegate(VariableEditDelegate(self))
        self.variables_view.setModel(self.variables_model)
        self.variables_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        box.layout().addWidget(self.variables_view)

        box = QGroupBox("Edit", )
        box.setLayout(QVBoxLayout())
        box.layout().setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(box)
        self._editor = MassVariablesEditor()

        box.layout().addWidget(self._editor)

        self.le_output_name = ledit = QLineEdit(text=self.output_table_name)
        self.connect_control("output_table_name", ledit.setText)

        @ledit.textEdited.connect
        def _edited(text):
            if self.output_table_name != text:
                self.output_table_name = text
                self._set_modified(True)

        form = QFormLayout(
            formAlignment=Qt.AlignLeft, labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow
        )
        form.addRow("Output table name", ledit)
        self.buttonsArea.layout().addLayout(form)

        gui.rubber(self.buttonsArea)

        bbox = gui.hBox(self.buttonsArea)
        breset_all = gui.button(
            bbox, self, "Reset All",
            objectName="button-reset-all",
            toolTip="Reset all variables to their input state.",
            autoDefault=False,
            callback=self.reset_all
        )
        breset = gui.button(
            bbox, self, "Reset Selected",
            objectName="button-reset",
            toolTip="Rest selected variable to its input state.",
            autoDefault=False,
            callback=self.reset_selected
        )
        bapply = gui.button(
            bbox, self, "Apply",
            objectName="button-apply",
            toolTip="Apply changes and commit data on output.",
            default=True,
            autoDefault=False,
            callback=self.commit
        )

        self.variables_view.setFocus(Qt.NoFocusReason)  # initial focus

        if self._saved_splitter_state:
            splitter.restoreState(self._saved_splitter_state)
        self.settingsAboutToBePacked.connect(self.__save_state)

    def __save_state(self):
        self._saved_splitter_state = bytes(self._edit_splitter.saveState())

    @Inputs.data
    def set_data(self, data):
        """Set input dataset."""
        self.closeContext()
        self.clear()
        self.data = data

        if self.data is not None:
            self.setup_model(data)
            self.le_output_name.setPlaceholderText(data.name)
            self.openContext(self.data)
            self._restore()
        else:
            self.le_output_name.setPlaceholderText("")

        self.commit()

    def clear(self):
        """Clear the widget state."""
        self.data = None
        self.variables_model.clear()
        self.clear_editor()
        self._selected_item = None
        self._domain_change_store = {}
        self._merge_dialog_settings = {}

    def reset_selected(self):
        """Reset the currently selected variables to their original state."""
        selected = self.variables_view.selectedIndexes()
        model = self.variables_model
        editor = self._editor
        with disconnected(editor.changed,
                          self._on_variable_changed):
            for midx in selected:
                tr = midx.data(TransformRole)
                if not tr:
                    continue  # nothing to reset
                model.setData(midx, [], TransformRole)

            self.open_editor(selected)
            self._invalidate()

    def reset_all(self):
        """Reset all variables to their original state."""
        self._domain_change_store = {}
        if self.data is not None:
            model = self.variables_model
            for i in range(model.rowCount()):
                midx = model.index(i)
                model.setData(midx, [], TransformRole)
            selected = self.variables_view.selectedIndexes()
            self.open_editor(selected)
            self._invalidate()

    def selected_var_index(self):
        """Return the current selected variable index."""
        rows = self.variables_view.selectedIndexes()
        return rows[0].row() if rows else -1

    def setup_model(self, data: Orange.data.Table):
        model = self.variables_model
        vars_ = []
        columns = []
        for i, _, var, coldata in enumerate_columns(data):
            var = abstract(var)
            vars_.append(var)
            if isinstance(var, Categorical):
                data = CategoricalVector(var, coldata)
            elif isinstance(var, Real):
                data = RealVector(var, coldata)
            elif isinstance(var, Time):
                data = TimeVector(var, coldata)
            elif isinstance(var, String):
                data = StringVector(var, coldata)
            columns.append(data)

        model[:] = vars_
        for i, d in enumerate(columns):
            model.setData(model.index(i), d, Qt.EditRole)

    def _restore(self, ):
        """
        Restore the edit transform from saved state.
        """
        model = self.variables_model
        for i in range(model.rowCount()):
            midx = model.index(i, 0)
            coldesc = model.data(midx, Qt.EditRole)  # type: DataVector
            tr = self._restore_transform(coldesc.vtype)
            if tr:
                model.setData(midx, tr, TransformRole)

        # Restore the current variable selection
        i = -1
        if self._selected_item is not None:
            for i, vec in enumerate(model):
                if vec.vtype.name == self._selected_item:
                    break
        if i == -1 and model.rowCount():
            i = 0

        if i != -1:
            itemmodels.select_row(self.variables_view, i)

    def _on_selection_changed(self):
        indexes = self.variables_view.selectionModel().selectedRows()
        self.open_editor(indexes)

    def open_editor(self, indexes: List[QModelIndex]):
        self.clear_editor()
        editdata = [(index.data(Qt.EditRole), index.data(TransformRole) or [])
                    for index in indexes]
        self._editor.setData(editdata)
        self._editor.changed.connect(
            self._on_variable_changed, Qt.UniqueConnection
        )

    def clear_editor(self):
        try:
            self._editor.changed.disconnect(self._on_variable_changed)
        except TypeError:
            pass
        self._editor.clear()

    @Slot()
    def _on_variable_changed(self):
        """User edited the current variable(s) in editor."""
        editor = self._editor
        editdata = editor.data()
        indexes = self.variables_view.selectionModel().selectedIndexes()
        model = self.variables_model
        for index, (var, tr) in zip(indexes, editdata):
            assert index.data(Qt.EditRole) == var
            model.setData(index, tr, TransformRole)
            self._store_transform(var, tr)
        self._invalidate()

    def _store_transform(self, var, transform):
        # type: (Variable, Sequence[Transform]) -> None
        self._domain_change_store[deconstruct(var)] = [deconstruct(t) for t in transform]

    def _restore_transform(self, var):
        # type: (Variable) -> List[Transform]
        tr_ = self._domain_change_store.get(deconstruct(var), [])
        tr = []

        for t in tr_:
            try:
                tr.append(reconstruct(*t))
            except (NameError, TypeError) as err:
                warnings.warn(
                    "Failed to restore transform: {}, {!r}".format(t, err),
                    UserWarning, stacklevel=2
                )
        return tr

    def _invalidate(self):
        self._set_modified(True)

    def _set_modified(self, state):
        self._invalidated = state
        b = self.findChild(QPushButton, "button-apply")
        if isinstance(b, QPushButton):
            f = b.font()
            f.setItalic(state)
            b.setFont(f)

    def commit(self):
        """
        Apply the changes to the input data and send the changed data to output.
        """
        self._set_modified(False)
        self.Error.duplicate_var_name.clear()

        data = self.data
        if data is None:
            self.Outputs.data.send(None)
            return
        model = self.variables_model

        def state(i):
            # type: (int) -> Tuple[DataVector, List[Transform]]
            midx = self.variables_model.index(i, 0)
            return (model.data(midx, Qt.EditRole),
                    model.data(midx, TransformRole))

        state = [state(i) for i in range(model.rowCount())]
        input_vars = data.domain.variables + data.domain.metas
        if self.output_table_name in ("", data.name) \
                and not any(requires_transform(var, trs)
                            for var, (_, trs) in zip(input_vars, state)):
            self.Outputs.data.send(data)
            return

        assert all(v_.vtype.name == v.name
                   for v, (v_, _) in zip(input_vars, state))
        output_vars = []
        unlinked_vars = []
        unlink_domain = False
        for (_, tr), v in zip(state, input_vars):
            if tr:
                var = apply_transform(v, data, tr)
                if requires_unlink(v, tr):
                    unlinked_var = var.copy(compute_value=None)
                    unlink_domain = True
                else:
                    unlinked_var = var
            else:
                unlinked_var = var = v
            output_vars.append(var)
            unlinked_vars.append(unlinked_var)

        if len(output_vars) != len({v.name for v in output_vars}):
            self.Error.duplicate_var_name()
            self.Outputs.data.send(None)
            return

        domain = data.domain
        nx = len(domain.attributes)
        ny = len(domain.class_vars)

        def construct_domain(vars_list):
            # Move non primitive Xs, Ys to metas (if they were changed)
            Xs = [v for v in vars_list[:nx] if v.is_primitive()]
            Ys = [v for v in vars_list[nx: nx + ny] if v.is_primitive()]
            Ms = vars_list[nx + ny:] + \
                 [v for v in vars_list[:nx + ny] if not  v.is_primitive()]
            return Orange.data.Domain(Xs, Ys, Ms)

        domain = construct_domain(output_vars)
        new_data = data.transform(domain)
        if unlink_domain:
            unlinked_domain = construct_domain(unlinked_vars)
            new_data = new_data.from_numpy(
                unlinked_domain,
                new_data.X, new_data.Y, new_data.metas, new_data.W,
                new_data.attributes, new_data.ids
            )
        if self.output_table_name:
            new_data.name = self.output_table_name
        self.Outputs.data.send(new_data)

    def sizeHint(self):
        sh = super().sizeHint()
        return sh.expandedTo(QSize(660, 550))

    def storeSpecificSettings(self):
        """
        Update setting before context closes - also when widget closes.
        """
        # self._merge_dialog_settings = self._editor.saveState()

    def send_report(self):

        if self.data is not None:
            model = self.variables_model
            state = ((model.data(midx, Qt.EditRole),
                      model.data(midx, TransformRole))
                     for i in range(model.rowCount())
                     for midx in [model.index(i)])
            parts = []
            for vector, trs in state:
                if trs:
                    parts.append(report_transform(vector.vtype, trs))
            if parts:
                html = ("<ul>" +
                        "".join(map("<li>{}</li>".format, parts)) +
                        "</ul>")
            else:
                html = "No changes"
            self.report_raw("", html)
        else:
            self.report_data(None)

    @classmethod
    def migrate_context(cls, context, version):
        # pylint: disable=bad-continuation
        if version is None or version <= 1:
            hints_ = context.values.get("domain_change_hints", ({}, -2))[0]
            store = []
            ns = "Orange.data.variable"
            mapping = {
                "DiscreteVariable":
                    lambda name, args, attrs:
                        ("Categorical", (name, tuple(args[0][1]), ())),
                "TimeVariable":
                    lambda name, _, attrs:
                        ("Time", (name, ())),
                "ContinuousVariable":
                    lambda name, _, attrs:
                        ("Real", (name, (3, "f"), ())),
                "StringVariable":
                    lambda name, _, attrs:
                        ("String", (name, ())),
            }
            for (module, class_name, *rest), target in hints_.items():
                if module != ns:
                    continue
                f = mapping.get(class_name)
                if f is None:
                    continue
                trs = []
                key_mapped = f(*rest)
                item_mapped = f(*target[2:])
                src = reconstruct(*key_mapped, globals())   # type: Variable
                dst = reconstruct(*item_mapped, globals())  # type: Variable
                if src.name != dst.name:
                    trs.append(Rename(dst.name))
                if src.annotations != dst.annotations:
                    trs.append(Annotate(dst.annotations))
                if isinstance(src, Categorical):
                    if src.categories != dst.categories:
                        assert len(src.categories) == len(dst.categories)
                        trs.append(CategoriesMapping(
                            list(zip(src.categories, dst.categories))))
                store.append((deconstruct(src), [deconstruct(tr) for tr in trs]))
            context.values["_domain_change_store"] = (dict(store), -2)


def enumerate_columns(
        table: Orange.data.Table
) -> Iterable[Tuple[int, str, Orange.data.Variable, Callable[[], ndarray]]]:
    domain = table.domain
    for i, (var, role) in enumerate(
            chain(zip(domain.attributes, repeat("x")),
                  zip(domain.class_vars, repeat("y")),
                  zip(domain.metas, repeat("m"))),
    ):
        if i >= len(domain.variables):
            i = len(domain.variables) - i - 1
        data = partial(table_column_data, table, i)
        yield i, role, var, data


def table_column_data(
        table: Orange.data.Table,
        var: Union[Orange.data.Variable, int],
        dtype=None
) -> MArray:
    col, copy = table.get_column_view(var)
    var = table.domain[var]  # type: Orange.data.Variable
    if var.is_primitive() and not np.issubdtype(col.dtype, np.inexact):
        col = col.astype(float)
        copy = True

    if dtype is None:
        if isinstance(var, Orange.data.TimeVariable):
            dtype = np.dtype("M8[us]")
            col = col * 1e6
        elif isinstance(var, Orange.data.ContinuousVariable):
            dtype = np.dtype(float)
        elif isinstance(var, Orange.data.DiscreteVariable):
            _values = tuple(var.values)
            _n_values = len(_values)
            dtype = np.dtype(int, metadata={
                "__formatter": lambda i: _values[i] if 0 <= i < _n_values else "?"
            })
        elif isinstance(var, Orange.data.StringVariable):
            dtype = np.dtype(object)
        else:
            assert False
    mask = orange_isna(var, col)

    if dtype != col.dtype:
        col = col.astype(dtype)
        copy = True

    if not copy:
        col = col.copy()
    return MArray(col, mask=mask)


def report_transform(var, trs):
    # type: (Variable, List[Transform]) -> str
    """
    Return a html fragment summarizing the changes applied by `trs` list.

    Parameters
    ----------
    var : Variable
        A variable descriptor no which trs operates
    trs : List[Transform]
        A non empty list of `Transform` instances.

    Returns
    -------
    report : str
    """
    # pylint: disable=too-many-branches
    ReinterpretTypeCode = {
        AsCategorical:  "C", AsContinuous: "N", AsString: "S", AsTime: "T",
    }

    def type_char(value: ReinterpretTransform) -> str:
        return ReinterpretTypeCode.get(type(value), "?")

    def strike(text):
        return "<s>{}</s>".format(escape(text))

    def i(text):
        return "<i>{}</i>".format(escape(text))

    def text(text):
        return "<span>{}</span>".format(escape(text))
    assert trs
    rename = annotate = catmap = unlink = None
    reinterpret = None

    for tr in trs:
        if isinstance(tr, Rename):
            rename = tr
        elif isinstance(tr, Annotate):
            annotate = tr
        elif isinstance(tr, CategoriesMapping):
            catmap = tr
        elif isinstance(tr, Unlink):
            unlink = tr
        elif isinstance(tr, ReinterpretTransformTypes):
            reinterpret = tr

    if reinterpret is not None:
        header = "{} → ({}) {}".format(
            var.name, type_char(reinterpret),
            rename.name if rename is not None else var.name
        )
    elif rename is not None:
        header = "{} → {}".format(var.name, rename.name)
    else:
        header = var.name
    if unlink is not None:
        header += "(unlinked from source)"

    values_section = None
    if catmap is not None:
        values_section = ("Values", [])
        lines = values_section[1]
        for ci, cj in catmap.mapping:
            if ci is None:
                item = cj + ("&nbsp;" * 3) + "(added)"
            elif cj is None:
                item = strike(ci)
            else:
                item = ci + " → " + cj
            lines.append(item)

    annotate_section = None
    if annotate is not None:
        annotate_section = ("Labels", [])
        lines = annotate_section[1]
        old = dict(var.annotations)
        new = dict(annotate.annotations)
        for name in sorted(set(old) - set(new)):
            lines.append(
                "<s>" + i(name) + " : " + text(old[name]) + "</s>"
            )
        for name in sorted(set(new) - set(old)):
            lines.append(
                i(name) + " : " + text(new[name]) + "&nbsp;" * 3 + i("(new)")
            )

        for name in sorted(set(new) & set(old)):
            if new[name] != old[name]:
                lines.append(
                    i(name) + " : " + text(old[name]) + " → " + text(new[name])
                )

    html = ["<div style='font-weight: bold;'>{}</div>".format(header)]
    for title, contents in filter(None, [values_section, annotate_section]):
        section_header = "<div>{}:</div>".format(title)
        section_contents = "<br/>\n".join(contents)
        html.append(section_header)
        html.append(
            "<div style='padding-left: 1em;'>" +
            section_contents +
            "</div>"
        )
    return "\n".join(html)


def abstract(var):
    # type: (Orange.data.Variable) -> Variable
    """
    Return `Varaible` descriptor for an `Orange.data.Variable` instance.

    Parameters
    ----------
    var : Orange.data.Variable

    Returns
    -------
    var : Variable
    """
    annotations = tuple(sorted(
        (key, str(value))
        for key, value in var.attributes.items()
    ))
    linked = var.compute_value is not None
    if isinstance(var, Orange.data.DiscreteVariable):
        return Categorical(var.name, tuple(var.values), annotations, linked)
    elif isinstance(var, Orange.data.TimeVariable):
        return Time(var.name, annotations, linked)
    elif isinstance(var, Orange.data.ContinuousVariable):
        return Real(var.name, (var.number_of_decimals, 'f'), annotations, linked)
    elif isinstance(var, Orange.data.StringVariable):
        return String(var.name, annotations, linked)
    else:
        raise TypeError


def _parse_attributes(mapping):
    # type: (Iterable[Tuple[str, str]]) -> Dict[str, Any]
    # Use the same functionality that parses attributes
    # when reading text files
    return Orange.data.Flags([
        "{}={}".format(*item) for item in mapping
    ]).attributes


def apply_transform(var, table, trs):
    # type: (Orange.data.Variable, Orange.data.Table, List[Transform]) -> Orange.data.Variable
    """
    Apply a list of `Transform` instances on an `Orange.data.Variable`.
    """
    if trs and isinstance(trs[0], ReinterpretTransformTypes):
        reinterpret, trs = trs[0], trs[1:]
        coldata = table_column_data(table, var)
        var = apply_reinterpret(var, reinterpret, coldata)
    if trs:
        return apply_transform_var(var, trs)
    else:
        return var


def requires_unlink(var: Orange.data.Variable, trs: List[Transform]) -> bool:
    # Variable is only unlinked if it has compute_value  or if it has other
    # transformations (that might had added compute_value)
    return trs is not None \
           and any(isinstance(tr, Unlink) for tr in trs) \
           and (var.compute_value is not None or len(trs) > 1)


def requires_transform(var: Orange.data.Variable, trs: List[Transform]) -> bool:
    # Unlink is treated separately: Unlink is required only if the variable
    # has compute_value. Hence tranform is required if it has any
    # transformations other than Unlink, or if unlink is indeed required.
    return trs is not None and (
        not all(isinstance(tr, Unlink) for tr in trs)
        or requires_unlink(var, trs)
    )


@singledispatch
def apply_transform_var(var, trs):
    # type: (Orange.data.Variable, List[Transform]) -> Orange.data.Variable
    raise NotImplementedError


@apply_transform_var.register(Orange.data.DiscreteVariable)
def apply_transform_discete(var, trs):
    # type: (Orange.data.DiscreteVariable, List[Transform]) -> Orange.data.Variable
    # pylint: disable=too-many-branches
    name, annotations = var.name, var.attributes
    mapping = None
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, CategoriesMapping):
            mapping = tr.mapping
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
        elif isinstance(tr, ModifyAnnotations):
            annotations = _parse_attributes(
                apply_mapping_transform(
                    {str(k): str(v) for k, v in annotations.items()},
                    tr.transform).items())

    source_values = var.values
    if mapping is not None:
        dest_values = list(unique_everseen(cj for ci, cj in mapping if cj is not None))
    else:
        dest_values = var.values

    def positions(values):
        rval = {c: i for i, c in enumerate(values)}
        assert len(rval) == len(values)
        return rval
    source_codes = positions(source_values)
    dest_codes = positions(dest_values)
    if mapping is not None:
        # construct a lookup table
        lookup = np.full(len(source_values), np.nan, dtype=float)
        for ci, cj in mapping:
            if ci is not None and cj is not None:
                i, j = source_codes[ci], dest_codes[cj]
                lookup[i] = j
        lookup = Lookup(var, lookup)
    else:
        lookup = Identity(var)
    variable = Orange.data.DiscreteVariable(
        name, values=dest_values, compute_value=lookup
    )
    variable.attributes.update(annotations)
    return variable


@apply_transform_var.register(Orange.data.ContinuousVariable)
def apply_transform_continuous(var, trs):
    # type: (Orange.data.ContinuousVariable, List[Transform]) -> Orange.data.Variable
    name, annotations = var.name, var.attributes
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
        elif isinstance(tr, ModifyAnnotations):
            annotations = _parse_attributes(
                apply_mapping_transform({str(k): str(v) for k, v in annotations.items()}, tr.transform).items())

    variable = Orange.data.ContinuousVariable(
        name=name, compute_value=Identity(var)
    )
    variable.attributes.update(annotations)
    return variable


@apply_transform_var.register(Orange.data.TimeVariable)
def apply_transform_time(var, trs):
    # type: (Orange.data.TimeVariable, List[Transform]) -> Orange.data.Variable
    name, annotations = var.name, var.attributes
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
        elif isinstance(tr, ModifyAnnotations):
            annotations = _parse_attributes(
                apply_mapping_transform(
                    {str(k): str(v) for k, v in annotations.items()},
                    tr.transform).items())
    variable = Orange.data.TimeVariable(
        name=name, have_date=var.have_date, have_time=var.have_time,
        compute_value=Identity(var)
    )
    variable.attributes.update(annotations)
    return variable


@apply_transform_var.register(Orange.data.StringVariable)
def apply_transform_string(var, trs):
    # type: (Orange.data.StringVariable, List[Transform]) -> Orange.data.Variable
    name, annotations = var.name, var.attributes
    out_type = Orange.data.StringVariable
    compute_value = Identity
    for tr in trs:
        if isinstance(tr, Rename):
            name = tr.name
        elif isinstance(tr, Annotate):
            annotations = _parse_attributes(tr.annotations)
        elif isinstance(tr, StrpTime):
            out_type = partial(
                Orange.data.TimeVariable, have_date=tr.have_date, have_time=tr.have_time
            )
            compute_value = partial(ReparseTimeTransform, tr=tr)
        elif isinstance(tr, ModifyAnnotations):
            annotations = _parse_attributes(
                apply_mapping_transform(
                    {str(k): str(v) for k, v in annotations.items()},
                    tr.transform).items())
    variable = out_type(name=name, compute_value=compute_value(var))
    variable.attributes.update(annotations)
    return variable


def time_parse(values: Sequence[str], name="__"):
    tvar = Orange.data.TimeVariable(name)
    parse_time = ftry(tvar.parse, ValueError, np.nan)
    _values = [parse_time(v) for v in values]
    if np.all(np.isnan(_values)):
        # try parsing it with pandas (like in transform)
        dti = pd.to_datetime(values, errors="coerce")
        _values = datetime_to_epoch(dti)
        date_only = getattr(dti, "_is_dates_only", False)
        if np.all(dti != pd.NaT):
            tvar.have_date = True
            tvar.have_time = not date_only
    return tvar, _values


as_string = np.frompyfunc(str, 1, 1)
parse_float = ftry(float, ValueError, float("nan"))

_parse_float = np.frompyfunc(parse_float, 1, 1)


def as_float_or_nan(
        arr: ndarray, out: Optional[ndarray] = None, where: Optional[ndarray] = True,
        dtype=None, **kwargs
) -> ndarray:
    """
    Convert elements of the input array using builtin `float`, fill elements
    where conversion failed with NaN.
    """
    if out is None:
        out = np.full(arr.shape, np.nan, float if dtype is None else dtype)
    if np.issubdtype(arr.dtype, np.inexact) or \
            np.issubdtype(arr.dtype, np.integer):
        np.copyto(out, arr, casting="unsafe", where=where)
        return out
    return _parse_float(arr, out, where=where, casting="unsafe", **kwargs)


def copy_attributes(dst: V, src: Orange.data.Variable) -> V:
    # copy `attributes` and `sparse` members from src to dst
    dst.attributes = dict(src.attributes)
    dst.sparse = src.sparse
    return dst


# Building (and applying) concrete type transformations on Table columns

@singledispatch
def apply_reinterpret(var, tr, data):
    # type: (Orange.data.Variable, ReinterpretTransform, MArray) -> Orange.data.Variable
    """
    Apply a re-interpret transform to an `Orange.data.Table`'s column
    """
    raise NotImplementedError


@apply_reinterpret.register(Orange.data.DiscreteVariable)
def apply_reinterpret_d(var, tr, data):
    # type: (Orange.data.DiscreteVariable, ReinterpretTransform, ndarray) -> Orange.data.Variable
    if isinstance(tr, AsCategorical):
        return var
    elif isinstance(tr, (AsString, AsTime)):
        # TimeVar will be interpreted by StrpTime later
        f = Lookup(var, np.array(var.values, dtype=object), unknown="")
        rvar = Orange.data.StringVariable(name=var.name, compute_value=f)
    elif isinstance(tr, AsContinuous):
        f = Lookup(var, np.array(list(map(parse_float, var.values))),
                   unknown=np.nan)
        rvar = Orange.data.ContinuousVariable(
            name=var.name, compute_value=f, sparse=var.sparse
        )
    else:
        assert False
    return copy_attributes(rvar, var)


@apply_reinterpret.register(Orange.data.ContinuousVariable)
def apply_reinterpret_c(var, tr, data: MArray):
    if isinstance(tr, AsCategorical):
        # This is ill defined and should not result in a 'compute_value'
        # (post-hoc expunge from the domain once translated?)
        values, index = categorize_unique(data)
        coldata = index.astype(float)
        coldata[index.mask] = np.nan
        tr = LookupMappingTransform(
            var, DictMissingConst(
                np.nan, {v: i for i, v in enumerate(values)}
            )
        )
        values = tuple(as_string(values))
        rvar = Orange.data.DiscreteVariable(
            name=var.name, values=values, compute_value=tr
        )
    elif isinstance(tr, AsContinuous):
        return var
    elif isinstance(tr, AsString):
        # TimeVar will be interpreted by StrpTime later
        tstr = ToStringTransform(var)
        rvar = Orange.data.StringVariable(name=var.name, compute_value=tstr)
    elif isinstance(tr, AsTime):
        rvar = Orange.data.TimeVariable(name=var.name, compute_value=Identity(var))
    else:
        assert False
    return copy_attributes(rvar, var)


@apply_reinterpret.register(Orange.data.StringVariable)
def apply_reinterpret_s(var: Orange.data.StringVariable, tr, data: MArray):
    if isinstance(tr, AsCategorical):
        # This is ill defined and should not result in a 'compute_value'
        # (post-hoc expunge from the domain once translated?)
        _, values = categorical_from_vector(data)
        mapping = DictMissingConst(
            np.nan, {v: float(i) for i, v in enumerate(values)}
        )
        tr = LookupMappingTransform(var, mapping)
        rvar = Orange.data.DiscreteVariable(
            name=var.name, values=values, compute_value=tr
        )
    elif isinstance(tr, AsContinuous):
        rvar = Orange.data.ContinuousVariable(
            var.name, compute_value=ToContinuousTransform(var)
        )
    elif isinstance(tr, (AsString, AsTime)):
        # TimeVar will be interpreted by StrpTime later
        return var
    else:
        assert False
    return copy_attributes(rvar, var)


@apply_reinterpret.register(Orange.data.TimeVariable)
def apply_reinterpret_t(var: Orange.data.TimeVariable, tr, data):
    if isinstance(tr, AsCategorical):
        values, _ = categorize_unique(data)
        or_values = values.astype(float) / 1e6
        mapping = DictMissingConst(
            np.nan, {v: i for i, v in enumerate(or_values)}
        )
        tr = LookupMappingTransform(var, mapping)
        values = tuple(as_string(values))
        rvar = Orange.data.DiscreteVariable(
            name=var.name, values=values, compute_value=tr
        )
    elif isinstance(tr, AsContinuous):
        rvar = Orange.data.TimeVariable(
            name=var.name, compute_value=Identity(var)
        )
    elif isinstance(tr, AsString):
        rvar = Orange.data.StringVariable(
            name=var.name, compute_value=ToStringTransform(var)
        )
    elif isinstance(tr, AsTime):
        return var
    else:
        assert False
    return copy_attributes(rvar, var)


def orange_isna(variable: Orange.data.Variable, data: ndarray) -> ndarray:
    """
    Return a bool mask masking N/A elements in `data` for the `variable`.
    """
    if variable.is_primitive():
        return np.isnan(data)
    else:
        return data == variable.Unknown


class ToStringTransform(Transformation):
    """
    Transform a variable to string.
    """
    def transform(self, c):
        if self.variable.is_string:
            return c
        elif self.variable.is_discrete or self.variable.is_time:
            r = column_str_repr(self.variable, c)
        elif self.variable.is_continuous:
            r = as_string(c)
        mask = orange_isna(self.variable, c)
        return np.where(mask, "", r)


class ToContinuousTransform(Transformation):
    def transform(self, c):
        if self.variable.is_time:
            return c
        elif self.variable.is_continuous:
            return c
        elif self.variable.is_discrete:
            lookup = Lookup(
                self.variable, as_float_or_nan(self.variable.values),
                unknown=np.nan
            )
            return lookup.transform(c)
        elif self.variable.is_string:
            return as_float_or_nan(c)
        else:
            raise TypeError


def datetime_to_epoch(dti: pd.DatetimeIndex, only_time) -> np.ndarray:
    """Convert datetime to epoch"""
    delta = dti - (dti.normalize() if only_time else pd.Timestamp("1970-01-01"))
    return (delta / pd.Timedelta("1s")).values


class ReparseTimeTransform(Transformation):
    """
    Re-parse the column's string repr as datetime.
    """
    def __init__(self, variable, tr):
        super().__init__(variable)
        self.tr = tr

    def transform(self, c):
        # if self.formats is none guess format option is selected
        formats = self.tr.formats if self.tr.formats is not None else [None]
        for f in formats:
            d = pd.to_datetime(c, errors="coerce", format=f)
            if pd.notnull(d).any():
                return datetime_to_epoch(d, only_time=not self.tr.have_date)
        return np.nan


class LookupMappingTransform(Transformation):
    """
    Map values via a dictionary lookup.
    """
    def __init__(
            self,
            variable: Orange.data.Variable,
            mapping: Mapping,
            dtype: Optional[np.dtype] = None
    ) -> None:
        super().__init__(variable)
        self.mapping = mapping
        self.dtype = dtype
        self._mapper = make_dict_mapper(mapping, dtype)

    def transform(self, c):
        return self._mapper(c)

    def __reduce_ex__(self, protocol):
        return type(self), (self.variable, self.mapping, self.dtype)

    def __eq__(self, other):
        return self.variable == other.variable \
               and self.mapping == other.mapping \
               and self.dtype == other.dtype

    def __hash__(self):
        return hash((type(self), self.variable, self.mapping, self.dtype))


@singledispatch
def column_str_repr(var: Orange.data.Variable, coldata: ndarray) -> ndarray:
    """Return a array of str representations of coldata for the `variable."""
    _f = np.frompyfunc(var.repr_val, 1, 1)
    return _f(coldata)


@column_str_repr.register(Orange.data.DiscreteVariable)
def column_str_repr_discrete(
        var: Orange.data.DiscreteVariable, coldata: ndarray
) -> ndarray:
    values = np.array(var.values, dtype=object)
    lookup = Lookup(var, values, "?")
    return lookup.transform(coldata)


@column_str_repr.register(Orange.data.StringVariable)
def column_str_repr_string(
        var: Orange.data.StringVariable, coldata: ndarray
) -> ndarray:
    return np.where(coldata == var.Unknown, "?", coldata)


if __name__ == "__main__":  # pragma: no cover
    import os
    WidgetPreview(OWEditDomain).run(
        Orange.data.Table(
            os.path.expanduser("~/Documents/brown-selected-annot.tab"))
    )
