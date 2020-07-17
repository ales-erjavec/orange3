"""
Edit Domain
-----------

A widget for manual editing of a domain's attributes.

"""
import warnings
from operator import itemgetter

from types import SimpleNamespace
from xml.sax.saxutils import escape
from itertools import zip_longest, repeat, chain
from contextlib import contextmanager
from collections import namedtuple, Counter
from functools import singledispatch, partial
from typing import (
    Tuple, List, Any, Optional, Union, Dict, Sequence, Iterable, NamedTuple,
    FrozenSet, Type, Callable, TypeVar, Mapping, MutableMapping,
    cast)

import numpy as np
import pandas as pd
from AnyQt.QtWidgets import (
    QWidget, QListView, QTreeView, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QAction, QActionGroup, QGroupBox,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle, QSizePolicy,
    QDialogButtonBox, QPushButton, QCheckBox, QComboBox,
    QDialog, QRadioButton, QGridLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QSplitter
)
from AnyQt.QtGui import QStandardItemModel, QStandardItem, QKeySequence, QIcon, \
    QPalette
from AnyQt.QtCore import (
    Qt, QSize, QModelIndex, QAbstractItemModel,
    QAbstractListModel)
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

import Orange.data

from Orange.preprocess.transformation import Transformation, Identity, Lookup
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels, unique_everseen
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils.buttons import FixedSizeButton
from Orange.widgets.widget import Input, Output
from Orange.widgets.data.utils.bracepattern import expand_pattern, infer_pattern

ndarray = np.ndarray  # pylint: disable=invalid-name
MArray = np.ma.MaskedArray
DType = Union[np.dtype, type]

A = TypeVar("A")  # pylint: disable=invalid-name
B = TypeVar("B")  # pylint: disable=invalid-name
V = TypeVar("V", bound=Orange.data.Variable)  # pylint: disable=invalid-name


class _DataType:
    def __eq__(self, other):
        """Equal if `other` has the same type and all elements compare equal."""
        if type(self) is not type(other):
            return False
        return super().__eq__(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self), super().__hash__()))

#: An ordered sequence of key, value pairs (variable annotations)
AnnotationsType = Tuple[Tuple[str, str], ...]


# Define abstract representation of the variable types edited

class Categorical(
    _DataType, NamedTuple("Categorical", [
        ("name", str),
        ("categories", Tuple[str, ...]),
        ("annotations", AnnotationsType),
        ("linked", bool)
    ])): pass


class Real(
    _DataType, NamedTuple("Real", [
        ("name", str),
        # a precision (int, and a format specifier('f', 'g', or '')
        ("format", Tuple[int, str]),
        ("annotations", AnnotationsType),
        ("linked", bool)
    ])): pass


class String(
    _DataType, NamedTuple("String", [
        ("name", str),
        ("annotations", AnnotationsType),
        ("linked", bool)
    ])): pass


class Time(
    _DataType, NamedTuple("Time", [
        ("name", str),
        ("annotations", AnnotationsType),
        ("linked", bool)
    ])): pass


Variable = Union[Categorical, Real, Time, String]
VariableTypes = (Categorical, Real, Time, String)


# Define variable transformations.

class Rename(_DataType, namedtuple("Rename", ["name"])):
    """
    Rename a variable.

    Parameters
    ----------
    name : str
        The new name
    """
    def __call__(self, var):
        # type: (Variable) -> Variable
        return var._replace(name=self.name)


#: Mapping of categories.
#: A list of pairs with the first element the original value and the second
#: element the new value. If the first element is None then a category level
#: is added. If the second element is None than the corresponding first level
#: is dropped. If there are duplicated elements on the right the corresponding
#: categories on the left are merged.
#: The mapped order is defined by the translated elements after None removal
#: and merges (the first occurrence of a multiplied elements defines its
#: position):
CategoriesMappingType = List[Tuple[Optional[str], Optional[str]]]


class CategoriesMapping(_DataType, namedtuple("CategoriesMapping", ["mapping"])):
    """
    Change categories of a categorical variable.

    Parameters
    ----------
    mapping : CategoriesMappingType
    """
    def __call__(self, var):
        # type: (Categorical) -> Categorical
        cat = tuple(unique_everseen(cj for _, cj in self.mapping if cj is not None))
        return var._replace(categories=cat)


class Annotate(_DataType, namedtuple("Annotate", ["annotations"])):
    """
    Replace variable annotations.
    """
    def __call__(self, var):
        return var._replace(annotations=self.annotations)


class ModifyAnnotations(_DataType, NamedTuple("ModifyAnnotations", [
    ("transform", Sequence['MappingTransform']),
])):
    def __call__(self, var: Variable) -> Variable:
        return var._replace(annotations=tuple(
            apply_mapping_transform(dict(var.annotations), self.transform)
        ))


class Unlink(_DataType, namedtuple("Unlink", [])):
    """Unlink variable from its source, that is, remove compute_value"""


Transform = Union[Rename, CategoriesMapping, Annotate, Unlink]
TransformTypes = (Rename, CategoriesMapping, Annotate, Unlink)

CategoricalTransformTypes = (CategoriesMapping, Unlink)


# Reinterpret vector transformations.
class CategoricalVector(
    _DataType, NamedTuple("CategoricalVector", [
        ("vtype", Categorical),
        ("data", Callable[[], MArray]),
    ])): ...


class RealVector(
    _DataType, NamedTuple("RealVector", [
        ("vtype", Real),
        ("data", Callable[[], MArray]),
    ])): ...


class StringVector(
    _DataType, NamedTuple("StringVector", [
        ("vtype", String),
        ("data", Callable[[], MArray]),
    ])): ...


class TimeVector(
    _DataType, NamedTuple("TimeVector", [
        ("vtype", Time),
        ("data", Callable[[], MArray]),
    ])): ...


DataVector = Union[CategoricalVector, RealVector, StringVector, TimeVector]
DataVectorTypes = (CategoricalVector, RealVector, StringVector, TimeVector)


class AsString(_DataType, NamedTuple("AsString", [])):
    """Reinterpret a data vector as a string."""
    def __call__(self, vector: DataVector) -> StringVector:
        var, _ = vector
        if isinstance(var, String):
            return vector
        return StringVector(
            String(var.name, var.annotations, False),
            lambda: as_string(vector.data()),
        )


class AsContinuous(_DataType, NamedTuple("AsContinuous", [])):
    """
    Reinterpret as a continuous variable (values that do not parse as
    float are NaN).
    """
    def __call__(self, vector: DataVector) -> RealVector:
        var, _ = vector
        if isinstance(var, Real):
            return vector
        elif isinstance(var, Categorical):
            def data() -> MArray:
                d = vector.data()
                a = categorical_to_string_vector(d, var.values)
                return MArray(as_float_or_nan(a, where=a.mask), mask=a.mask)
            return RealVector(
                Real(var.name, (6, 'g'), var.annotations, var.linked), data
            )
        elif isinstance(var, Time):
            return RealVector(
                Real(var.name, (6, 'g'), var.annotations, var.linked),
                lambda: vector.data().astype(float)
            )
        elif isinstance(var, String):
            def data():
                s = vector.data()
                return MArray(as_float_or_nan(s, where=s.mask), mask=s.mask)
            return RealVector(
                Real(var.name, (6, "g"), var.annotations, var.linked), data
            )
        raise AssertionError


class AsCategorical(_DataType, namedtuple("AsCategorical", [])):
    """Reinterpret as a categorical variable"""
    def __call__(self, vector: DataVector) -> CategoricalVector:
        # this is the main complication in type transformation since we need
        # the data and not just the variable description
        var, _ = vector
        if isinstance(var, Categorical):
            return vector
        if isinstance(var, (Real, Time, String)):
            data, values = categorical_from_vector(vector.data())
            return CategoricalVector(
                Categorical(var.name, values, var.annotations, var.linked),
                lambda: data
            )
        raise AssertionError


class AsTime(_DataType, namedtuple("AsTime", [])):
    """Reinterpret as a datetime vector"""
    def __call__(self, vector: DataVector) -> TimeVector:
        var, _ = vector
        if isinstance(var, Time):
            return vector
        elif isinstance(var, Real):
            return TimeVector(
                Time(var.name, var.annotations, var.linked),
                lambda: vector.data().astype("M8[us]")
            )
        elif isinstance(var, Categorical):
            def data():
                d = vector.data()
                s = categorical_to_string_vector(d, var.values)
                dt = pd.to_datetime(s, errors="coerce").values.astype("M8[us]")
                return MArray(dt, mask=d.mask)
            return TimeVector(
                Time(var.name, var.annotations, var.linked), data
            )
        elif isinstance(var, String):
            def data():
                s = vector.data()
                dt = pd.to_datetime(s, errors="coerce").values.astype("M8[us]")
                return MArray(dt, mask=s.mask)
            return TimeVector(
                Time(var.name, var.annotations, var.linked), data
            )
        raise AssertionError


ReinterpretTransform = Union[AsCategorical, AsContinuous, AsTime, AsString]
ReinterpretTransformTypes = (AsCategorical, AsContinuous, AsTime, AsString)


def deconstruct(obj):
    # type: (tuple) -> Tuple[str, Tuple[Any, ...]]
    """
    Deconstruct a tuple subclass to its class name and its contents.

    Parameters
    ----------
    obj : A tuple

    Returns
    -------
    value: Tuple[str, Tuple[Any, ...]]
    """
    cname = type(obj).__name__
    args = tuple(obj)
    return cname, args


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


def formatter_for_dtype(dtype: np.dtype) -> Callable[[Any], str]:
    if dtype.metadata is None:
        return str
    else:
        return dtype.metadata.get("__formatter", str)  # metadata abuse


def masked_unique(data: MArray) -> Tuple[MArray, ndarray]:
    if not np.any(data.mask):
        return np.ma.unique(data, return_inverse=True)
    elif data.dtype.kind == "O":
        # np.ma.unique does not work for object arrays
        # (no ma.minimum_fill_value for object arrays)
        # maybe sorted(set(data.data[...]))
        unq = np.unique(data.data[~data.mask])
        mapper = make_dict_mapper(
            DictMissingConst(len(unq), ((v, i) for i, v in enumerate(unq)))
        )
        index = mapper(data.data)
        unq = np.array(unq.tolist() + [data.fill_value], dtype=data.dtype)
        unq_mask = [False] * unq.size
        unq_mask[-1] = True
        unq = MArray(unq, mask=unq_mask)
        return unq, index
    else:
        unq, index = np.ma.unique(data, return_inverse=True)
        assert not np.any(unq.mask[:-1]), \
            "masked value if present must be in last position"
        return unq, index


def categorical_from_vector(data: MArray) -> Tuple[MArray, Tuple[str, ...]]:
    formatter = formatter_for_dtype(data.dtype)
    unq, index = categorize_unique(data)
    if formatter is not str:
        # str(np.array([0], "M8[s]")[0]) is different then
        # str(np.array([0], "M8[s]").astype(object)[0]) which is what
        # as_string is doing
        names = tuple(map(formatter, unq.astype(object)))
    else:
        names = tuple(as_string(unq))
    data = MArray(
        index, mask=data.mask,
        dtype=np.dtype(int, metadata={
            "__formater": lambda i: names[i] if 0 <= i < unq.size else "?"
        })
    )
    return data, names


def categorize_unique(data: MArray) -> Tuple[ndarray, MArray]:
    unq, index = masked_unique(data)
    if np.any(unq.mask):
        unq = unq[:-1]
        assert not np.any(unq.mask), "masked value if present must be last"
    unq = unq.data
    index[data.mask] = -1
    index = MArray(index, mask=data.mask)
    return unq, index


def categorical_to_string_vector(data: MArray, values: Tuple[str, ...]) -> MArray:
    lookup = np.asarray(values, object)
    out = np.full(data.shape, "", dtype=object)
    mask_ = ~data.mask
    out[mask_] = lookup[data.data[mask_]]
    return MArray(out, mask=data.mask, fill_value="")


class GroupItemsDialog(QDialog):
    """
    A dialog for group less frequent values.
    """
    DEFAULT_LABEL = "other"

    def __init__(
            self, variable: Categorical,
            data: Union[np.ndarray, List, MArray],
            selected_attributes: List[str], dialog_settings: Dict[str, Any],
            parent: QWidget = None, flags: Qt.WindowFlags = Qt.Dialog, **kwargs
    ) -> None:
        super().__init__(parent, flags, **kwargs)
        self.variable = variable
        self.data = data
        self.selected_attributes = selected_attributes

        # grouping strategy
        self.selected_radio = radio1 = QRadioButton("Group selected values")
        self.frequent_abs_radio = radio2 = QRadioButton(
            "Group values with less than"
        )
        self.frequent_rel_radio = radio3 = QRadioButton(
            "Group values with less than"
        )
        self.n_values_radio = radio4 = QRadioButton(
            "Group all except"
        )

        # if selected attributes available check the first radio button,
        # otherwise disable it
        if selected_attributes:
            radio1.setChecked(True)
        else:
            radio1.setEnabled(False)
            # they are remembered by number since radio button instance is
            # new object for each dialog
            checked = dialog_settings.get("selected_radio", 0)
            [radio2, radio3, radio4][checked].setChecked(True)

        label2 = QLabel("occurrences")
        label3 = QLabel("occurrences")
        label4 = QLabel("most frequent values")

        self.frequent_abs_spin = spin2 = QSpinBox()
        max_val = len(data)
        spin2.setMinimum(1)
        spin2.setMaximum(max_val)
        spin2.setValue(dialog_settings.get("frequent_abs_spin", 10))
        spin2.setMinimumWidth(
            self.fontMetrics().width("X") * (len(str(max_val)) + 1) + 20
        )
        spin2.valueChanged.connect(self._frequent_abs_spin_changed)

        self.frequent_rel_spin = spin3 = QDoubleSpinBox()
        spin3.setMinimum(0)
        spin3.setDecimals(1)
        spin3.setSingleStep(0.1)
        spin3.setMaximum(100)
        spin3.setValue(dialog_settings.get("frequent_rel_spin", 10))
        spin3.setMinimumWidth(self.fontMetrics().width("X") * (2 + 1) + 20)
        spin3.setSuffix(" %")
        spin3.valueChanged.connect(self._frequent_rel_spin_changed)

        self.n_values_spin = spin4 = QSpinBox()
        spin4.setMinimum(0)
        spin4.setMaximum(len(variable.categories))
        spin4.setValue(
            dialog_settings.get(
                "n_values_spin", min(10, len(variable.categories))
            )
        )
        spin4.setMinimumWidth(
            self.fontMetrics().width("X") * (len(str(max_val)) + 1) + 20
        )
        spin4.valueChanged.connect(self._n_values_spin_spin_changed)

        grid_layout = QGridLayout()
        # first row
        grid_layout.addWidget(radio1, 0, 0, 1, 2)
        # second row
        grid_layout.addWidget(radio2, 1, 0, 1, 2)
        grid_layout.addWidget(spin2, 1, 2)
        grid_layout.addWidget(label2, 1, 3)
        # third row
        grid_layout.addWidget(radio3, 2, 0, 1, 2)
        grid_layout.addWidget(spin3, 2, 2)
        grid_layout.addWidget(label3, 2, 3)
        # fourth row
        grid_layout.addWidget(radio4, 3, 0)
        grid_layout.addWidget(spin4, 3, 1)
        grid_layout.addWidget(label4, 3, 2, 1, 2)

        group_box = QGroupBox()
        group_box.setLayout(grid_layout)

        # grouped variable name
        new_name_label = QLabel("New value name: ")
        self.new_name_line_edit = n_line_edit = QLineEdit(
            dialog_settings.get("name_line_edit", self.DEFAULT_LABEL)
        )
        # it is shown gray when user removes the text and let user know that
        # word others is default one
        n_line_edit.setPlaceholderText(self.DEFAULT_LABEL)
        name_hlayout = QHBoxLayout()
        name_hlayout.addWidget(new_name_label)
        name_hlayout.addWidget(n_line_edit)

        # confirm_button = QPushButton("Apply")
        # cancel_button = QPushButton("Cancel")
        buttons = QDialogButtonBox(
            orientation=Qt.Horizontal,
            standardButtons=(QDialogButtonBox.Ok | QDialogButtonBox.Cancel),
            objectName="dialog-button-box",
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # join components
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(group_box)
        self.layout().addLayout(name_hlayout)
        self.layout().addWidget(buttons)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _frequent_abs_spin_changed(self) -> None:
        self.frequent_abs_radio.setChecked(True)

    def _n_values_spin_spin_changed(self) -> None:
        self.n_values_radio.setChecked(True)

    def _frequent_rel_spin_changed(self) -> None:
        self.frequent_rel_radio.setChecked(True)

    def get_merge_attributes(self) -> List[str]:
        """
        Returns attributes that will be merged

        Returns
        -------
        List of attributes' to be merged names
        """
        if self.selected_radio.isChecked():
            return self.selected_attributes

        if isinstance(self.data, MArray):
            non_nan = self.data[~self.data.mask]
        elif isinstance(self.data, np.ndarray):
            non_nan = self.data[~np.isnan(self.data)]
        else:  # list
            non_nan = [x for x in self.data if x is not None]

        counts = Counter(non_nan)
        if self.n_values_radio.isChecked():
            keep_values = self.n_values_spin.value()
            values = counts.most_common()[keep_values:]
            indices = [i for i, _ in values]
        elif self.frequent_abs_radio.isChecked():
            indices = [v for v, c in counts.most_common()
                       if c < self.frequent_abs_spin.value()]
        else:  # self.frequent_rel_radio.isChecked():
            n_all = sum(counts.values())
            indices = [v for v, c in counts.most_common()
                       if c / n_all * 100 < self.frequent_rel_spin.value()]

        indices = np.array(indices, dtype=int)  # indices must be ints
        return np.array(self.variable.categories)[indices].tolist()

    def get_merged_value_name(self) -> str:
        """
        Returns
        -------
        New label of merged values
        """
        return self.new_name_line_edit.text() or self.DEFAULT_LABEL

    def get_dialog_settings(self) -> Dict[str, Any]:
        """
        Returns
        -------
        Return the dictionary with vlues set by user in each of the line edits
        and selected radio button.
        """
        settings_dict = {
            "frequent_abs_spin": self.frequent_abs_spin.value(),
            "frequent_rel_spin": self.frequent_rel_spin.value(),
            "n_values_spin": self.n_values_spin.value(),
            "name_line_edit": self.new_name_line_edit.text()
        }
        checked = [
            i for i, s in enumerate(
                [self.frequent_abs_radio,
                 self.frequent_rel_radio,
                 self.n_values_radio]
            ) if s.isChecked()]
        # when checked empty radio button for selected values is selected
        # it is not stored in setting since its selection depends on users
        # selection of values in list
        if checked:
            settings_dict["selected_radio"] = checked[0]
        return settings_dict


@contextmanager
def disconnected(signal, slot, connection_type=Qt.AutoConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, connection_type)


#: In 'reordable' models holds the original position of the item
#: (if applicable).
SourcePosRole = Qt.UserRole
#: The original name
SourceNameRole = Qt.UserRole + 2

#: The added/dropped state (type is ItemEditState)
EditStateRole = Qt.UserRole + 1


class ItemEditState:
    NoState = 0
    Dropped = 1
    Added = 2


#: Role used to retrieve the count of 'key' values in the model.
MultiplicityRole = Qt.UserRole + 0x67


class CountedListModel(itemmodels.PyListModel):
    """
    A list model counting how many times unique `key` values appear in
    the list.

    The counts are cached and invalidated on any change to the model involving
    the changes to `keyRoles`.
    """
    #: cached counts
    __counts_cache = None  # type: Optional[Counter]

    def data(self, index, role=Qt.DisplayRole):
        # type: (QModelIndex, int) -> Any
        if role == MultiplicityRole:
            key = self.key(index)
            counts = self.__counts()
            return counts.get(key, 1)
        return super().data(index, role)

    def setData(self, index, value, role=Qt.EditRole):
        # type: (QModelIndex, Any, int)-> bool
        rval = super().setData(index, value, role)
        if role in self.keyRoles():
            self.invalidateCounts()
        return rval

    def setItemData(self, index, data):
        # type: (QModelIndex, Dict[int, Any]) -> bool
        rval = super().setItemData(index, data)
        if self.keyRoles().intersection(set(data.keys())):
            self.invalidateCounts()
        return rval

    def endInsertRows(self):
        super().endInsertRows()
        self.invalidateCounts()

    def endRemoveRows(self):
        super().endRemoveRows()
        self.invalidateCounts()

    def endResetModel(self) -> None:
        super().endResetModel()
        self.invalidateCounts()

    def invalidateCounts(self) -> None:
        """
        Invalidate the cached counts.
        """
        self.__counts_cache = None
        # emit the change for the whole model
        self.dataChanged.emit(
            self.index(0), self.index(self.rowCount() - 1), [MultiplicityRole]
        )

    def __counts(self):
        # type: () -> Counter
        if self.__counts_cache is not None:
            return self.__counts_cache
        counts = Counter()
        for index in map(self.index, range(self.rowCount())):
            key = self.key(index)
            try:
                counts[key] += 1
            except TypeError:  # pragma: no cover
                warnings.warn(f"key value '{key}' is not hashable")
        self.__counts_cache = counts
        return self.__counts_cache

    def key(self, index):
        # type: (QModelIndex) -> Any
        """
        Return the 'key' value that is to be counted.

        The default implementation returns Qt.EditRole value for the index

        Parameters
        ----------
        index : QModelIndex
            The model index.

        Returns
        -------
        key : Any
        """
        return self.data(index, Qt.EditRole)

    def keyRoles(self):
        # type: () -> FrozenSet[int]
        """
        Return a set of item roles on which `key` depends.

        The counts are invalidated and recomputed whenever any of the roles in
        this set changes.

        By default the only role returned is Qt.EditRole
        """
        return frozenset({Qt.EditRole})


class CountedStateModel(CountedListModel):
    """
    Count by EditRole (name) and EditStateRole (ItemEditState)
    """
    # The purpose is to count the items with target name only for
    # ItemEditState.NoRole, i.e. excluding added/dropped values.
    #
    def key(self, index):  # type: (QModelIndex) -> Tuple[Any, Any]
        # reimplemented
        return self.data(index, Qt.EditRole), self.data(index, EditStateRole)

    def keyRoles(self):  # type: () -> FrozenSet[int]
        # reimplemented
        return frozenset({Qt.EditRole, EditStateRole})


class CategoriesEditDelegate(QStyledItemDelegate):
    """
    Display delegate for editing categories.

    Displayed items are styled for add, remove, merge and rename operations.
    """
    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex)-> None
        super().initStyleOption(option, index)
        text = str(index.data(Qt.EditRole))
        sourcename = str(index.data(SourceNameRole))
        editstate = index.data(EditStateRole)
        counts = index.data(MultiplicityRole)
        if not isinstance(counts, int):
            counts = 1
        suffix = None
        if editstate == ItemEditState.Dropped:
            option.state &= ~QStyle.State_Enabled
            option.font.setStrikeOut(True)
            text = sourcename
            suffix = "(dropped)"
        elif editstate == ItemEditState.Added:
            suffix = "(added)"
        else:
            text = f"{sourcename} \N{RIGHTWARDS ARROW} {text}"
            if counts > 1:
                suffix = "(merged)"
        if suffix is not None:
            text = text + " " + suffix
        option.text = text


class CategoriesEditor(QWidget):
    changed = Signal()
    edited = Signal()

    _data: Optional[CategoricalVector] = None
    _transform: Optional[CategoriesMapping] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QGridLayout(margin=0, spacing=1)
        self.merge_dialog_settings = {}
        self.setLayout(layout)
        self.categories_edit = QListView(
            editTriggers=QListView.DoubleClicked | QListView.EditKeyPressed,
            selectionMode=QListView.ExtendedSelection,
            uniformItemSizes=True,
        )
        self.categories_edit.setItemDelegate(CategoriesEditDelegate(self))
        self.categories_model = CountedStateModel(
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        )
        self.categories_edit.setModel(self.categories_model)
        self.categories_model.dataChanged.connect(self.changed)

        self.categories_edit.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self.categories_model.layoutChanged.connect(self._on_selection_changed)
        self.categories_model.rowsMoved.connect(self._on_selection_changed)

        layout.addWidget(self.categories_edit, 0, 0, 1, 2)
        hlayout = QHBoxLayout(spacing=1, margin=0)
        layout.addLayout(hlayout, 1, 0, 1, 2)

        self.categories_action_group = group = QActionGroup(
            self, objectName="action-group-categories", enabled=False
        )
        self.move_value_up = QAction(
            "\N{UPWARDS ARROW}", group,
            toolTip="Move the selected item up.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.AltModifier |
                                  Qt.Key_BracketLeft),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.move_value_up.triggered.connect(self._move_up)

        self.move_value_down = QAction(
            "\N{DOWNWARDS ARROW}", group,
            toolTip="Move the selected item down.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.AltModifier |
                                  Qt.Key_BracketRight),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.move_value_down.triggered.connect(self._move_down)

        self.add_new_item = QAction(
            "+", group,
            objectName="action-add-item",
            toolTip="Append a new item.",
            shortcut=QKeySequence(QKeySequence.New),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.remove_item = QAction(
            "\N{MINUS SIGN}", group,
            objectName="action-remove-item",
            toolTip="Delete the selected item.",
            shortcut=QKeySequence(QKeySequence.Delete),
            shortcutContext=Qt.WidgetShortcut,
        )
        self.merge_items = QAction(
            "M", group,
            objectName="action-merge-item",
            toolTip="Merge selected items.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_Equal),
            shortcutContext=Qt.WidgetShortcut
        )

        self.add_new_item.triggered.connect(self._add_category)
        self.remove_item.triggered.connect(self._remove_category)
        self.merge_items.triggered.connect(self._merge_categories)

        button1 = FixedSizeButton(
            self, defaultAction=self.move_value_up,
            accessibleName="Move up"
        )
        button2 = FixedSizeButton(
            self, defaultAction=self.move_value_down,
            accessibleName="Move down"
        )
        button3 = FixedSizeButton(
            self, defaultAction=self.add_new_item,
            accessibleName="Add"
        )
        button4 = FixedSizeButton(
            self, defaultAction=self.remove_item,
            accessibleName="Remove"
        )
        button5 = FixedSizeButton(
            self, defaultAction=self.merge_items,
            accessibleName="Merge",
        )
        self.categories_edit.addActions([
            self.move_value_up, self.move_value_down,
            self.add_new_item, self.remove_item
        ])
        hlayout.addWidget(button1)
        hlayout.addWidget(button2)
        hlayout.addSpacing(3)
        hlayout.addWidget(button3)
        hlayout.addWidget(button4)
        hlayout.addSpacing(3)
        hlayout.addWidget(button5)
        hlayout.addStretch(10)

    @Slot()
    def _on_selection_changed(self):
        rows = self.categories_edit.selectionModel().selectedRows()
        if len(rows) == 1:
            i = rows[0].row()
            self.move_value_up.setEnabled(i != 0)
            self.move_value_down.setEnabled(
                i != self.categories_model.rowCount() - 1)
        else:
            self.move_value_up.setEnabled(False)
            self.move_value_down.setEnabled(False)

    def _move_rows(self, rows, offset):
        if not rows:
            return
        assert len(rows) == 1
        i = rows[0].row()
        if offset > 0:
            offset += 1
        self.categories_model.moveRows(QModelIndex(), i, 1, QModelIndex(), i + offset)
        self.edited.emit()
        self.changed.emit()

    def _move_up(self):
        rows = self.categories_edit.selectionModel().selectedRows()
        self._move_rows(rows, -1)

    def _move_down(self):
        rows = self.categories_edit.selectionModel().selectedRows()
        self._move_rows(rows, 1)

    def _add_category(self):
        """
        Add a new category
        """
        view = self.categories_edit
        model = view.model()
        with disconnected(model.dataChanged, self.changed,
                          Qt.UniqueConnection):
            row = model.rowCount()
            if not model.insertRow(model.rowCount()):
                return
            index = model.index(row, 0)
            model.setItemData(
                index, {
                    Qt.EditRole: "",
                    EditStateRole: ItemEditState.Added
                }
            )
            view.setCurrentIndex(index)
            view.edit(index)
        self.edited.emit()
        self.changed.emit()

    def _remove_category(self):
        """
        Remove the current selected category.

        If the item is an existing category present in the source variable it
        is marked as removed in the view. But if it was added in the set
        transformation it is removed entirely from the model and view.
        """
        view = self.categories_edit
        rows = view.selectionModel().selectedRows(0)
        if not rows:
            return  # pragma: no cover
        for index in rows:
            model = index.model()
            state = index.data(EditStateRole)
            pos = index.data(SourcePosRole)
            if pos is not None and pos >= 0:
                # existing level -> only mark/toggle its dropped state,
                model.setData(
                    index,
                    ItemEditState.Dropped if state != ItemEditState.Dropped
                    else ItemEditState.NoState,
                    EditStateRole)
            elif state == ItemEditState.Added:
                # new level -> remove it
                model.removeRow(index.row())
            else:
                assert False, "invalid state '{}' for {}" \
                    .format(state, index.row())

    def setData(self, data: CategoricalVector, transform: CategoriesMapping):
        self._data = data
        self.setTransform(transform)

    def setTransform(self, transform: CategoriesMapping):
        vtype = self._data.vtype if self._data is not None else None
        if transform is None and vtype is not None:
            transform = CategoriesMapping([(v, v) for v in vtype.categories])

        if self._transform == transform:
            return

        items = []
        if transform is not None:
            ci_index = {c: i for i, c in enumerate(vtype.categories)}
            for ci, cj in transform.mapping:
                if ci is None and cj is not None:
                    # level added
                    item = {
                        Qt.EditRole: cj,
                        EditStateRole: ItemEditState.Added,
                        SourcePosRole: None
                    }
                elif ci is not None and cj is None:
                    # ci level dropped
                    item = {
                        Qt.EditRole: ci,
                        EditStateRole: ItemEditState.Dropped,
                        SourcePosRole: ci_index[ci],
                        SourceNameRole: ci
                    }
                elif ci is not None and cj is not None:
                    # rename or reorder
                    item = {
                        Qt.EditRole: cj,
                        EditStateRole: ItemEditState.NoState,
                        SourcePosRole: ci_index[ci],
                        SourceNameRole: ci
                    }
                else:
                    assert False, "invalid mapping: {!r}".format(transform.mapping)
                items.append(item)
        else:
            items = []
        model = self.categories_model
        with disconnected(model.dataChanged, self.changed):
            model.clear()
            model.insertRows(0, len(items))
            for i, item in enumerate(items):
                model.setItemData(model.index(i, 0), item)
        self.add_new_item.actionGroup().setEnabled(vtype is not None)
        self.changed.emit()

    def clear(self):
        self._data = None
        self._transform = None
        self.categories_model.clear()

    def __categories_mapping(self):
        # type: () -> CategoriesMappingType
        """
        Encode and return the current state as a CategoriesMappingType
        """
        model = self.categories_model
        source = self._data.vtype.categories

        res = []  # type: CategoriesMappingType
        for i in range(model.rowCount()):
            midx = model.index(i, 0)
            category = midx.data(Qt.EditRole)
            source_pos = midx.data(SourcePosRole)  # type: Optional[int]
            if source_pos is not None:
                source_name = source[source_pos]
            else:
                source_name = None
            state = midx.data(EditStateRole)
            if state == ItemEditState.Dropped:
                res.append((source_name, None))
            elif state == ItemEditState.Added:
                res.append((None, category))
            else:
                res.append((source_name, category))
        return res

    def transform(self) -> CategoriesMapping:
        return CategoriesMapping(self.__categories_mapping())

    def _reset_name_merge(self) -> None:
        """Reset renamed and merged variables in the model."""
        view = self.categories_edit
        model = view.model()  # type: QAbstractItemModel
        with disconnected(model.dataChanged, self.changed):
            for index in map(model.index, range(model.rowCount())):
                model.setData(
                    index, model.data(index, SourceNameRole), Qt.EditRole
                )

    def _merge_categories(self) -> None:
        """
        Merge less common categories into one with the dialog for merge
        selection.
        """
        data = self._data
        if data is None:
            return
        # settings =
        view = self.categories_edit
        model = view.model()  # type: QAbstractItemModel
        selected_attributes = [ind.data(Qt.EditRole)
                               for ind in view.selectedIndexes()]
        dlg = GroupItemsDialog(
            data.vtype, data.data(), selected_attributes,
            # self.merge_dialog_settings.get(self.var, {}),
            {},
            self,
            windowTitle="Import Options",
            sizeGripEnabled=True,
        )
        dlg.setWindowModality(Qt.WindowModal)
        status = dlg.exec_()
        dlg.deleteLater()
        self.merge_dialog_settings[data.vtype] = dlg.get_dialog_settings()

        def complete_merge(text, merge_attributes):
            # write the new text for edit role in all rows
            self._reset_name_merge()
            with disconnected(model.dataChanged, self.changed):
                for index in map(model.index, range(model.rowCount())):
                    if index.data(SourceNameRole) in merge_attributes:
                        model.setData(index, text, Qt.EditRole)
            self.changed.emit()
            self.edited.emit()

        if status == QDialog.Accepted:
            complete_merge(
                dlg.get_merged_value_name(), dlg.get_merge_attributes()
            )


SourceKeyRole = SourceNameRole
SourceValueRole = SourceNameRole + 1
EditHintOptionList = Qt.UserRole + 56
ModifiedStateRole = EditStateRole + 1346


# Transforms that operate on a MutableMapping
class DeleteKey(_DataType, NamedTuple("DeleteKey", [
    ("key", str),
])):
    def __call__(self, mapping: MutableMapping) -> MutableMapping:
        try:
            del mapping[self.key]
        except KeyError:
            pass
        return mapping


class AddItem(_DataType, NamedTuple("AddItem", [
    ("key", str),
    ("value", str)
])):
    def __call__(self, mapping: MutableMapping) -> MutableMapping:
        mapping[self.key] = self.value
        return mapping


class RenameKey(_DataType, NamedTuple("RenameKey", [
    ("key", str),
    ("target", str)
])):
    def __call__(self, mapping: MutableMapping) -> MutableMapping:
        try:
            v = mapping[self.key]
        except KeyError:
            return mapping
        else:
            mapping[self.target] = v
            del mapping[self.key]
        return mapping


class SetValue(_DataType, NamedTuple("SetValue", [
    ("key", str),
    ("value", str)
])):
    def __call__(self, mapping: MutableMapping) -> MutableMapping:
        mapping[self.key] = self.value
        return mapping


MappingTransform = Union[DeleteKey, AddItem, RenameKey, SetValue]
MappingTransform_ = Union[DeleteKey, AddItem, SetValue]


def mapping_diff(a: Mapping, b: Mapping) -> Sequence[MappingTransform_]:
    res = []
    for ak, av in a.items():
        if ak not in b:
            res.append(DeleteKey(ak))
        elif av != b[ak]:
            res.append(SetValue(ak, b[ak]))
    for bk, bv in b.items():
        if bk not in a:
            res.append(AddItem(bk, b[bk]))
    return res


def apply_mapping_transform(mapping: MutableMapping, tr: Sequence[MappingTransform]):
    mapping = dict(mapping)
    for t in tr:
        mapping = t(mapping)
    return mapping


def trace_tr_for_key(
        tr: Sequence[MappingTransform], key, value=None
) -> Tuple[Tuple[Optional[str], Optional[str]], Sequence[MappingTransform]]:
    res = []
    kv = (key, value)
    for t in tr:
        if t.key == key:
            res.append(t)
            if isinstance(t, RenameKey):
                key = t.target
                kv = (key, kv[1])
            elif isinstance(t, DeleteKey):
                return (None, None), res
            elif isinstance(t, SetValue):
                kv = (kv[0], t.value)
            elif isinstance(t, AddItem):
                kv = (kv[0], t.value)
    return kv, res


class OptionsEditItemDelegate(QStyledItemDelegate):
    def createEditor(
            self, parent: QWidget, option: QStyleOptionViewItem,
            index: QModelIndex
    ) -> QWidget:
        data = index.data(Qt.EditRole)
        options = index.data(EditHintOptionList)
        if not isinstance(options, Sequence):
            options = None
        else:
            options = [str(opt) for opt in options]
        if options is not None and len(set(options)) > 1:
            w = QComboBox(parent, editable=True)
            w.lineEdit().setPlaceholderText("...")
            w.addItems(options)
            if isinstance(data, str):
                w.setCurrentText(data)
            w.setParent(parent)
            return w
        return super().createEditor(parent, option, index)


class KeyValueEditor(QWidget):
    changed = Signal()
    edited = Signal()

    class KeyEditDelegate(OptionsEditItemDelegate):
        def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex) -> None:
            super().initStyleOption(option, index)
            state = index.data(EditStateRole)
            modified = index.data(ModifiedStateRole) or state != ItemEditState.NoState
            if state == ItemEditState.Added:
                option.text += " (added)"
            elif state == ItemEditState.Dropped:
                option.font.setStrikeOut(True)
            else:
                original_key = index.data(SourceKeyRole)
                if option.text != original_key:
                    option.text = f"{original_key} \N{RIGHTWARDS ARROW} {option.text}"
                modified = True
            option.font.setItalic(modified)
            multi = index.data(MultiplicityRole)  # Is key present in all mappings
            if not multi:
                color = option.palette.color(QPalette.Text)
                color.setAlpha(100)
                option.palette.setColor(QPalette.Text, color)

        def setModelData(self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex) -> None:
            mp = editor.metaObject().userProperty()
            if mp is not None:
                value = mp.read(editor)
                changed = model.data(index, Qt.EditRole) != value
            else:
                changed = True
            super().setModelData(editor, model, index)
            model.setData(index, model.data(index, ModifiedStateRole) or changed,
                          ModifiedStateRole)

    class ValueEditDelegate(OptionsEditItemDelegate):
        def displayText(self, value: Any, locale: 'QLocale') -> str:
            s = super().displayText(value, locale)
            if s:
                return s
            return str(s)

        def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex) -> None:
            super().initStyleOption(option, index)
            state = index.data(EditStateRole)
            modified = index.data(ModifiedStateRole) or state != ItemEditState.NoState
            if state == ItemEditState.Added:
                option.text += " (added)"
            elif state == ItemEditState.Dropped:
                option.font.setStrikeOut(True)
            else:
                source_value = index.data(SourceValueRole)
                if option.text != source_value:
                    option.text = f"{source_value} \N{RIGHTWARDS ARROW} {option.text}"
                modified = True
            option.font.setItalic(modified)
            data = index.data(Qt.EditRole)
            if data is ...:
                option.text = "..."

            multi = index.data(MultiplicityRole)  # Is key present in all items
            if not multi:
                color = option.palette.color(QPalette.Text)
                color.setAlpha(100)
                option.palette.setColor(QPalette.Text, color)

        def setModelData(self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex) -> None:
            super().setModelData(editor, model, index)
            mp = editor.metaObject().userProperty()
            if mp is not None:
                value = mp.read(editor)
                changed = model.data(index, Qt.EditRole) != value
            else:
                changed = True
            model.setData(index, model.data(index, ModifiedStateRole) or changed,
                          ModifiedStateRole)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QGridLayout(margin=0, spacing=1)
        self.setLayout(layout)
        hlayout = QHBoxLayout(margin=0, spacing=1)

        self.labels_edit = view = QTreeView(
            objectName="annotation-pairs-edit",
            rootIsDecorated=False,
            editTriggers=QTreeView.DoubleClicked | QTreeView.EditKeyPressed,
        )
        self.labels_model = model = QStandardItemModel()
        view.setModel(model)

        view.selectionModel().selectionChanged.connect(
            self.__on_selection_changed)
        view.setItemDelegateForColumn(0, self.KeyEditDelegate(self))
        view.setItemDelegateForColumn(1, self.ValueEditDelegate(self))
        agrp = QActionGroup(view, objectName="annotate-action-group")
        action_add = QAction(
            "+", self, objectName="action-add-item",
            toolTip="Add a new label.",
            shortcut=QKeySequence(QKeySequence.New),
            shortcutContext=Qt.WidgetShortcut
        )
        action_delete = QAction(
            "\N{MINUS SIGN}", self, objectName="action-delete-item",
            toolTip="Remove selected label.",
            shortcut=QKeySequence(QKeySequence.Delete),
            shortcutContext=Qt.WidgetShortcut
        )
        agrp.addAction(action_add)
        agrp.addAction(action_delete)
        view.addActions([action_add, action_delete])

        action_add.triggered.connect(self.addItem)

        def remove_label():
            rows = view.selectionModel().selectedRows(0)
            for row in sorted((i.row() for i in rows), reverse=True):
                self.removeItem(row)
        action_delete.triggered.connect(remove_label)

        self.add_label_action = action_add
        self.remove_label_action = action_delete

        # Necessary signals to know when the labels change
        model.dataChanged.connect(self.changed, Qt.UniqueConnection)
        model.rowsInserted.connect(self.changed, Qt.UniqueConnection)
        model.rowsRemoved.connect(self.changed, Qt.UniqueConnection)

        layout.addWidget(view, 0, 0, 1, 2)
        layout.addLayout(hlayout, 1, 0, 1, 2)
        hlayout.setContentsMargins(0, 0, 0, 0)
        button = FixedSizeButton(
            self, defaultAction=self.add_label_action,
            accessibleName="Add",
        )
        hlayout.addWidget(button)

        button = FixedSizeButton(
            self, defaultAction=self.remove_label_action,
            accessibleName="Remove",
        )
        hlayout.addWidget(button)
        hlayout.addStretch(10)

    @Slot()
    def __on_selection_changed(self):
        selected = self.labels_edit.selectionModel().selectedRows()
        self.remove_label_action.setEnabled(bool(len(selected)))

    @Slot(int)
    def removeItem(self, row: int):
        model = self.view().model()
        idx = model.index(row, 0)
        if idx.data(EditStateRole) == ItemEditState.Added:
            model.removeRows(row, 1, QModelIndex())
        else:
            state = model.data(idx, EditStateRole)
            if state == ItemEditState.Dropped:
                state = ItemEditState.NoState
            else:
                state = ItemEditState.Dropped
            model.setData(idx, state, EditStateRole)
            model.setData(idx.sibling(idx.row(), 1), state, EditStateRole)

    @Slot()
    def addItem(self):
        model = self.model()
        view = self.view()

        model.insertRows(model.rowCount(), 1, QModelIndex())
        row = model.rowCount() - 1
        idx1 = model.index(row, 0)
        idx2 = model.index(row, 1)
        model.setData(idx1, ItemEditState.Added, EditStateRole)
        model.setData(idx2, ItemEditState.Added, EditStateRole)
        view.setCurrentIndex(idx1)
        view.edit(idx1)

    def setData(self, mapping: Mapping) -> None:
        """
        Set the mapping to edit.
        """
        # self.clear()
        mass_key_value_model(self.labels_model, [list(mapping.items())], [])

    __mappings: Sequence[Sequence[Tuple[str, str]]] = ()

    def setMappings(self, mappings: Sequence[Mapping[str, str]], trs):
        self.labels_model.setRowCount(0)
        self.__mappings = [tuple(m.items()) for m in mappings]
        mass_key_value_model(self.labels_model, self.__mappings, trs)

    def mappings(self,):
        trs = mass_key_value_transforms(self.labels_model, )
        trs = chain(trs, repeat([]))
        return [(dict(m), tr) for m, tr in zip(self.__mappings, trs)]

    def getData(self):
        """Retrieve the modified mapping.
        """
        model = self.model()
        index = model.index
        items = []
        for ki, vi in ((index(i, 0), index(i, 1)) for i in range(model.rowCount())):
            key = ki.data(Qt.EditRole)
            value = vi.data(Qt.EditRole)
            if key:
                items.append((key, value))
        return tuple(reversed(tuple(unique_everseen(items, key=itemgetter(0)))))

    def setTransform(self, transform: Sequence[MappingTransform]): ...

    def transform(self) -> Sequence[MappingTransform]:
        model = self.model()
        index = model.index
        transform = []
        for ki, vi in ((index(i, 0), index(i, 1)) for i in
                       range(model.rowCount())):
            state = ki.data(EditStateRole)
            key = ki.data(Qt.EditRole)
            key_ = ki.data(SourceKeyRole)
            value = vi.data(Qt.EditRole)
            value_ = vi.data(SourceValueRole)
            if state == ItemEditState.Added:
                transform.append(AddItem(key_, value))
            elif state == ItemEditState.Dropped:
                transform.append(DeleteKey(key_))
            else:
                if key != key_:
                    transform.append(RenameKey(key_, key))
                if key == key_ and key_ is ...:
                    print("not touching keys ...")
                if isinstance(value, str) and value != value_:
                    transform.append(SetValue(key, value))
                if value is ...:
                    print("not touching values ...")
        return transform

    def setModel(self, model: QAbstractItemModel):
        self.labels_edit.setModel(model)
        self.labels_edit.selectionModel().selectionChanged.connect(
            self.__on_selection_changed)

    def model(self):
        return self.labels_edit.model()

    def view(self):
        return self.labels_edit


def same_value(values: Iterable[A]) -> Optional[A]:
    values = set(values)
    if len(values) == 1:
        return values.pop()
    else:
        return None


def mass_key_value_model(
        model: QStandardItemModel,
        mappings: Sequence[Sequence[Tuple[str, str]]],
        transforms: Sequence[Sequence[MappingTransform]]
) -> QAbstractItemModel:
    mappings = [dict(m) for m in mappings]
    keys_added = list(unique_everseen(
        t.key for t in chain.from_iterable(transforms) if isinstance(t, AddItem)
    ))
    keys_ = list(unique_everseen(chain.from_iterable(mappings)))
    keys = keys_ + [k for k in keys_added if k not in keys_]

    # every key could be added removed or subject to SetValue in different
    # transforms
    transforms_for_key = [
        (key, [trace_tr_for_key(tr, key, None)[1] for tr in transforms])
        for key in keys
    ]
    model.setRowCount(0)
    model.setColumnCount(2)
    model.setHorizontalHeaderLabels(["Key", "Value"])
    model.setRowCount(len(keys))

    class Item(SimpleNamespace):
        key: str
        key_renamed: Optional[str] = None
        values: Sequence[str] = ()
        items: Sequence[Tuple[str, str]]
        values_transformed: Sequence[str] = ()
        transforms: Sequence[Sequence[MappingTransform]] = ()

        Add, Delete, Rename, Set = 1, 2, 4, 8
        flags = 0

        def add_item_transform(self):
            return find(self.transforms, lambda t: isinstance(t, AddItem))

        def rename_key_transform(self):
            return find(self.transforms, lambda t: isinstance(t, RenameKey))

        def set_value_transform(self):
            return find(self.transforms, lambda t: isinstance(t, SetValue))

    def key_in_all_mappings(key):  # is key present in all mappings
        return all(key in m for m in mappings)

    def item(i, j):
        return model.itemFromIndex(model.index(i, j))

    for i, ki, vi in ((i, item(i, 0), item(i, 1))
                      for i in range(model.rowCount())):
        key = keys[i]
        # all transforms acting on key for every mapping
        trs = transforms_for_key[i][1]
        kvs = [trace_tr_for_key(t, key, m.get(key))[0]
               for t, m in zip(trs, mappings)]

        values_ = [v for k, v in kvs]
        keys_ = [k if v is not None else None for k, v in kvs]
        key_in_all = key_in_all_mappings(key)
        state = ItemEditState.NoState
        effective_key = key
        if all(t == [DeleteKey(key)] for t in trs):
            # all keys are deleted.
            state = ItemEditState.Dropped
        elif all(len(t) == 1 and isinstance(t[0], AddItem) for t in trs):
            # all add items same key (might have different values)
            state = ItemEditState.Added
        else:
            effective_key = same_value(keys_)
            if effective_key is None:
                effective_key = ...
        ki.setData(trs, Qt.UserRole)
        ki.setData(effective_key, Qt.EditRole)
        ki.setData(key_in_all, MultiplicityRole)

        model.setItemData(ki.index(), {
            EditStateRole: state,
            MultiplicityRole: key_in_all,
            EditHintOptionList: [k for k, _ in kvs if k is not None],  # keys if renamed
            SourceKeyRole: key,
            TransformRole: list(zip(mappings, trs))
        })
        values = [m.get(keys[i]) for m in mappings]
        print(key, values_, trs, kvs)
        effective_value = same_value(values_)
        if effective_value is None:
            effective_value = ...

        effective_source_value = same_value(values_)
        if effective_source_value is None:
            effective_source_value = ...
        vi.setData(effective_value, Qt.EditRole)
        model.setItemData(vi.index(), {
            EditStateRole: state,
            Qt.EditRole: effective_value,
            Qt.DisplayRole: effective_value,
            SourceValueRole: effective_source_value,
            EditHintOptionList: [v for v in values_ if v is not None],
            MultiplicityRole: key_in_all and same_value(values),
            TransformRole: list(zip(mappings, trs))
        })
    return model


def mass_key_value_transforms(model: QAbstractItemModel):
    index = model.index
    trs: Sequence[Tuple[Mapping, MappingTransform]]
    trs = model.index(0, 0).data(TransformRole)
    if trs is None:
        return []
    mappings = [m for m, ts in trs]
    transforms = [[] for _ in trs]

    for ki, vi in ((index(i, 0), index(i, 1)) for i in range(model.rowCount())):
        state = ki.data(EditStateRole)
        key: str = ki.data(SourceKeyRole)
        key_ = ki.data(Qt.EditRole)

        value_ = vi.data(Qt.EditRole)
        value = vi.data(SourceValueRole)
        trs = ki.data(TransformRole)  # source
        if trs is None:
            trs = [(m, []) for m in mappings]
        if state == ItemEditState.Added:
            assert key_ is not ...
            if key_ is not None and value_ is not None:
                for t, (m, tr_) in zip(transforms, trs):
                    (k, v), _ = trace_tr_for_key(tr_, key, m.get(key))
                    if value_ is not ...:
                        v = value_
                    if v is not None:
                        t.append(AddItem(key_, v))
        elif state == ItemEditState.Dropped:
            for t, (m, _) in zip(transforms, trs):
                if key in m:
                    t.append(DeleteKey(key))
        else:
            for t, (m, tr_) in zip(transforms, trs):
                # original transformed k, v pairs
                (k, v), _ = trace_tr_for_key(tr_, key, m.get(key))
                if key not in m:
                    if key is not ... and value_ is not ...:
                        t.append(AddItem(key_, value_))
                    elif k is not None and v is not None:  # preserve existing AddItems
                        t.append(AddItem(k, v if value_ is ... else value_))
                    continue
                if key in m and k is None and (key_ is ... or value_ is ...):
                    t.append(DeleteKey(key))  # preserve DeleteKey when
                    continue

                effective_key = key
                if key_ is ...:
                    if k != key:  # preserve existing renames
                        t.append(RenameKey(key, k))
                        effective_key = k
                elif key_ != key:
                    t.append(RenameKey(key, key_))
                    effective_key = key_

                if value_ is ...:
                    if v != m.get(key):  # preserve existing value set
                        t.append(SetValue(effective_key, v))
                else:
                    if m.get(key) != value_:
                        t.append(SetValue(effective_key, value_))
    return transforms


TypeTransforms = {
    Real: AsContinuous,
    Categorical: AsCategorical,
    Time: AsTime,
    String: AsString,
}


class MassVariablesEditor(QWidget):
    changed = Signal()
    edited = Signal()

    def setNamePattern(self, pattern: str):
        cpos = self.name_edit.cursorPosition()
        self.name_edit.setText(pattern)
        self.name_edit.setCursorPosition(cpos)

    def namePattern(self) -> str:
        return self.name_edit.text()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__history: Dict[Variable, Sequence[Transform]] = {}
        layout = QFormLayout(
            fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow,
            objectName="editor-form-layout"
        )
        self.name_edit = QLineEdit(
            objectName="name-pattern-edit", placeholderText="name...",
        )

        self.name_edit.editingFinished.connect(self.edited)
        self.name_edit.textChanged.connect(self.changed)

        layout.addRow("Name", self.name_edit)
        self.type_combo = typecb = QComboBox(objectName="type-combo")
        typecb.addItem("Keep", VariableTypes)
        typecb.insertSeparator(1)
        typecb.addItem(variable_icon(Categorical), "Categorical", Categorical)
        typecb.addItem(variable_icon(Real), "Numeric", Real)
        typecb.addItem(variable_icon(String), "Text", String)
        typecb.addItem(variable_icon(Time), "Time", Time)
        typecb.activated[int].connect(self.__reinterpret_activated)
        layout.addRow("Type", typecb)
        self.unlink_var_cb = QCheckBox("Unlink", objectName="unlink-edit")
        self.unlink_var_cb.toggled.connect(self.changed)
        self.unlink_var_cb.toggled.connect(self.edited)

        layout.addRow("", self.unlink_var_cb)
        self.categories_editor = CategoriesEditor(objectName="categories-editor")

        self.categories_editor.changed.connect(self.changed)
        self.categories_editor.edited.connect(self.edited)

        layout.addRow("Values", self.categories_editor)
        self.annotations_edit = KeyValueEditor(objectName="annotations-editor")
        self.annotations_edit.edited.connect(self.edited)
        self.annotations_edit.changed.connect(self.changed)
        layout.addRow("Labels", self.annotations_edit)
        self.setLayout(layout)

        class RowItem(SimpleNamespace):
            label: QWidget
            field: QWidget
            row: int

            def setVisible(self, visible: bool) -> None:
                if visible and not self.field.isVisibleTo(self.field.parent()):
                    layout.insertRow(self.row, self.label, self.field)
                elif not visible and self.field.isVisibleTo(self.field.parent()):
                    layout.takeRow(self.row)
                self.label.setVisible(visible)
                self.field.setVisible(visible)

        self.__categories_edit_row = RowItem(
            label=layout.itemAt(3, QFormLayout.LabelRole).widget(),
            field=self.categories_editor,
            row=3
        )
        self.__categories_edit_row.setVisible(False)

    class ItemData(SimpleNamespace):
        # The source data vector
        data: DataVector = None
        type_transform: ReinterpretTransform = None
        # Data after applied type_transform
        reinterpret_data: DataVector = None
        # The transforms effecting the variable (after applied type_transform)
        transforms: Sequence[Transform] = None
        # The transformed variable
        transformed_vtype: Variable = None

        @property
        def vtype(self) -> Variable:
            return self.data.vtype

        @staticmethod
        def create(
                data: DataVector,
                reinterpret_transform: Optional[ReinterpretTransform],
                transforms: Sequence[Transform]
        ):
            if reinterpret_transform is not None:
                reinterpret_data = reinterpret_transform(data)
                var_t = reinterpret_data.vtype
            else:
                reinterpret_data = data
                var_t = data.vtype

            for tr in transforms:
                var_t = tr(var_t)
            return MassVariablesEditor.ItemData(
                data=data, type_transform=reinterpret_transform,
                reinterpret_data=reinterpret_data,
                transforms=transforms, transformed_vtype=var_t,
            )

    __items: List[ItemData] = []

    def setData(self, items: Sequence[Tuple[DataVector, Sequence[Transform]]]):
        """
        Set the editor data.

        Note
        ----
        This must be a `DataVector` as the vector's values are needed for type
        reinterpretation/casts.

        If the `transform` sequence contains ReinterpretTransform then it
        must be in the first position.
        """
        ItemData = MassVariablesEditor.ItemData
        items_ = []
        for dvec, transform in items:
            type_transform, transform = take_first_if(
                transform, lambda t: isinstance(t, ReinterpretTransformTypes)
            )
            assert not any(isinstance(t, ReinterpretTransformTypes)
                           for t in transform)

            # self.__history[dvec.vtype] = tuple(transform)
            items_.append(ItemData.create(dvec, type_transform, transform))
        self.__setItemData(items_)

    def __setItemData(self, items: Sequence[ItemData]):
        self.__items = list(items)
        type_ = same_value(type(it.reinterpret_data.vtype) for it in items)
        typecombo = self.type_combo

        if type_ is not None:
            type_index = typecombo.findData(type_, Qt.UserRole)
            typecombo.setCurrentIndex(type_index if type_index != -1 else 0)
        else:
            typecombo.setCurrentIndex(0)  # Keep

        names = [it.transformed_vtype.name for it in items]
        self.setNamePattern(infer_pattern(names))

        annots = [it.vtype.annotations for it in items]
        annots_map_trs = []
        for it in items:
            mat = find_instance(it.transforms, ModifyAnnotations)
            if mat is not None:
                annots_map_trs.append(mat.transform)
            else:
                annots_map_trs.append(
                    mapping_diff(dict(it.vtype.annotations),
                    dict(it.transformed_vtype.annotations))
                )
        self.annotations_edit.setMappings(list(map(dict, annots)), annots_map_trs)
        # mass_key_value_model(self.annotations_edit.labels_model, annots, annots_map_trs)

        can_edit_categories = len(items) == 1 and \
                              isinstance(items[0].transformed_vtype, Categorical)
        if can_edit_categories:
            item = items[0]
            mapping = find_instance(item.transforms, CategoriesMapping)
            self.categories_editor.setData(
                item.reinterpret_data if item.reinterpret_data is not None else item.data,
                mapping
            )
        else:
            self.categories_editor.clear()
        if can_edit_categories and not self.categories_editor.isVisibleTo(self):
            self.__categories_edit_row.setVisible(True)
        elif not can_edit_categories and self.categories_editor.isVisibleTo(self):
            self.__categories_edit_row.setVisible(False)
        self.categories_editor.setEnabled(can_edit_categories)
        self.annotations_edit.setEnabled(bool(items))

    def data(self) -> Sequence[Tuple[DataVector, Sequence[Transform]]]:
        ItemData = MassVariablesEditor.ItemData
        items = self.__items
        if not items:
            return []
        nameptr = self.namePattern()
        target_type: Type[Variable] = self.type_combo.currentData(Qt.UserRole)
        if len(items) > 1:
            names = expand_pattern(nameptr, len(items))
        else:
            names = [nameptr]

        unlink = self.unlink_var_cb.isChecked()
        # annots_tr = self.annotations_edit.transform()
        annot_trs = mass_key_value_transforms(self.annotations_edit.model())
        annot_trs = self.annotations_edit.mappings()
        items_t = [ItemData.create(it.data, None, []) for it in items]
        for name, item, item_t, (_, annot_tr) in zip_longest(names, items, items_t, annot_trs):
            trs = []
            vtype = item.vtype
            if not isinstance(vtype, target_type):
                reinterpret_transform = TypeTransforms[target_type]()
                item_t.type_transform = reinterpret_transform
                item_t.reinterpret_data = reinterpret_transform(item.data)
                vtype = item_t.reinterpret_data.vtype
                trs += [reinterpret_transform]

            if name is not None and name != vtype.name:
                trs.append(Rename(name))

            if annot_tr is not None:
                annots = apply_mapping_transform(
                    dict(item.data.vtype.annotations), annot_tr)
                annots = tuple(annots.items())
                if set(annots) != set(item.data.vtype.annotations):
                    trs.append(ModifyAnnotations(annot_tr))
            if vtype.linked and unlink:
                trs.append(Unlink())
            if isinstance(vtype, Categorical):
                if self.categories_editor.isEnabled():
                    assert len(items) == 1
                    mapping = self.categories_editor.transform()
                    if any(_1 != _2 or _2 != _3
                           for (_1, _2), _3 in
                           zip_longest(mapping.mapping, vtype.categories)):
                        trs.append(mapping)
                else:
                    mapping = find_instance(item.transforms, CategoriesMapping)
                    if mapping is not None:
                        trs.append(mapping)
            item_t.transforms = trs
            print(trs)
        return [(it.data, it.transforms) for it in items_t]

    def clear(self):
        self.__setItemData([])
        self.changed.emit()

    def __reinterpret_activated(self, index):
        Specific = {
            Categorical: CategoricalTransformTypes
        }
        target_type = self.type_combo.itemData(index, Qt.UserRole)
        items_t = []
        for item in self.__items:
            data = item.data
            var = data.vtype
            type_transform = item.type_transform
            tr = item.transforms

            if item.reinterpret_data is not None:
                self.__history[item.reinterpret_data.vtype] = tr
            else:
                self.__history[var] = tr

            # take/preserve the general transforms that apply to all types
            specific = Specific.get(type(var), ())
            tr = [t for t in tr if not isinstance(t, specific)]

            if isinstance(var, target_type):
                if type_transform is not None:
                    type_transform = None
            else:
                type_transform = TypeTransforms[target_type]()

            item_ = MassVariablesEditor.ItemData.create(
                data, type_transform, tr
            )

            if item_.reinterpret_data is not None:
                var_ = item_.reinterpret_data.vtype
            else:
                var_ = item_.data.vtype

            # restore type specific transforms from history
            if var_ in self.__history:
                tr_ = self.__history[var]
            else:
                tr_ = []
            # merge tr and _tr
            tr = tr + [t for t in tr_ if isinstance(t, specific)]
            for t in tr:
                var_ = t(var_)
            item_.transforms = tr
            item_.transformed_vtype = var_
            items_t.append(item_)
        changed = any(it.type_transform != it_t.type_transform or
                      it.transforms != it_t.transforms
                      for it, it_t in zip(self.__items, items_t))
        self.__setItemData(items_t)

        if changed:
            self.changed.emit()
            self.edited.emit()


def take_first_if(
        seq: Sequence[A], predicate: Callable[[A], bool]
) -> Tuple[Optional[A], Sequence[A]]:
    if len(seq) == 0:
        return None, seq
    el0 = seq[0]
    if predicate(el0):
        return el0, seq[1:]
    else:
        return None, seq


def find(seq: Iterable[A], predicate: Callable[[A], bool]) -> Optional[A]:
    for el in seq:
        if predicate(el):
            return el


def find_instance(seq: Iterable[A], type_: Type[B]) -> Optional[B]:
    res = find(seq, lambda t: isinstance(t, type_))
    if res is not None:
        return cast('B', res)
    else:
        return None


def variable_icon(var):
    # type: (Union[Variable, Type[Variable], ReinterpretTransform]) -> QIcon
    if not isinstance(var, type):
        var = type(var)

    if issubclass(var, (Categorical, AsCategorical)):
        return gui.attributeIconDict[1]
    elif issubclass(var, (Real, AsContinuous)):
        return gui.attributeIconDict[2]
    elif issubclass(var, (String, AsString)):
        return gui.attributeIconDict[3]
    elif issubclass(var, (Time, AsTime)):
        return gui.attributeIconDict[4]
    else:
        return gui.attributeIconDict[-1]


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

    want_control_area = False

    def __init__(self):
        super().__init__()
        self.data = None  # type: Optional[Orange.data.Table]
        self._invalidated = False
        mainlayout = self.mainArea.layout()
        assert isinstance(mainlayout, QVBoxLayout)
        self._edit_splitter = splitter = QSplitter(
            Qt.Horizontal,
            childrenCollapsible=False,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.MinimumExpanding),
            objectName="edit-splitter"
        )
        # layout = QHBoxLayout()
        mainlayout.addWidget(splitter)
        box = QGroupBox("Variables", objectName="variables-group-box")
        box.setLayout(QVBoxLayout())
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
        box.setLayout(QVBoxLayout(margin=4))
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
        mainlayout.addLayout(form)

        bbox = QDialogButtonBox(
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum),
            styleSheet=f"button-layout: {QDialogButtonBox.MacLayout:d};",
        )
        bapply = QPushButton(
            "Apply",
            objectName="button-apply",
            toolTip="Apply changes and commit data on output.",
            default=True,
            autoDefault=False
        )
        bapply.clicked.connect(self.commit)
        breset = QPushButton(
            "Reset Selected",
            objectName="button-reset",
            toolTip="Rest selected variable to its input state.",
            autoDefault=False
        )
        breset.clicked.connect(self.reset_selected)
        breset_all = QPushButton(
            "Reset All",
            objectName="button-reset-all",
            toolTip="Reset all variables to their input state.",
            autoDefault=False
        )
        breset_all.clicked.connect(self.reset_all)

        bbox.addButton(bapply, QDialogButtonBox.AcceptRole)
        bbox.addButton(breset, QDialogButtonBox.ResetRole)
        bbox.addButton(breset_all, QDialogButtonBox.ResetRole)

        mainlayout.addWidget(bbox)
        self.variables_view.setFocus(Qt.NoFocusReason)  # initial focus

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)
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
            self.info.set_input_summary(len(data),
                                        format_summary_details(data))
            self.setup_model(data)
            self.le_output_name.setPlaceholderText(data.name)
            self.openContext(self.data)
            self._restore()
        else:
            self.le_output_name.setPlaceholderText("")
            self.info.set_input_summary(self.info.NoInput)

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
            index = self.selected_var_index()
            if index >= 0:
                self.open_editor(index)
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
            self.info.set_output_summary(self.info.NoOutput)
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
            self.info.set_output_summary(len(data),
                                         format_summary_details(data))
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
            self.info.set_output_summary(self.info.NoOutput)
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
        self.info.set_output_summary(len(new_data),
                                     format_summary_details(new_data))

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
        lookup = np.full(len(source_values), np.nan, dtype=np.float)
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
    variable = Orange.data.StringVariable(
        name=name, compute_value=Identity(var)
    )
    variable.attributes.update(annotations)
    return variable


def ftry(
        func: Callable[..., A],
        error: Union[Type[BaseException], Tuple[Type[BaseException]]],
        default: B
) -> Callable[..., Union[A, B]]:
    """
    Wrap a `func` such that if `errors` occur `default` is returned instead."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except error:
            return default
    return wrapper


class DictMissingConst(dict):
    """
    `dict` with a constant for `__missing__()` value.
    """
    __slots__ = ("__missing",)

    def __init__(self, missing, *args, **kwargs):
        self.__missing = missing
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        return self.__missing


def make_dict_mapper(
        mapping: Mapping, dtype: Optional[DType] = None
) -> Callable:
    """
    Wrap a `mapping` into a callable ufunc-like function with
    `out`, `dtype`, `where`, ... parameters. If `dtype` is passed to
    `make_dict_mapper` it is used as a the default return dtype,
    otherwise the default dtype is `object`.
    """
    _vmapper = np.frompyfunc(mapping.__getitem__, 1, 1)

    def mapper(arr, out=None, dtype=dtype, **kwargs):
        arr = np.asanyarray(arr)
        if out is None and dtype is not None and arr.shape != ():
            out = np.empty_like(arr, dtype)
        return _vmapper(arr, out, dtype=dtype, **kwargs)
    return mapper


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
        out = np.full(arr.shape, np.nan, np.float if dtype is None else dtype)
    if np.issubdtype(arr.dtype, np.inexact) or \
            np.issubdtype(arr.dtype, np.integer):
        np.copyto(out, arr, casting="unsafe", where=where)
        return out
    return _parse_float(arr, out, where=where, **kwargs)


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
    elif isinstance(tr, AsString):
        f = Lookup(var, np.array(var.values, dtype=object), unknown="")
        rvar = Orange.data.StringVariable(
            name=var.name, compute_value=f
        )
    elif isinstance(tr, AsContinuous):
        f = Lookup(var, np.array(list(map(parse_float, var.values))),
                   unknown=np.nan)
        rvar = Orange.data.ContinuousVariable(
            name=var.name, compute_value=f, sparse=var.sparse
        )
    elif isinstance(tr, AsTime):
        _tvar, values = time_parse(var.values)
        f = Lookup(var, np.array(values), unknown=np.nan)
        rvar = Orange.data.TimeVariable(
            name=var.name, have_date=_tvar.have_date,
            have_time=_tvar.have_time, compute_value=f,
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
        tstr = ToStringTransform(var)
        rvar = Orange.data.StringVariable(
            name=var.name, compute_value=tstr
        )
    elif isinstance(tr, AsTime):
        rvar = Orange.data.TimeVariable(
            name=var.name, compute_value=Identity(var)
        )
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
    elif isinstance(tr, AsString):
        return var
    elif isinstance(tr, AsTime):
        tvar, _ = time_parse(np.unique(data.data[~data.mask]))
        rvar = Orange.data.TimeVariable(
            name=var.name, have_date=tvar.have_date, have_time=tvar.have_time,
            compute_value=ReparseTimeTransform(var)
        )
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


def datetime_to_epoch(dti: pd.DatetimeIndex) -> np.ndarray:
    """Convert datetime to epoch"""
    data = dti.values.astype("M8[us]")
    mask = np.isnat(data)
    data = data.astype(float) / 1e6
    data[mask] = np.nan
    return data


class ReparseTimeTransform(Transformation):
    """
    Re-parse the column's string repr as datetime.
    """
    def transform(self, c):
        c = column_str_repr(self.variable, c)
        c = pd.to_datetime(c, errors="coerce")
        return datetime_to_epoch(c)


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
