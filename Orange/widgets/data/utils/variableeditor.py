"""
Variable editor widget and supporting utilities.
"""
import abc
import warnings
from collections import Counter
from itertools import zip_longest, chain, repeat
from operator import itemgetter
from types import SimpleNamespace
from typing import (
    Dict, Sequence, Tuple, Union, List, Optional, Callable, Type, Mapping,
    MutableMapping, TypeVar, Any, FrozenSet, cast, Iterable, Hashable
)

import numpy as np
import pandas as pd

from AnyQt.QtCore import (
    Signal, Slot, Qt, QModelIndex, QPersistentModelIndex, QAbstractItemModel,
    QPoint
)
from AnyQt.QtGui import (
    QKeySequence, QIcon, QStandardItemModel, QStandardItem, QPalette
)
from AnyQt.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QComboBox, QCheckBox, QStyledItemDelegate,
    QAbstractItemView, QStyleOptionViewItem, QStyle, QMenu, QGridLayout,
    QListView, QHBoxLayout, QActionGroup, QAction, QDialog, QRadioButton,
    QLabel, QSpinBox, QDoubleSpinBox, QGroupBox, QDialogButtonBox, QSizePolicy,
    QVBoxLayout, QTreeView
)

from orangewidget.utils.itemmodels import PyListModel, signal_blocking

from Orange.widgets import gui
from Orange.widgets.data.utils.bracepattern import infer_pattern, expand_pattern
from Orange.widgets.utils import datatype, unique_everseen, map_rect_to, \
    map_rect_to_global, disconnected
from Orange.widgets.utils.buttons import FixedSizeButton


A = TypeVar("A")
B = TypeVar("B")
H = TypeVar("H", bound=Hashable)

ndarray = np.ndarray
MArray = np.ma.MaskedArray
DType = Union[np.dtype, type]

#: An ordered sequence of key, value pairs (variable annotations)
AnnotationsType = Tuple[Tuple[str, str], ...]

# Define abstract representation of the variable types edited


@datatype
class Categorical:
    name: str
    categories: Tuple[str, ...]
    annotations: AnnotationsType
    linked: bool


@datatype
class Real:
    name: str
    # a precision (int, and a format specifier('f', 'g', or '')
    format: Tuple[int, str]
    annotations: AnnotationsType
    linked: bool


@datatype
class String:
    name: str
    annotations: AnnotationsType
    linked: bool


@datatype
class Time:
    name: str
    annotations: AnnotationsType
    linked: bool


Variable = Union[Categorical, Real, Time, String]
VariableTypes = (Categorical, Real, Time, String)

# Define variable transformations.


class Transform:
    @abc.abstractmethod
    def __call__(self, var: Variable) -> Variable:
        raise NotImplementedError


@datatype
class Rename(Transform):
    """Rename a variable."""
    #: The new name
    name: str

    def __call__(self, var: Variable) -> Variable:
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


@datatype
class CategoriesMapping(Transform):
    """Change categories of a categorical variable."""
    mapping: CategoriesMappingType

    def __call__(self, var):
        # type: (Categorical) -> Categorical
        cat = tuple(unique_everseen(cj for _, cj in self.mapping if cj is not None))
        return var._replace(categories=cat)


@datatype
class Annotate(Transform):
    """
    Replace variable annotations.
    """
    annotations: AnnotationsType

    def __call__(self, var):
        return var._replace(annotations=self.annotations)


@datatype
class ModifyAnnotations(Transform):
    transform: Sequence['MappingTransform']

    def __call__(self, var: Variable) -> Variable:
        return var._replace(annotations=tuple(
            apply_mapping_transform(dict(var.annotations), self.transform)
        ))


@datatype
class Unlink(Transform):
    """Unlink variable from its source, that is, remove compute_value"""
    def __call__(self, var: Variable) -> Variable:
        return var


@datatype
class StrpTime(Transform):
    """Use format on variable interpreted as time"""
    label: str
    formats: List[str]
    have_date: bool
    have_time: bool


Transform = Union[Rename, CategoriesMapping, Annotate, Unlink]
TransformTypes = (Rename, CategoriesMapping, Annotate, Unlink)

CategoricalTransformTypes = (CategoriesMapping, Unlink)


class MappingTransform:
    """Transform that operates on a MutableMapping."""
    key: str

    def __call__(self, mapping: MutableMapping) -> MutableMapping:
        raise NotImplementedError


@datatype
class DeleteKey(MappingTransform):
    key: str

    def __call__(self, mapping: MutableMapping) -> MutableMapping:
        try:
            del mapping[self.key]
        except KeyError:
            pass
        return mapping


@datatype
class AddItem(MappingTransform):
    key: str
    value: str

    def __call__(self, mapping: MutableMapping) -> MutableMapping:
        mapping[self.key] = self.value
        return mapping


@datatype
class RenameKey(MappingTransform):
    key: str
    target: str

    def __call__(self, mapping: MutableMapping) -> MutableMapping:
        try:
            v = mapping[self.key]
        except KeyError:
            return mapping
        else:
            mapping[self.target] = v
            del mapping[self.key]
        return mapping


@datatype
class SetValue(MappingTransform):
    key: str
    value: str

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


# Reinterpret vector transformations.
class BaseVector:
    vtype: Union[Categorical, Real, String, Time]
    data: Callable[[], MArray]

    def __iter__(self):
        return iter(tuple(self.vtype, self.data))


@datatype
class CategoricalVector(BaseVector):
    vtype: Categorical
    data: Callable[[], MArray]


@datatype
class RealVector(BaseVector):
    vtype: Real
    data: Callable[[], MArray]


@datatype
class StringVector(BaseVector):
    vtype: String
    data: Callable[[], MArray]


@datatype
class TimeVector(BaseVector):
    vtype: Time
    data: Callable[[], MArray]


DataVector = Union[CategoricalVector, RealVector, StringVector, TimeVector]
DataVectorTypes = (CategoricalVector, RealVector, StringVector, TimeVector)


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


def formatter_for_dtype(dtype: np.dtype) -> Callable[[Any], str]:
    if dtype.metadata is None:
        return str
    else:
        return dtype.metadata.get("__formatter", str)  # metadata abuse


class DictMissingConst(dict):
    """
    `dict` with a constant for `__missing__()` value.
    """
    __slots__ = ("missing",)

    def __init__(self, missing, *args, **kwargs):
        self.missing = missing
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        return self.missing

    def __eq__(self, other):
        return isinstance(other, DictMissingConst) \
           and super().__eq__(other) \
           and self.missing == other.missing


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


def categorize_unique(data: MArray) -> Tuple[ndarray, MArray]:
    unq, index = masked_unique(data)
    if np.any(unq.mask):
        unq = unq[:-1]
        assert not np.any(unq.mask), "masked value if present must be last"
    unq = unq.data
    index[data.mask] = -1
    index = MArray(index, mask=data.mask)
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


def categorical_to_string_vector(data: MArray, values: Tuple[str, ...]) -> MArray:
    lookup = np.asarray(values, object)
    out = np.full(data.shape, "", dtype=object)
    mask_ = ~data.mask
    out[mask_] = lookup[data.data[mask_]]
    return MArray(out, mask=data.mask, fill_value="")


@datatype
class AsString:
    """Reinterpret a data vector as a string."""
    def __call__(self, vector: DataVector) -> StringVector:
        var = vector.vtype
        if isinstance(var, String):
            return vector
        return StringVector(
            String(var.name, var.annotations, False),
            lambda: as_string(vector.data()),
        )


@datatype
class AsContinuous:
    """
    Reinterpret as a continuous variable (values that do not parse as
    float are NaN).
    """
    def __call__(self, vector: DataVector) -> RealVector:
        var = vector.vtype
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


@datatype
class AsCategorical:
    """Reinterpret as a categorical variable"""
    def __call__(self, vector: DataVector) -> CategoricalVector:
        # this is the main complication in type transformation since we need
        # the data and not just the variable description
        var = vector.vtype
        if isinstance(var, Categorical):
            return vector
        if isinstance(var, (Real, Time, String)):
            data, values = categorical_from_vector(vector.data())
            return CategoricalVector(
                Categorical(var.name, values, var.annotations, var.linked),
                lambda: data
            )
        raise AssertionError


@datatype
class AsTime:
    """Reinterpret as a datetime vector"""
    def __call__(self, vector: DataVector) -> TimeVector:
        var = vector.vtype
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

#: Role used to retrieve the count of 'key' values in the model.
MultiplicityRole = Qt.UserRole + 0x67
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


class CountedListModel(PyListModel):
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

    class CatEditComboBox(QComboBox):
        prows: List[QPersistentModelIndex]

    def createEditor(
            self, parent: QWidget, option: 'QStyleOptionViewItem',
            index: QModelIndex
    ) -> QWidget:
        view = option.widget
        assert isinstance(view, QAbstractItemView)
        selmodel = view.selectionModel()
        rows = selmodel.selectedRows(0)
        if len(rows) < 2:
            return super().createEditor(parent, option, index)
        # edit multiple selection
        cb = CategoriesEditDelegate.CatEditComboBox(
            editable=True, insertPolicy=QComboBox.InsertAtBottom)
        cb.setParent(view, Qt.Popup)
        cb.addItems(
            list(unique_everseen(str(row.data(Qt.EditRole)) for row in rows)))
        prows = [QPersistentModelIndex(row) for row in rows]
        cb.prows = prows
        return cb

    def updateEditorGeometry(
            self, editor: QWidget, option: 'QStyleOptionViewItem',
            index: QModelIndex
    ) -> None:
        if isinstance(editor, CategoriesEditDelegate.CatEditComboBox):
            view = cast(QAbstractItemView, option.widget)
            view.scrollTo(index)
            vport = view.viewport()
            vrect = view.visualRect(index)
            vrect = map_rect_to(vport, view, vrect)
            vrect = vrect.intersected(vport.geometry())
            vrect = map_rect_to_global(vport, vrect)
            size = editor.sizeHint().expandedTo(vrect.size())
            editor.resize(size)
            editor.move(vrect.topLeft())
        else:
            super().updateEditorGeometry(editor, option, index)

    def setModelData(
            self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex
    ) -> None:
        if isinstance(editor, CategoriesEditDelegate.CatEditComboBox):
            text = editor.currentText()
            with signal_blocking(model):
                for prow in editor.prows:
                    if prow.isValid():
                        model.setData(QModelIndex(prow), text, Qt.EditRole)
            # this could be better
            model.dataChanged.emit(
                model.index(0, 0), model.index(model.rowCount() - 1, 0),
                (Qt.EditRole,)
            )
        else:
            super().setModelData(editor, model, index)


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
        self.rename_selected_items = QAction(
            "Rename selected items", group,
            iconText="=",
            objectName="action-rename-selected-items",
            toolTip="Rename selected items.",
            shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_Equal),
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
        self.rename_selected_items.triggered.connect(self._rename_selected_categories)
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
            self, defaultAction=self.rename_selected_items,
            accessibleName="Merge selected items"
        )
        button6 = FixedSizeButton(
            self, defaultAction=self.merge_items,
            accessibleName="Merge",
        )
        self.categories_edit.addActions([
            self.move_value_up, self.move_value_down,
            self.add_new_item, self.remove_item, self.rename_selected_items,
        ])
        self.categories_edit.setContextMenuPolicy(Qt.CustomContextMenu)

        def context_menu(pos: QPoint):
            viewport = self.categories_edit.viewport()
            menu = QMenu(self.categories_edit)
            menu.setAttribute(Qt.WA_DeleteOnClose)
            menu.addActions([self.rename_selected_items, self.remove_item])
            menu.popup(viewport.mapToGlobal(pos))

        self.categories_edit.customContextMenuRequested.connect(context_menu)
        hlayout.addWidget(button1)
        hlayout.addWidget(button2)
        hlayout.addSpacing(3)
        hlayout.addWidget(button3)
        hlayout.addWidget(button4)
        hlayout.addSpacing(3)
        hlayout.addWidget(button5)
        hlayout.addWidget(button6)
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
        status = dlg.exec()
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

    def _rename_selected_categories(self):
        """
        Rename selected categories and merging them.
        Popup an editable combo box for selection/edit of a new value.
        """
        view = self.categories_edit
        selmodel = view.selectionModel()
        index = view.currentIndex()
        if not selmodel.isSelected(index):
            indices = selmodel.selectedRows(0)
            if indices:
                index = indices[0]
        # delegate to the CategoriesEditDelegate
        view.edit(index)



SourceKeyRole = SourceNameRole
SourceValueRole = SourceNameRole + 1
EditHintOptionList = Qt.UserRole + 56
ModifiedStateRole = EditStateRole + 1346


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
            multi = index.data(MultiplicityRole)  # Is key present in all mappings
            if state == ItemEditState.Added:
                option.text += " (added)"
            elif state == ItemEditState.Dropped:
                option.font.setStrikeOut(True)
            elif multi:
                original_key = index.data(SourceKeyRole)
                if option.text != original_key:
                    option.text = f"{original_key} \N{RIGHTWARDS ARROW} {option.text}"
                    modified = True
            option.font.setItalic(modified)
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


def same_value(values: Iterable[H]) -> Optional[H]:
    values = set(values)
    if len(values) == 1:
        return values.pop()
    else:
        return None


#: ItemDataRole storing the data vector transform
#: (`List[Union[ReinterpretTransform, Transform]]`)
TransformRole = Qt.UserRole + 42


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

    def key_in_all_mappings(key):  # is key present in all mappings
        return all(key in m for m in mappings)

    def item(i: int, j: int) -> QStandardItem:
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
        self.unlink_var_cb.stateChanged.connect(self.changed)
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

        all_linked = all(item.vtype.linked for item in items)
        unlink_trs = [find_instance(item.transforms, Unlink) is not None
                      for item in items]
        all_have_unlinked = all(unlink_trs)
        none_have_unlinked = not any(unlink_trs)

        self.unlink_var_cb.setEnabled(
            bool(items) and all_linked and (all_have_unlinked or none_have_unlinked)
        )
        if items and all_have_unlinked:
            state = Qt.Checked
        elif none_have_unlinked:
            state = Qt.Unchecked
        else:
            state = Qt.PartiallyChecked
        self.unlink_var_cb.setCheckState(state)

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
