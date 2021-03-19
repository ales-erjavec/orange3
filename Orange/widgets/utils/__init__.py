import enum
import inspect
import sys
from collections import deque
from contextlib import contextmanager
from typing import (
    TypeVar, Callable, Any, Iterable, Optional, Hashable, Type, Union, Tuple,
    NamedTuple, Mapping
)
from xml.sax.saxutils import escape
from typing import NamedTuple as DataType

from AnyQt.QtCore import QObject, QRect, Qt
from AnyQt.QtWidgets import QWidget

from Orange.data.variable import TimeVariable
from Orange.util import deepgetattr


def vartype(var):
    if var.is_discrete:
        return 1
    elif var.is_continuous:
        if isinstance(var, TimeVariable):
            return 4
        return 2
    elif var.is_string:
        return 3
    else:
        return 0


def progress_bar_milestones(count, iterations=100):
    return {int(i * count / float(iterations)) for i in range(iterations)}


def getdeepattr(obj, attr, *arg, **kwarg):
    if isinstance(obj, dict):
        return obj.get(attr)
    return deepgetattr(obj, attr, *arg, **kwarg)


def to_html(s):
    return s.replace("<=", "≤").replace(">=", "≥"). \
        replace("<", "&lt;").replace(">", "&gt;").replace("=\\=", "≠")

getHtmlCompatibleString = to_html


def get_variable_values_sorted(variable):
    """
    Return a list of sorted values for given attribute, if all its values can be
    cast to int's.
    """
    if variable.is_continuous:
        return []
    try:
        return sorted(variable.values, key=int)
    except ValueError:
        return variable.values


def dumpObjectTree(obj, _indent=0):
    """
    Dumps Qt QObject tree. Aids in debugging internals.
    See also: QObject.dumpObjectTree()
    """
    assert isinstance(obj, QObject)
    print('{indent}{type} "{name}"'.format(indent=' ' * (_indent * 4),
                                           type=type(obj).__name__,
                                           name=obj.objectName()),
          file=sys.stderr)
    for child in obj.children():
        dumpObjectTree(child, _indent + 1)


def getmembers(obj, predicate=None):
    """Return all the members of an object in a list of (name, value) pairs sorted by name.

    Behaves like inspect.getmembers. If a type object is passed as a predicate,
    only members of that type are returned.
    """

    if isinstance(predicate, type):
        def mypredicate(x):
            return isinstance(x, predicate)
    else:
        mypredicate = predicate
    return inspect.getmembers(obj, mypredicate)


def qname(type_: type) -> str:
    """Return the fully qualified name for a `type_`."""
    return "{0.__module__}.{0.__qualname__}".format(type_)


_T1 = TypeVar("_T1")  # pylint: disable=invalid-name
_E = TypeVar("_E", bound=enum.Enum)  # pylint: disable=invalid-name
_A = TypeVar("_A")  # pylint: disable=invalid-name
_B = TypeVar("_B")  # pylint: disable=invalid-name


def apply_all(seq, op):
    # type: (Iterable[_T1], Callable[[_T1], Any]) -> None
    """Apply `op` on all elements of `seq`."""
    # from itertools recipes `consume`
    deque(map(op, seq), maxlen=0)


def ftry(
        func: Callable[..., _A],
        error: Union[Type[BaseException], Tuple[Type[BaseException]]],
        default: _B
) -> Callable[..., Union[_A, _B]]:
    """
    Wrap a `func` such that if `errors` occur `default` is returned instead.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except error:
            return default
    return wrapper


def unique_everseen(iterable, key=None):
    # type: (Iterable[_T1], Optional[Callable[[_T1], Hashable]]) -> Iterable[_T1]
    """
    Return an iterator over unique elements of `iterable` preserving order.

    If `key` is supplied it is used as a substitute for determining
    'uniqueness' of elements.

    Parameters
    ----------
    iterable : Iterable[T]
    key : Callable[[T], Hashable]

    Returns
    -------
    unique : Iterable[T]
    """
    seen = set()
    if key is None:
        key = lambda t: t
    for el in iterable:
        el_k = key(el)
        if el_k not in seen:
            seen.add(el_k)
            yield el


def enum_get(etype: Type[_E], name: str, default: _T1) -> Union[_E, _T1]:
    """
    Return an Enum member by `name`. If no such member exists in `etype`
    return `default`.
    """
    try:
        return etype[name]
    except LookupError:
        return default


def instance_tooltip(domain, row, skip_attrs=()):
    def show_part(_point_data, singular, plural, max_shown, _vars):
        cols = [escape('{} = {}'.format(var.name, _point_data[var]))
                for var in _vars[:max_shown + len(skip_attrs)]
                if _vars is domain.class_vars
                or var not in skip_attrs][:max_shown]
        if not cols:
            return ""
        n_vars = len(_vars)
        if n_vars > max_shown:
            cols[-1] = "... and {} others".format(n_vars - max_shown + 1)
        return \
            "<b>{}</b>:<br/>".format(singular if n_vars < 2 else plural) \
            + "<br/>".join(cols)

    parts = (("Class", "Classes", 4, domain.class_vars),
             ("Meta", "Metas", 4, domain.metas),
             ("Feature", "Features", 10, domain.attributes))
    return "<br/>".join(show_part(row, *columns) for columns in parts)


def map_rect_to(widget: QWidget, parent: QWidget, rect: QRect) -> QRect:
    """Map `rect` from `widget` to `parent` coordinate system."""
    return QRect(widget.mapTo(parent, rect.topLeft()), rect.size())


def map_rect_to_global(widget: QWidget, rect: QRect) -> QRect:
    """Map `rect` from `widget` to global screen coordinate system."""
    return QRect(widget.mapToGlobal(rect.topLeft()), rect.size())


@contextmanager
def disconnected(signal, slot, connection_type=Qt.AutoConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, connection_type)


_NamedTupleMeta = type(NamedTuple)  # type: ignore


class _DataTypeMethods:
    def __eq__(self: tuple, other):
        """Equal if `other` has the same type and all elements compare equal."""
        if type(self) is not type(other):
            return False
        return tuple.__eq__(self, other)

    def __ne__(self: tuple, other):
        return not self == other

    def __hash__(self: tuple):
        return hash((type(self), tuple.__hash__(self)))


class _DataTypeMeta(_NamedTupleMeta):
    def __new__(cls, typename, bases, ns, **kwargs):
        if ns.get('_root', False):
            return super().__new__(cls, typename, bases, ns)
        cls = super().__new__(cls, typename, bases, ns, **kwargs)
        cls.__eq__ = _DataTypeMethods.__eq__
        cls.__ne__ = _DataTypeMethods.__ne__
        cls.__hash__ = _DataTypeMethods.__hash__
        cls.__module__ = ns.get("__module__", cls.__module__)
        return cls


# Replace the DataType (alias for NamedTuple) in globals without the type
# checker being any the wiser. NamedTuple are special cased. The only way
# for type checkers to consistently apply NamedTuple aliasing is
# import ... as ...,
globals()["DataType"] = _DataTypeMeta("DataType", (), {"_root": True})


class A(DataType):
    a: str
    b: int


class B(DataType):
    a: str
    b: int


assert A("a", 1) != B("a", 1)
assert not A("a", 1) == B("a", 1)
assert hash(A("a", 1)) != hash(B("a", 1))

A(0., "a")

from dataclasses import dataclass, dataclass as datatype


class DataTypeSetializeMixin:
    def __reduce__(self) -> 'Tuple[str, tuple]':
        args = tuple(getattr(self, f) for f in self.__dataclass_fields__)
        return qname(type(self)), args

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _regtypes[qname(cls)] = cls

    def _replace(self, **kwargs):
        return self.updated(**kwargs)

    def updated(self, **kwargs):
        state = {f: getattr(self, f) for f in type(self).__dataclass_fields__}
        state.update(kwargs)
        return type(self)(**state)

    def as_dict(self):
        return {f: getattr(self, f) for f in type(self).__dataclass_fields__}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]):
        return cls(**state)


_regtypes = {}


def reconstruct(qname, args, types):
    try:
        constructor = types[qname]
    except KeyError:
        raise NameError(qname)
    return constructor(*args)


def _datatype(cls: type) -> type:
    __module__= m = cls.__module__
    __name__ = n = cls.__name__
    __qualname__ = q = cls.__qualname__

    cls = dataclass(frozen=True, unsafe_hash=True)(cls)
    # cls.__reduce__ = DataTypeSetializeMixin.__reduce__
    cls._replace = DataTypeSetializeMixin._replace
    cls.updated = DataTypeSetializeMixin.updated
    cls.as_dict = DataTypeSetializeMixin.as_dict
    cls.from_dict = DataTypeSetializeMixin.from_dict

    return cls
    # cls.__init_subclass__
    # class DataType(DataTypeSetializeMixin, cls):
    #     # pass
    #     __module__ = m
    #     __qualname__ = q
    #     __name__ = n
    # DataType.__name__ = __name__
    # DataType.__qualname__ = __qualname__
    # DataType.__name__ = __name__
    return DataType


def datatype_deserialize(qname, args, types):
    return reconstruct(qname, args, types)


def datatype_deconstruct(obj):
    return qname(type(obj)), tuple(obj.as_dict().values())
    # rc, type, *args = obj.__reduce__()
    # if rc is copyreg.__newobj__:
    #     return qname(type), args[0]


globals()["datatype"] = _datatype
datatype.deserialize = datatype_deserialize
datatype.deconstruct = datatype_deconstruct
datatype: Callable[[type], type]

@datatype
class A:
    a: str
    b: int


@dataclass(frozen=True, unsafe_hash=True)
class B:
    a: str
    b: int


assert A("a", 1) != B("a", 1)
assert not A("a", 1) == B("a", 1)
# assert hash(A("a", 1)) != hash(B("a", 1))

a = A("1", 3).a
A(0., "1")
B(0., "1")
