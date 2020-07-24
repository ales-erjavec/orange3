import enum
import inspect
import sys
from collections import deque, Hashable
from typing import (
    TypeVar, Callable, Any, Iterable, Type, Union, Optional, NamedTuple
)
from typing import NamedTuple as DataType

from AnyQt.QtCore import QObject

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
    return set([int(i*count/float(iterations)) for i in range(iterations)])


def getdeepattr(obj, attr, *arg, **kwarg):
    if isinstance(obj, dict):
        return obj.get(attr)
    return deepgetattr(obj, attr, *arg, **kwarg)


def to_html(str):
    return str.replace("<=", "&#8804;").replace(">=", "&#8805;").\
        replace("<", "&#60;").replace(">", "&#62;").replace("=\\=", "&#8800;")

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


def apply_all(seq, op):
    # type: (Iterable[_T1], Callable[[_T1], Any]) -> None
    """Apply `op` on all elements of `seq`."""
    # from itertools recipes `consume`
    deque(map(op, seq), maxlen=0)


def enum_get(etype: Type[_E], name: str, default: _T1) -> Union[_E, _T1]:
    """
    Return an Enum member by `name`. If no such member exists in `etype`
    return `default`.
    """
    try:
        return etype[name]
    except LookupError:
        return default


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
