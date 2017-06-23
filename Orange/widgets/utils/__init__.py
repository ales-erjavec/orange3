import inspect
import sys
import enum
from collections import deque

import typing
from typing import TypeVar, Deque, Callable, Any, Iterable, Optional, Type

from AnyQt.QtCore import QObject
from AnyQt.QtGui import QColor, QFont

from Orange.data.variable import TimeVariable
from Orange.util import deepgetattr

if typing.TYPE_CHECKING:
    H = typing.TypeVar("H", bound=typing.Hashable)
    E = typing.TypeVar("E", bound=enum.Enum)


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



_T1 = TypeVar("_T1")


def apply_all(seq, op):
    # type: (Iterable[_T1], Callable[[_T1], Any]) -> None
    """Apply `op` on all elements of `seq`."""
    # from itertools recipes `consume`
    deque(map(op, seq), maxlen=0)


def qfont_adjust_size(font, dsize):
    # type: (QFont, int) -> QFont
    """
    Return a font with adjusted size.

    `dsize` is added to the font's size units (i.e. pixel size or points size
    depending on which one is specified).
    """
    font = QFont(font)
    psize = font.pointSize()
    pxsize = font.pixelSize()
    if psize != -1:
        font.setPointSize(psize + dsize)
    elif pxsize != -1:
        font.setPixelSize(pxsize + dsize)
    return font


def qcolor_alpha(color, alpha):
    # type: (QColor, int) -> QColor
    """
    Return a copy of `color` with alpha channel set to `alpha`
    """
    color = QColor(color)
    color.setAlpha(alpha)
    return color


def unique(iterable):
    # type: (Iterable[H]) -> Iterable[H]
    """
    Return an iterator of unique elements in `iterable` while preserving the
    order they are encountered in.

    The elements of `iterable` must be hashable.

    Parameters
    ----------
    iterable : Iterable[Hashable]

    Returns
    -------
    unique : Iterable[Hashable]
       Unique elements from `iterable`.
    """
    seen = set([])

    def first_seen(e):
        isfirst = e not in seen
        if isfirst:
            seen.add(e)
        return isfirst
    return (e for e in iterable if first_seen(e))


if typing.TYPE_CHECKING:
    @typing.overload
    def enum_lookup(enumtype: Type[E], name: str) -> Optional[E]: ...

    @typing.overload
    def enum_lookup(enumtype: Type[E], name: str, default: E) -> E: ...


def enum_lookup(enumtype, name, default=None):
    """
    Return an value from `enumtype` by its symbolic `name`.

    `default` is returned if the value is not found.
    """
    try:
        return enumtype[name]
    except LookupError:
        return default

