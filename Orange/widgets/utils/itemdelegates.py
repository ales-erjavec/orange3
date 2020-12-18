import math
from typing import Optional, Tuple

import numpy as np

from AnyQt.QtCore import QModelIndex, QSize
from AnyQt.QtWidgets import QStyle, QStyleOptionViewItem, QStyledItemDelegate, QApplication

from Orange.misc.cache import LRUCache
from orangewidget.gui import OrangeUserRole

NumberTypes = (int, float, np.floating, np.integer)


class FixedFormatNumericColumnDelegate(QStyledItemDelegate):
    """
    A numeric delegate displaying in a fixed format.

    Parameters
    ----------
    ndecimals: int
        The number of decimals in the display
    ndigits: int
        The max number of digits in the integer part. If the model returns
        `ColumnDataSpanRole` data for a column then the `ndigits` is derived
        from that.

        .. note:: This is only used for size hinting.

    """
    ColumnDataSpanRole = next(OrangeUserRole)

    def __init__(self, *args, ndecimals=3, ndigits=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.ndecimals = ndecimals
        self.ndigits = ndigits
        self.__sh_cache = LRUCache(maxlen=200)

    def displayText(self, value, locale) -> str:
        if isinstance(value, NumberTypes):
            return f"{value:.{self.ndecimals}f}"
        return super().displayText(value, locale)

    def spanData(self, index: QModelIndex) -> Optional[Tuple[float, float]]:
        """
        Return the min, max numeric data values in the column that `index`
        is in.
        """
        model = index.model()
        span = model.data(index, self.ColumnDataSpanRole)
        try:
            min, max = span
        except (ValueError, TypeError):
            return None
        if isinstance(min, NumberTypes) and isinstance(max, NumberTypes):
            return float(min), float(max)
        else:
            return None

    def template(self, value: float, ndecimals=3) -> str:
        sign = math.copysign(1., value)
        # number of digits (integer part)
        ndigits = int(math.ceil(math.log10(abs(value) + 1)))
        template = "X" * ndigits + "." + "X" * ndecimals
        if sign == -1.:
            template = "-" + template
        return template

    def sizeHint(
            self, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QSize:
        widget = option.widget
        template = self.template(-10 ** self.ndigits, self.ndecimals)
        span = self.spanData(index)
        if span is not None:
            vmin, vmax = span
            t1, t2 = self.template(vmin, self.ndecimals), self.template(vmax, self.ndecimals)
            template = max((t1, t2), key=len)
        style = widget.style() if widget is not None else QApplication.style()
        # Keep ref to style wrapper. This is ugly, wrong but the wrapping of
        # C++ QStyle instance takes ~5% unless the wrapper already exists.
        self.__style = style
        opt = QStyleOptionViewItem(option)
        opt.features |= QStyleOptionViewItem.HasDisplay
        sh = QSize()
        key = option.font.key(), template
        if key not in self.__sh_cache:
            for d in map(str, range(10)):
                opt.text = template.replace("X", d)
                sh_ = style.sizeFromContents(
                    QStyle.CT_ItemViewItem, opt, QSize(), widget)
                sh = sh.expandedTo(sh_)
            self.__sh_cache[key] = sh
        else:
            sh = self.__sh_cache[key]
        return QSize(sh)
