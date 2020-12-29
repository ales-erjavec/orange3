import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from AnyQt.QtCore import QModelIndex, QSize, Qt, QPointF
from AnyQt.QtGui import QStaticText, QPainter, QTransform
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
        if isinstance(value, _Number):
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


_Real = (float, np.float64, np.float32, np.float16)
_Integral = (int, np.integer)
_Number = _Integral + _Real
_String = (str, np.str_)

isnan = math.isnan


class DataDelegate(QStyledItemDelegate):
    """
    A QStyledItemDelegate optimized for displaying fixed tabular data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        @lru_cache(maxsize=100 * 200)
        def sttext(text: str) -> QStaticText:
            return QStaticText(text)
        self.__static_text_cache = sttext
        self.__static_text_lru_cache = LRUCache(100 * 200)

    def displayText(self, value, locale):
        if isinstance(value, _Integral):
            return super().displayText(int(value), locale)
        elif isinstance(value, _Real):
            if isnan(value):
                return "N/A"
            else:
                super().displayText(float(value), locale)
        elif isinstance(value, _String):
            return str(value)
        return super().displayText(value, locale)

    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        super().initStyleOption(option, index)
        model = index.model()
        v = model.data(index, Qt.DisplayRole)
        if isinstance(v, _Number):
            option.displayAlignment = \
                (option.displayAlignment & ~Qt.AlignHorizontal_Mask) | \
                Qt.AlignRight

    def paint(self, painter, option, index):
        # type: (QPainter, QStyleOptionViewItem, QModelIndex) -> None
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        if widget is not None:
            style = widget.style()
        else:
            style = QApplication.style()
        self.__style = style
        text = opt.text
        opt.text = ""
        trect = style.subElementRect(QStyle.SE_ItemViewItemText, opt, widget)
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
        # text margin (as in QCommonStylePrivate::viewItemDrawText)
        margin = style.pixelMetric(QStyle.PM_FocusFrameHMargin, None, widget) + 1
        trect = trect.adjusted(margin, 0, -margin, 0)
        opt.text = text
        if opt.textElideMode != Qt.ElideNone:
            st = self.__static_text_elided_cache(opt, trect.width())
        else:
            st = self.__static_text_cache(text)
        tsize = st.size()
        textalign = opt.displayAlignment
        text_pos_x = text_pos_y = 0.0

        if textalign & Qt.AlignLeft:
            text_pos_x = trect.left()
        elif textalign & Qt.AlignRight:
            text_pos_x = trect.x() + trect.width() - tsize.width()
        elif textalign & Qt.AlignHCenter:
            text_pos_x = trect.center().x() - tsize.width() / 2

        if textalign & Qt.AlignTop:
            text_pos_y = trect.top()
        elif textalign & Qt.AlignBottom:
            text_pos_y = trect.top() + trect.height() - tsize.height()
        elif textalign & Qt.AlignVCenter:
            text_pos_y = trect.center().y() - tsize.height() / 2

        painter.setFont(opt.font)
        painter.drawStaticText(QPointF(text_pos_x, text_pos_y), st)

    def __static_text_elided_cache(
            self, option: QStyleOptionViewItem, width: int) -> QStaticText:
        """
        Return a QStaticText instance for depicting the text of the `option`
        item.
        """
        key = option.text, option.font.key(), option.textElideMode, width
        try:
            st = self.__static_text_lru_cache[key]
        except KeyError:
            fm = option.fontMetrics
            text = fm.elidedText(option.text, option.textElideMode, width)
            st = QStaticText(text)
            st.prepare(QTransform(), option.font)
            self.__static_text_lru_cache[key] = st
        return st
