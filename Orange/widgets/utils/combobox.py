from AnyQt.QtCore import Qt
from AnyQt.QtGui import QBrush, QColor, QPalette, QPen, QFont, QFontMetrics
from AnyQt.QtWidgets import QStylePainter, QStyleOptionComboBox, QStyle

from orangewidget.utils.combobox import ComboBoxSearch, ComboBox

__all__ = [
    "ComboBoxSearch", "ComboBox", "ItemStyledComboBox"
]


class ItemStyledComboBox(ComboBox):
    """
    A QComboBox that draws its text using current item's foreground and font
    role.

    Note
    ----
    Stylesheets etc. can completely ignore this.
    """
    def paintEvent(self, event) -> None:
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        foreground = self.currentData(Qt.ForegroundRole)
        if isinstance(foreground, (QBrush, QColor)):
            foreground = QBrush(foreground)
            if foreground.style() != Qt.NoBrush:
                # some styles take WindowText some just use current pen?
                option.palette.setBrush(QPalette.WindowText, foreground)
                option.palette.setBrush(QPalette.ButtonText, foreground)
                option.palette.setBrush(QPalette.Text, foreground)
                painter.setPen(QPen(foreground, painter.pen().widthF()))
        font = self.currentData(Qt.FontRole)
        if isinstance(font, QFont):
            option.fontMetrics = QFontMetrics(font)
            painter.setFont(font)
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)
