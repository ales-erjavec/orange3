from typing import Optional, Union, Any

from AnyQt.QtCore import Qt, QPointF, QRectF
from AnyQt.QtGui import QStaticText, QFont, QPalette, QPainter, QTransform
from AnyQt.QtWidgets import QGraphicsObject, QWidget, QStyleOptionGraphicsItem


class StaticTextItem(QGraphicsObject):
    """
    A text graphics object for displaying text in a QGraphicsScene.

    This class uses QStaticText for text rendering.
    """
    __slots__ = (
        '__boundingRect',
        '__staticText',
        '__palette',
        '__anchor',
        '__font',
    )

    def __init__(self, text, *args, **kwargs):
        # type: (Union[str, QStaticText], Any, Any) -> None
        self.__boundingRect = None             # type: Optional[QRectF]
        if isinstance(text, str):
            text = QStaticText(text)
            text.setTextFormat(Qt.PlainText)
        self.__staticText = QStaticText(text)  # type: QStaticText
        self.__palette = None                  # type: Optional[QPalette]
        self.__anchor = QPointF(0.0, 0.0)      # type: QPointF
        self.__font = QFont(kwargs.pop("font", QFont()))  # type: QFont
        anchor = kwargs.pop("anchor", self.__anchor)
        if isinstance(anchor, tuple):
            anchor = QPointF(*anchor)
        self.__anchor = QPointF(anchor)

        super().__init__(*args, **kwargs)

    def setText(self, text):  # type: (str) -> None
        """Set the text to display"""
        if text != self.__staticText.text():
            self.prepareGeometryChange()
            self.__staticText.setText(text)

    def text(self):  # type: () -> str
        """Return the displayed text"""
        return self.__staticText.text()

    def setStaticText(self, text):  # type: (QStaticText) -> None
        """Set the QStaticText for display"""
        if text != self.__staticText:
            self.prepareGeometryChange()
            self.__staticText = QStaticText(text)

    def staticText(self):  # type: () -> QStaticText
        """Return the static text"""
        return QStaticText(self.__staticText)

    def setFont(self, font):  # type: (QFont) -> None
        """Set the font for this item."""
        if font != self.__font:
            self.prepareGeometryChange()
            self.__font = QFont(font)

    def font(self):  # type: () -> QFont
        """Return the font for this item."""
        return QFont(self.__font)

    def setAnchor(self, rx, ry):  # type: (float, float) -> None
        """
        Set the item anchor position in relative object bounding coordinates.

        The default (0.0, 0.0) positions the top left corner of the text at
        `pos()`, (0.5, 0.5) would center the text both vertically and
        horizontally, ...
        """
        anchor = QPointF(rx, ry)
        if anchor != self.__anchor:
            self.prepareGeometryChange()
            self.__anchor = anchor

    def anchor(self):  # type: () -> QPointF
        """
        Return the anchor position.
        """
        return QPointF(self.__anchor)

    def setPalette(self, palette):  # type: (QPalette) -> None
        """
        Set the palette.

        The palette's `Text` color role is used for the default text color.
        If a palette is not set then the default palette is used.
        """
        if palette != self.__palette:
            self.__palette = QPalette(palette)
            self.update()

    def palette(self):  # type: () -> QPalette
        """
        Return the palette.
        """
        if self.__palette is None:
            return QPalette()
        else:
            return QPalette(self.__palette)

    def boundingRect(self):  # type: () -> QRectF
        if self.__boundingRect is None:
            # QStaticText.size is the last transformed/font draw size.
            # We need the size in identity transform with current font.
            st = QStaticText(self.__staticText)
            st.prepare(QTransform(), self.__font)
            size = st.size()
            x = -size.width() * self.__anchor.x()
            y = -size.height() * self.__anchor.y()
            self.__boundingRect = QRectF(x, y, size.width(), size.height())
        return QRectF(self.__boundingRect)

    def prepareGeometryChange(self):  # type: () -> None
        super().prepareGeometryChange()
        self.__boundingRect = None

    def paint(self, painter, option, widget=None):
        # type: (QPainter, QStyleOptionGraphicsItem, QWidget) -> None
        if self.__palette is not None:
            palette = self.__palette
        else:
            palette = option.palette
        br = self.boundingRect()
        painter.save()
        painter.setFont(self.__font)
        painter.setPen(palette.color(QPalette.Text))
        painter.drawStaticText(br.topLeft(), self.__staticText)
        painter.restore()
