from typing import Optional, Union, Any, Iterable, List

from AnyQt.QtCore import Qt, QPointF, QRectF, QSizeF, QEvent, QMarginsF
from AnyQt.QtGui import (
    QStaticText, QFont, QPalette, QPainter, QTransform, QFontMetrics,
    QPaintEngine, QPixmap
)
from AnyQt.QtWidgets import (
    QWidget, QSizePolicy, QStyleOptionGraphicsItem, QGraphicsObject,
    QGraphicsWidget, QGraphicsItemGroup, QGraphicsItem, QGraphicsScene,
    QGraphicsLayoutItem, QWIDGETSIZE_MAX,
)

__all__ = [
    'StaticTextItem', 'TextListWidget', 'SimpleLayoutItem',
    'GraphicsPixmapWidget'
]


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


class TextListWidget(QGraphicsWidget):
    """
    A linear text list widget.

    Displays a list of uniformly spaced text lines.

    Parameters
    ----------
    parent: Optional[QGraphicsItem]
    items: Iterable[str]
    alignment: Qt.Alignment
    orientation: Qt.Orientation
    """
    def __init__(
            self,
            parent: Optional[QGraphicsItem] = None,
            items: Iterable[str] = (),
            alignment: Union[Qt.AlignmentFlag, Qt.Alignment] = Qt.AlignLeading,
            orientation: Qt.Orientation = Qt.Vertical,
            **kwargs: Any
    ) -> None:
        sizePolicy = kwargs.pop("sizePolicy", None)  # type: Optional[QSizePolicy]
        super().__init__(None, **kwargs)
        self.setFlag(QGraphicsWidget.ItemClipsChildrenToShape, True)
        self.__items: List[str] = []
        self.__textitems: List[StaticTextItem] = []
        self.__group: Optional[QGraphicsItemGroup] = None
        self.__spacing = 0
        self.__alignment = Qt.AlignmentFlag(alignment)
        self.__orientation = orientation

        if items is not None:
            self.setItems(items)

        if sizePolicy is not None:
            self.setSizePolicy(sizePolicy)

        if parent is not None:
            self.setParentItem(parent)

    def setItems(self, items: Iterable[str]) -> None:
        """
        Set items for display

        Parameters
        ----------
        items: Iterable[str]
        """
        self.__clear()
        self.__items = list(items)
        self.__setup()
        self.__layout()
        self.updateGeometry()

    def setAlignment(self, alignment: Qt.AlignmentFlag) -> None:
        """
        Set the text item's alignment.
        """
        if self.__alignment != alignment:
            self.__alignment = alignment
            self.__layout()

    def alignment(self) -> Qt.AlignmentFlag:
        """Return the text item's alignment."""
        return self.__alignment

    def setOrientation(self, orientation: Qt.Orientation) -> None:
        """
        Set text orientation.

        If Qt.Vertical items are put in a vertical layout
        if Qt.Horizontal the n items are drawn rotated 90 degrees and laid out
        horizontally with first text items's top corner in the bottom left
        of `self.geometry()`.

        Parameters
        ----------
        orientation: Qt.Orientation
        """
        if self.__orientation != orientation:
            self.__orientation = orientation
            self.__layout()

    def orientation(self) -> Qt.Orientation:
        return self.__orientation

    def clear(self) -> None:
        """
        Remove all items.
        """
        self.__clear()
        self.__items = []
        self.updateGeometry()

    def count(self) -> int:
        """
        Return the number of items
        """
        return len(self.__items)

    def sizeHint(self, which: Qt.SizeHint, constraint=QSizeF()) -> QSizeF:
        """Reimplemented."""
        if which == Qt.PreferredSize:
            sh = self.__naturalsh()
            if self.__orientation == Qt.Vertical:
                if 0 < constraint.height() < sh.height():
                    sh = scaled(sh, constraint, Qt.KeepAspectRatioByExpanding)
            else:
                sh = sh.transposed()
                if 0 < constraint.width() < sh.width():
                    sh = scaled(sh, constraint, Qt.KeepAspectRatioByExpanding)
        else:
            sh = super().sizeHint(which, constraint)
        return sh

    def __naturalsh(self) -> QSizeF:
        """Return the natural size hint (preferred sh with no constraints)."""
        fm = QFontMetrics(self.font())
        spacing = self.__spacing
        N = len(self.__items)
        width = max((fm.width(text) for text in self.__items),
                    default=0)
        height = N * fm.height() + max(N - 1, 0) * spacing
        return QSizeF(width, height)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.LayoutRequest:
            self.__layout()
        elif event.type() == QEvent.GraphicsSceneResize:
            self.__layout()
        elif event.type() == QEvent.ContentsRectChange:
            self.__layout()
        return super().event(event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.FontChange:
            self.updateGeometry()
            font = self.font()
            for item in self.__textitems:
                item.setFont(font)
        elif event.type() == QEvent.PaletteChange:
            palette = self.palette()
            for item in self.__textitems:
                item.setPalette(palette)
        super().changeEvent(event)

    def __setup(self) -> None:
        self.__clear()
        font = self.font()
        assert self.__group is None
        group = QGraphicsItemGroup()

        for text in self.__items:
            t = StaticTextItem(text, group)
            t.setFont(font)
            t.setToolTip(text)
            self.__textitems.append(t)
        group.setParentItem(self)
        self.__group = group

    def __layout(self) -> None:
        margins = QMarginsF(*self.getContentsMargins())
        if self.__orientation == Qt.Horizontal:
            # transposed margins
            margins = QMarginsF(
                margins.bottom(), margins.left(), margins.top(), margins.right()
            )
            crect = self.rect().transposed().marginsRemoved(margins)
        else:
            crect = self.rect().marginsRemoved(margins)

        spacing = self.__spacing

        align_horizontal = self.__alignment & Qt.AlignHorizontal_Mask
        align_vertical = self.__alignment & Qt.AlignVertical_Mask
        if align_vertical == 0:
            align_vertical = Qt.AlignTop
        if align_horizontal == 0:
            align_horizontal = Qt.AlignLeft

        N = len(self.__items)

        if not N:
            return

        assert self.__group is not None
        fm = QFontMetrics(self.font())
        naturalheight = fm.height()
        cell_height = (crect.height() - (N - 1) * spacing) / N

        if cell_height > naturalheight and N > 1:
            cell_height = naturalheight
            spacing = (crect.height() - N * cell_height) / N

        advance = cell_height + spacing

        if align_vertical == Qt.AlignTop:
            align_dy = 0.
        elif align_vertical == Qt.AlignVCenter:
            align_dy = advance / 2.0 - naturalheight / 2.0
        else:
            align_dy = advance - naturalheight

        if align_horizontal == Qt.AlignLeft:
            for i, item in enumerate(self.__textitems):
                item.setPos(crect.left(), crect.top() + i * advance + align_dy)
        elif align_horizontal == Qt.AlignHCenter:
            for i, item in enumerate(self.__textitems):
                item.setPos(
                    crect.center().x() - item.boundingRect().width() / 2,
                    crect.top() + i * advance + align_dy
                )
        else:
            for i, item in enumerate(self.__textitems):
                item.setPos(
                    crect.right() - item.boundingRect().width(),
                    crect.top() + i * advance + align_dy
                )

        if self.__orientation == Qt.Vertical:
            self.__group.setRotation(0)
            self.__group.setPos(0, 0)
        else:
            self.__group.setRotation(-90)
            self.__group.setPos(self.rect().bottomLeft())

    def __clear(self) -> None:
        def remove(items: Iterable[QGraphicsItem],
                   scene: Optional[QGraphicsScene]):
            for item in items:
                if scene is not None:
                    scene.removeItem(item)
                else:
                    item.setParentItem(None)

        self.__textitems = []
        if self.__group is not None:
            remove([self.__group], self.scene())
            self.__group = None


class SimpleLayoutItem(QGraphicsLayoutItem):
    """
    A graphics layout item wrapping a QGraphicsItem instance to be
    managed by a layout.

    The item is positioned at the layout geometry top left corner and its
    boundingRect().size() is used as the preferred size hint

    Parameters
    ----------
    item: QGraphicsItem
    parent: Optional[QGraphicsLayoutItem]
        The parent layout item.
    anchor: Tuple[float, float]
        The anchor in this layout item's geometry relative coord. system
        (0, 0) corresponds to top left corner and (1, 1) corresponds to
        bottom right corner).
    anchorItem: Tuple[float, float]
        The relative anchor in `item` 's bounding rect.
    """
    __slots__ = (
        "__anchorThis",
        "__anchorItem",
        "item"
    )

    def __init__(
            self,
            item: QGraphicsItem,
            parent: Optional[QGraphicsLayoutItem] = None,
            anchor=(0., 0.),
            anchorItem=(0., 0.),
            **kwargs
    ) -> None:
        sizePolicy: Optional[QSizePolicy] = kwargs.pop("sizePolicy", None)
        super().__init__(parent, **kwargs)
        self.__anchorThis = anchor
        self.__anchorItem = anchorItem
        self.item = item
        if sizePolicy is not None:
            self.setSizePolicy(sizePolicy)
        self.__layout()

    def setGeometry(self, rect: QRectF) -> None:
        super().setGeometry(rect)
        self.__layout()

    def sizeHint(self, which: Qt.SizeHint, constraint=QSizeF(-1, -1)) -> QSizeF:
        if which == Qt.PreferredSize:
            brect = self.item.boundingRect()
            brect = self.item.mapRectToParent(brect)
            return brect.size()
        else:
            return QSizeF()

    def updateGeometry(self):
        super().updateGeometry()
        parent = self.parentLayoutItem()
        if parent is not None:
            parent.updateGeometry()

    def __layout(self):
        item = self.item
        geom = self.geometry()
        margins = QMarginsF(*self.getContentsMargins())
        crect = geom.marginsRemoved(margins)
        anchorpos = qrect_pos_relative(crect, *self.__anchorThis)
        brect = self.item.boundingRect()
        anchorpositem = qrect_pos_relative(brect, *self.__anchorItem)
        anchorpositem = item.mapToParent(anchorpositem)
        item.setPos(item.pos() + (anchorpos - anchorpositem))


def qrect_pos_relative(rect: QRectF, rx: float, ry: float) -> QPointF:
    return QPointF(rect.x() + rect.width() * rx, rect.y() + rect.height() * ry)


class GraphicsPixmapWidget(QGraphicsWidget):
    def __init__(
            self,
            parent: Optional[QGraphicsItem] = None,
            pixmap: Optional[QPixmap] = None,
            scaleContents=False,
            aspectMode=Qt.KeepAspectRatio,
            **kwargs
    ) -> None:
        self.__scaleContents = scaleContents
        self.__aspectMode = aspectMode
        self.__pixmap = QPixmap(pixmap) if pixmap is not None else QPixmap()
        super().__init__(None, **kwargs)
        self.setFlag(QGraphicsWidget.ItemUsesExtendedStyleOption, True)
        self.setContentsMargins(0, 0, 0, 0)
        if parent is not None:
            self.setParentItem(parent)

    def setPixmap(self, pixmap: QPixmap) -> None:
        self.prepareGeometryChange()
        self.__pixmap = QPixmap(pixmap)
        self.updateGeometry()

    def pixmap(self) -> QPixmap:
        return QPixmap(self.__pixmap)

    def setAspectRatioMode(self, mode: Qt.AspectRatioMode) -> None:
        if self.__aspectMode != mode:
            self.__aspectMode = mode
            self.updateGeometry()

    def aspectRatioMode(self) -> Qt.AspectRatioMode:
        return self.__aspectMode

    def setScaleContents(self, scale: bool) -> None:
        if self.__scaleContents != scale:
            self.__scaleContents = bool(scale)
            self.updateGeometry()
            self.__updateScale()

    def scaleContents(self) -> bool:
        return self.__scaleContents

    def sizeHint(self, which, constraint=QSizeF(-1, -1)) -> QSizeF:
        if which == Qt.PreferredSize:
            sh = QSizeF(self.__pixmap.size())
            if self.__scaleContents:
                sh = scaled(sh, constraint, self.__aspectMode)
            return sh
        elif which == Qt.MinimumSize:
            if self.__scaleContents:
                return QSizeF(0, 0)
            else:
                return QSizeF(self.__pixmap.size())
        elif which == Qt.MaximumSize:
            if self.__scaleContents:
                return QSizeF()
            else:
                return QSizeF(self.__pixmap.size())
        else:
            # Qt.MinimumDescent
            return QSizeF()

    def pixmapTransform(self) -> QTransform:
        if self.__pixmap.isNull():
            return QTransform()

        pxsize = QSizeF(self.__pixmap.size())
        crect = self.contentsRect()
        transform = QTransform()
        transform = transform.translate(crect.left(), crect.top())

        if self.__scaleContents:
            csize = scaled(pxsize, crect.size(), self.__aspectMode)
        else:
            csize = pxsize

        xscale = csize.width() / pxsize.width()
        yscale = csize.height() / pxsize.height()

        return transform.scale(xscale, yscale)

    def paint(
            self, painter: QPainter, option: QStyleOptionGraphicsItem,
            widget: Optional[QWidget] = None
    ) -> None:
        if self.__pixmap.isNull():
            return
        pixmap = self.__pixmap
        crect = self.contentsRect()

        exposed = option.exposedRect
        exposedcrect = crect.intersected(exposed)
        pixmaptransform = self.pixmapTransform()
        # map exposed rect to exposed pixmap coords
        assert pixmaptransform.type() <= QTransform.TxRotate
        pixmaptransform, ok = pixmaptransform.inverted()
        engine = painter.paintEngine()
        export = engine.type() in (QPaintEngine.Pdf, QPaintEngine.SVG)
        if export:
            # If exporting to vector file formats, pre(up)scale the image.
            # QtSvg is missing 'image-rendering' attribute (QTBUG-4145)
            # Don't know how PDF 'hints' raster scaling algorithm.
            pixmap = pixmap.scaled(
                crect.toAlignedRect().size(), Qt.IgnoreAspectRatio,
                transformMode=Qt.FastTransformation
            )
        if not ok or export:
            painter.drawPixmap(
                crect, pixmap, QRectF(QPointF(0, 0), QSizeF(pixmap.size()))
            )
            return

        exposedpixmap = pixmaptransform.mapRect(exposed)
        painter.drawPixmap(exposedcrect, pixmap, exposedpixmap)


def scaled(size, constraint, mode=Qt.KeepAspectRatio):
    # type: (QSizeF, QSizeF, Qt.AspectRatioMode) -> QSizeF
    """
    Return size scaled to fit in the constrains using the aspect mode `mode`.

    If  width or height of constraint are 0 or negative they are ignored,
    ie. the result is not constrained in that dimension.
    """
    size, constraint = QSizeF(size), QSizeF(constraint)
    if constraint.width() < 0 and constraint.height() < 0:
        return size

    if mode == Qt.IgnoreAspectRatio:
        if constraint.width() >= 0:
            size.setWidth(constraint.width())
        if constraint.height() >= 0:
            size.setHeight(constraint.height())
    elif mode == Qt.KeepAspectRatio:
        if constraint.width() < 0:
            constraint.setWidth(QWIDGETSIZE_MAX)
        if constraint.height() < 0:
            constraint.setHeight(QWIDGETSIZE_MAX)
        size.scale(constraint, mode)
    elif mode == Qt.KeepAspectRatioByExpanding:
        if constraint.width() < 0:
            constraint.setWidth(0)
        if constraint.height() < 0:
            constraint.setHeight(0)
        size.scale(constraint, mode)
    return size
