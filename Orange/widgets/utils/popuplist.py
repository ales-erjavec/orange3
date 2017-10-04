import enum

from AnyQt.QtCore import Qt, QSize, QRect, QAbstractItemModel, QModelIndex
from AnyQt.QtGui import QIcon, QPainter, QStandardItemModel
from AnyQt.QtWidgets import QFrame, QStyle, QStyleOptionViewItem, QAbstractItemDelegate
from AnyQt.QtCore import pyqtSlot as Slot, pyqtSignal as Signal


#: A clickable widget disclosing extra contents in a popup.
#:
#: .. image:: PopupButton.png
#:

class _PopupButton(QFrame):
    """
    """
    class Style(enum.IntEnum):
        ...

    class Direction(enum.IntEnum):
        ...

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__model = QStandardItemModel(self)  # type: QAbstractItemModel
        self.__deleteModel = True
        self.__delegate = None
        self.__deleteDelegate = True
        self.__iconSize = QSize()

    @Slot()
    def __invalidate(self):
        self.__current = None
        self.updateGeometry()
        self.update()

    @Slot()
    def __destroyed(self):
        self.__current = None
        self.__model = None
        self.updateGeometry()
        self.update()
        if self.__popup is not None:
            self.__deletePopup()

    def setModel(self, model):
        # type: (QAbstractItemModel) -> None
        """
        Set the message source model.

        Parameters
        ----------
        model : QAbstractItemModel
        """
        if self.__model is model:
            return

        if self.__model is not None:
            self.__model.dataChanged.disconnect(self.__invalidate)
            self.__model.rowsInserted.disconnect(self.__invalidate)
            self.__model.rowsRemoved.disconnect(self.__invalidate)
            self.__model.modelReset.disconnect(self.__invalidate)
            self.__model.destroyed.disconnect(self.__destroyed)

        self.__model = model

        model.dataChanged.connect(self.__invalidate, Qt.UniqueConnection)
        model.rowsInserted.connect(self.__invalidate, Qt.UniqueConnection)
        model.rowsRemoved.connect(self.__invalidate, Qt.UniqueConnection)
        model.modelReset.connect(self.__invalidate, Qt.UniqueConnection)
        model.destroyed.connect(self.__destroyed, Qt.UniqueConnection)

        self.updateGeometry()
        self.__update()
        self.update()

    def model(self):
        # type: () -> QAbstractItemModel
        # if self.__model is None:
        #     model = QStandardItemModel(0, 1, self)
        #     self.setModel(model)
        return self.__model

    def setItemDelegate(self, delegate):
        # type: (QAbstractItemDelegate) -> None
        if self.__delegate is delegate:
            return
        if self.__deleteDelegate and self.__delegate.parent() is self:
            self.__delegate.deleteLater()
            self.__delegate = None

        self.__delegate = delegate

        self.update()

    def count(self):
        # type: () -> int
        """
        Return the number of display items.
        """
        if self.__model is not None:
            return self.__model.rowCount()
        else:
            return 0

    def iconSize(self):
        if self.__iconSize.isNull():
            s = self.style().pixelMetric(QStyle.PM_SmallIconSize)
            return QSize(s, s)
        else:
            return QSize(self.__iconSize)

    def setIconSize(self, size):
        # type: (QSize) -> None
        if self.__iconSize != size:
            self.__iconSize = QSize(size)
            self.updateGeometry()
            self.update()

    def setItemData(self, index, value, role=Qt.UserRole):
        self.__model.setData(
            self.__model.index(index), value, role
        )

    def itemData(self, index, role=Qt.UserRole):
        return self.__model.data(
            self.__model.index(index, role)
        )

    def showPopup(self):
        ...

    def initStyleOption(self, option):
        # type: (QStyleOptionViewItem) -> None
        option.initFrom(self)
        option.text = self.displayText()
        option.iconSize = self.iconSize()
        option.icon = self.displayIcon()
        if self.hasPopupContents():
            option.text += " \N{VERTICAL ELLIPSIS}"

    def paintEvent(self, event):
        opt = QStyleOptionViewItem()
        self.initStyleOption(opt)
        rect = opt.rect

        painter = QPainter(self)
        painter.setFont(opt.font)
        margin = 0
        icontextmargin = 2
        iconrect = QRect()
        textrect = QRect()
        indicator_rect = QRect()

        if not opt.icon.isNull():
            mode = QIcon.Active if opt.state & QStyle.State_Active else QIcon.Disabled
            opt.icon.paint(painter, iconrect, Qt.AlignCenter, mode, QIcon.Off)
            textrect.setLeft(iconrect.right() + icontextmargin)
        fm = opt.fontMetrics  # type: QFontMetrics
        if text:
            text = fm.elidedText(text, Qt.ElideRight, textrect.width())
        painter.drawText()


from AnyQt.QtWidgets import QPushButton, QStyleOptionButton, QMenu, QApplication
from AnyQt.QtWidgets import QAbstractButton
from AnyQt.QtGui import QFontMetrics

#: A clickable widget disclosing extra contents in a popup.
#:
#: .. image:: PopupButton.png
#:


class PopupButton(QAbstractButton):
    Up, Down = 1, 2

    def __init__(self):
        super().__init__()
        self.__popup = None  #: Optional[QWidget]
        self.__text = ""
        self.__textFormat = Qt.PlainText

    def setPopup(self, popup):
        self.__popup = popup

    def popup(self):
        return self.__popup

    def showPopup(self):
        self.__popup.show()

    def hidePopup(self):
        self.__popup.hide()  # or close? have interface like

    def _displayText(self):
        return self.text()

    def initStyleOption(self, opt):
        # type: (QStyleOptionButton) -> None
        opt.initFrom(self)
        opt.text = self._displayText().replace("&", "&&")
        opt.icon = self.icon()
        opt.iconSize = self.iconSize()
        opt.features = QStyleOptionButton.None_
        if self.__popup is not None:
            opt.features |= QStyleOptionButton.HasMenu

        if self.isDown():
            opt.state |= QStyle.State_Sunken
        else:
            opt.state |= QStyle.State_Raised

    def sizeHint(self):
        self.ensurePolished()
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        style = self.style()  # type: QStyle
        fm = opt.fontMetrics  # type: QFontMetrics
        iconsize = opt.iconSize  # type: QSize

        size = QSize(0, 0)
        if not opt.icon.isNull():
            size.setWidth(iconsize.width() + 4)
            size.setHeight(iconsize.height())

        textsize = fm.size(Qt.TextSingleLine, opt.text)
        size = QSize(
            max(size.height(), textsize.height()),
            size.width() + 4 + textsize.width()
        )
        if self.__popup:
            opt.rect.setSize(size)
            pw = style.pixelMetric(QStyle.PM_MenuButtonIndicator, opt, self)
            size.setWidth(size.width() + pw)
        sh = style.sizeFromContents(QStyle.CT_PushButton, opt, size, self)
        return sh.expandedTo(QApplication.globalStrut())

    def paintEvent(self, event):
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        style = self.style()  # type: QStyle
        painter = QPainter(self)
        style.drawControl(QStyle.CE_PushButton, opt, painter, self)


class FlatButton(QPushButton):
    def __init__(self):
        super().__init__()

    def paintEvent(self, event):
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        style = self.style()  # type: QStyle
        opt.iconSize = QSize(16, 16)
        opt.icon = style.standardIcon(QStyle.SP_MessageBoxCritical)
        opt.features = QStyleOptionButton.HasMenu
        painter = QPainter(self)
        style.drawControl(QStyle.CE_PushButtonBevel, opt, painter)


def main(argv=None):
    from AnyQt.QtWidgets import QApplication, QWidget, QVBoxLayout
    app = QApplication(argv or [])
    w = QWidget()
    w.setLayout(QVBoxLayout())
    b = PopupButton()
    b.setPopup(QMenu())
    style = app.style()
    b.setIcon(style.standardIcon(QStyle.SP_MessageBoxCritical))
    b.setText("Text")
    # b.setFlat(True)
    # b.setMenu(QMenu())
    # b.menu().addAction("Action")
    w.layout().addWidget(b)
    w.layout().addStretch(10)
    w.show()
    w.raise_()
    return app.exec()


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
