from AnyQt.QtCore import Qt, QPoint, QMetaObject
from AnyQt.QtGui import QKeySequence, QWheelEvent
from AnyQt.QtWidgets import QDoubleSpinBox, QStyle, QAction, QLineEdit, QSlider

__all__ = ["DoubleSpinBoxWithSlider"]


class DoubleSpinBoxWithSlider(QDoubleSpinBox):
    """
    A QDoubleSpinBox with a popup slider widget for easier manipulation.

    Allows both exact text entry as well as a better mouse manipulation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        style = self.style()
        # TODO: Need icon that is more suitable
        icon = style.standardIcon(QStyle.SP_ArrowDown)
        action = QAction(
            icon, "Activate slider", self,
            shortcut=QKeySequence(Qt.AltModifier | Qt.Key_Down),
            shortcutContext=Qt.WidgetWithChildrenShortcut
        )
        le = self.lineEdit()
        le.addAction(action, QLineEdit.TrailingPosition)
        action.triggered.connect(self.__popup)
        self.addAction(action)

    def __popup(self):
        # Popup the slider.
        if self.findChild(QSlider, "-slider-popup") is not None:
            # Popup is already shown, do nothing
            return
        le = self.lineEdit()
        # Slider could also be horizontal below or above the spinbox?
        slider = PopupSlider(Qt.Vertical, self, objectName="-slider-popup")
        slider.setWindowFlags(Qt.Popup)
        slider.setAttribute(Qt.WA_DeleteOnClose)
        slider.setInvertedAppearance(True)
        # close keyboard shortcuts
        ac = QAction(
            "Close", slider, shortcut=QKeySequence.Close,
        )
        ac.setShortcuts([
            QKeySequence(QKeySequence.Close),
            QKeySequence(Qt.Key_Escape),
            QKeySequence(Qt.Key_Return),
            QKeySequence(Qt.Key_Enter)
        ])
        ac.triggered.connect(slider.close)
        slider.addAction(ac)
        fac = 10 ** self.decimals()
        slider.setRange(fac * self.minimum(), fac * self.maximum())
        slider.setSingleStep(fac * self.singleStep())
        slider.setPageStep(10 * fac * self.singleStep())
        slider.setValue(fac * self.value())

        geom = le.geometry()
        topr = le.mapToGlobal(geom.topRight())  # type: QPoint
        topl = topr - QPoint(slider.sizeHint().width(), 0)
        slider.move(topl)

        def setvalue(val):
            self.setValue(val / fac)

        slider.valueChanged.connect(setvalue)
        slider.show()
        slider.setFocus(Qt.PopupFocusReason)


class PopupSlider(QSlider):
    # are we in a middle of a kinetic scroll phase (this really only works
    # with Qt 5.12)
    __phaseDidBegin = False

    def wheelEvent(self, event):
        # type: (QWheelEvent) -> None
        phase = event.phase()
        if phase == Qt.ScrollBegin:
            self.__phaseDidBegin = True
        elif phase == Qt.ScrollEnd:
            self.__phaseDidBegin = False
        if self.isWindow() \
                and not self.rect().contains(event.pos()) \
                and (not self.__phaseDidBegin
                     or phase == Qt.ScrollBegin):
            event.ignore()
            self.close()
        else:
            super().wheelEvent(event)
