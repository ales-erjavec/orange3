"""
Tests for DropShadowFrame wiget.

"""
import math

from AnyQt.QtWidgets import (
    QMainWindow, QWidget, QListView, QTextEdit, QHBoxLayout, QToolBar,
    QVBoxLayout
)
from AnyQt.QtGui import QColor

from AnyQt.QtCore import Qt, QPoint, QPropertyAnimation, QVariantAnimation
from .. import dropshadow

from .. import test


class TestDropShadow(test.QAppTestCase):
    def test(self):
        lv = QListView()
        mw = QMainWindow()
        # Add two tool bars, the shadow should extend over them.
        mw.addToolBar(Qt.BottomToolBarArea, QToolBar())
        mw.addToolBar(Qt.TopToolBarArea, QToolBar())
        mw.setCentralWidget(lv)

        f = dropshadow.DropShadowFrame(color=Qt.blue, radius=20)

        f.setWidget(lv)

        self.assertIs(f.parentWidget(), mw)
        self.assertIs(f.widget(), lv)

        mw.show()

        self.app.processEvents()

        self.singleShot(3000, lambda: f.setColor(Qt.red))
        self.singleShot(4000, lambda: f.setRadius(30))
        self.singleShot(5000, lambda: f.setRadius(40))
        self.app.exec_()

    def test1(self):
        class FT(QToolBar):
            def paintEvent(self, e):
                pass

        w = QMainWindow()
        ftt, ftb = FT(), FT()
        ftt.setFixedHeight(15)
        ftb.setFixedHeight(15)

        w.addToolBar(Qt.TopToolBarArea, ftt)
        w.addToolBar(Qt.BottomToolBarArea, ftb)

        f = dropshadow.DropShadowFrame()
        te = QTextEdit()
        c = QWidget()
        c.setLayout(QVBoxLayout())
        c.layout().setContentsMargins(20, 0, 20, 0)
        c.layout().addWidget(te)
        w.setCentralWidget(c)
        f.setWidget(te)
        f.radius = 15
        f.color = QColor(Qt.blue)
        w.show()

        self.singleShot(3000, lambda: f.setColor(Qt.red))
        self.singleShot(4000, lambda: f.setRadius(30))
        self.singleShot(5000, lambda: f.setRadius(40))

        self.app.exec_()

    def test_offset(self):
        w = QWidget()
        w.setLayout(QHBoxLayout())
        w.setContentsMargins(30, 30, 30, 30)
        ww = QTextEdit()
        w.layout().addWidget(ww)
        f = dropshadow.DropShadowFrame(radius=20)
        f.setWidget(ww)
        oanim = QVariantAnimation(
            f, startValue=0.0, endValue=2 * math.pi, loopCount=-1,
            duration=2000,
        )
        @oanim.valueChanged.connect
        def _(value):
            f.setOffset(QPoint(4 * math.cos(value), 4 * math.sin(value)))
        oanim.start()
        w.show()
        self.app.exec_()


if __name__ == "__main__":
    test.unittest.main()
