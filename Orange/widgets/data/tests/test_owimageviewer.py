import unittest.mock

from PyQt4.QtGui import \
    QPixmap, QGraphicsView, QGraphicsScene, QFocusEvent, QPainter
from PyQt4.QtCore import Qt, QCoreApplication

from Orange.widgets.tests.base import GuiTest

from Orange.widgets.data.owimageviewer import (
    GraphicsThumbnailGrid, GraphicsThumbnailWidget
)


class Test(GuiTest):
    def setUp(self):
        self._scene = QGraphicsScene()

    def tearDown(self):
        self._scene.clear()
        self._scene.deleteLater()
        self._scene = None

    def test_grid(self):
        grid = GraphicsThumbnailGrid()
        self._scene.addItem(grid)
        pm = QPixmap(10, 10)
        pm.fill(Qt.black)
        grid.addThumbnail(pm)
        self.assertEqual(grid.count(), 1)
        tm0 = grid.thumbnailAt(0)
        self.assertIsInstance(tm0, GraphicsThumbnailWidget)

        tm1 = GraphicsThumbnailWidget(pm)
        tm2 = GraphicsThumbnailWidget(pm)

        grid.addThumbnail(tm1)
        self.assertIs(tm1, grid.thumbnailAt(1))
        grid.insertThumbnail(2, tm2)
        self.assertIs(tm2, grid.thumbnailAt(2))
        self.assertEqual(grid.count(), 3)

        with self.assertRaises(ValueError):
            grid.insertThumbnail(2, tm2)

        grid.removeThumbnail(tm1)

        self.assertEqual(grid.count(), 2)
        self.assertSequenceEqual([grid.thumbnailAt(0), grid.thumbnailAt(1)],
                                 [tm0, tm2])
        self.assertIsNone(tm1.parentItem())
        grid.insertThumbnail(1, tm1)
        self.assertSequenceEqual(grid.items(), [tm0, tm1, tm2])

        grid.clear()
        self.assertEqual(grid.count(), 0)
        self.assertSequenceEqual(grid.items(), [])

        recorder = unittest.mock.MagicMock()
        grid.currentThumbnailChanged.connect(recorder)
        grid.addThumbnail(tm0)
        # note the scene must be in a view/visible and active for this to work
        self.assertIsNone(grid.currentItem())

        def setFocus(item, reason=Qt.TabFocusReason):
            # set focus on an QGraphicsItem; ensuring that focusin event is
            # dispatched
            item.setFocus(Qt.TabFocusReason)
            # focus in events only get delivered when the scene is in an
            # visible/active view; therefore just send it ourselves
            QCoreApplication.sendEvent(
                item, QFocusEvent(QFocusEvent.FocusIn, reason))
        setFocus(tm0)

        self.assertIs(grid.currentItem(), tm0)
        recorder.assert_called_with(tm0)

        recorder.reset_mock()
        grid.removeThumbnail(tm0)
        recorder.assert_called_with(None)
        self.assertIs(grid.currentItem(), None)

        grid.addThumbnail(tm0)
        grid.addThumbnail(tm1)
        grid.addThumbnail(tm2)

        # one selected, one selected and focused, one with no state
        tm2.setSelected(True)
        tm1.setSelected(True)
        setFocus(tm1)

        # render the scene, to call the contained item's paint method
        pix = QPixmap(60, 60)
        painter = QPainter(pix)
        self._scene.render(painter)
        painter.end()

        grid.clear()
