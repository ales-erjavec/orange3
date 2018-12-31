import enum
from xml.sax.saxutils import escape
from typing import Optional, List

import pkg_resources

from AnyQt.QtCore import QObject, QEvent, Qt, QRectF, QPointF
from AnyQt.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
from AnyQt.QtGui import (
    QPainterPath, QColor, QPen, QBrush, QKeySequence, QIcon, QPalette,
)
from AnyQt.QtWidgets import (
    QGraphicsPathItem, QGraphicsRectItem, QPinchGesture, QAction, QActionGroup,
    QToolButton, QGraphicsSceneMouseEvent, QGestureEvent,
    QWidget, QAbstractButton,
    QMenu)

import pyqtgraph as pg


class PlotTool(QObject):
    """
    An abstract tool operating on a :class:`~pg.ViewBox`.

    Subclasses of `PlotTool` implement various actions responding to
    user input events. For instance `PlotZoomTool`, when active, allows
    the user to select/draw a rectangular region on a plot in which to
    zoom.

    The tool works by installing itself as an `eventFilter` on to the
    :class:`~pg.ViewBox` instance and dispatching events to the appropriate
    event handlers (`mousePressEvent`, ...).

    When subclassing note that the event handlers (`mousePressEvent`, ...)
    are actually event filters and need to return a boolean value
    indicating if the event was handled (filtered) and should not propagate
    further to the view box.

    See Also
    --------
    QObject.eventFilter

    """
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__viewbox = None  # type: Optional[pg.ViewBox]

    def setViewBox(self, viewbox):
        # type: (pg.ViewBox) -> None
        """
        Set the view box to operate on.

        Call ``setViewBox(None)`` to remove the tool from the current
        view box. If an existing view box is already set it is first
        removed.

        .. note::
            The PlotTool will install itself as an event filter on the
            view box.

        Parameters
        ----------
        viewbox : pg.ViewBox or None

        """
        if self.__viewbox is viewbox:
            return
        if self.__viewbox is not None:
            self.__viewbox.removeEventFilter(self)
            self.__viewbox.destroyed.disconnect(self.__viewdestroyed)

        self.__viewbox = viewbox

        if self.__viewbox is not None:
            self.__viewbox.installEventFilter(self)
            self.__viewbox.destroyed.connect(self.__viewdestroyed)

    def viewBox(self):
        # type: () -> pg.ViewBox
        """
        Return the view box.

        Returns
        -------
        viewbox : pg.ViewBox
        """
        return self.__viewbox

    @Slot()
    def __viewdestroyed(self):
        self.__viewbox = None

    def mousePressEvent(self, event):
        # type: (QGraphicsSceneMouseEvent) -> bool
        """
        Handle a mouse press event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event.

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def mouseMoveEvent(self, event):
        # type: (QGraphicsSceneMouseEvent) -> bool
        """
        Handle a mouse move event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def mouseReleaseEvent(self, event):
        # type: (QGraphicsSceneMouseEvent) -> bool
        """
        Handle a mouse release event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def mouseDoubleClickEvent(self, event):
        # type: (QGraphicsSceneMouseEvent) -> bool
        """
        Handle a mouse double click event.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The event.

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def gestureEvent(self, event):
        # type: (QGestureEvent) -> bool
        """
        Handle a gesture event.

        Parameters
        ----------
        event : QGraphicsSceneGestureEvent
            The event.

        Returns
        -------
        status : bool
            True if the event was handled (and should not
            propagate further to the view box) and False otherwise.
        """
        return False

    def eventFilter(self, obj, event):
        # type: (QObject, QEvent) -> bool
        """
        Reimplemented from `QObject.eventFilter`.
        """
        if obj is self.__viewbox:
            if event.type() == QEvent.GraphicsSceneMousePress:
                return self.mousePressEvent(event)
            elif event.type() == QEvent.GraphicsSceneMouseMove:
                return self.mouseMoveEvent(event)
            elif event.type() == QEvent.GraphicsSceneMouseRelease:
                return self.mouseReleaseEvent(event)
            elif event.type() == QEvent.GraphicsSceneMouseDoubleClick:
                return self.mouseDoubleClickEvent(event)
            elif event.type() == QEvent.Gesture:
                return self.gestureEvent(event)
        return super().eventFilter(obj, event)

    @staticmethod
    def pushViewRect(viewbox, rect):
        # type: (pg.ViewBox, QRectF) -> None
        viewbox.showAxRect(rect)
        viewbox.axHistoryPointer += 1
        viewbox.axHistory[viewbox.axHistoryPointer:] = [rect]

    @staticmethod
    def popViewRect(viewbox):
        # type: (pg.ViewBox) -> None
        if viewbox.axHistoryPointer == 0:
            viewbox.autoRange()
            viewbox.axHistory = []
            viewbox.axHistoryPointer = -1
        else:
            viewbox.scaleHistory(-1)


class PlotSelectionTool(PlotTool):
    """
    A tool for selecting a region on a plot.
    """
    #: Selection modes
    Rect, Lasso = 1, 2

    #: Selection was started by the user.
    selectionStarted = Signal(QPainterPath)
    #: The current selection has been updated
    selectionUpdated = Signal(QPainterPath)
    #: The selection has finished (user has released the mouse button)
    selectionFinished = Signal(QPainterPath)

    def __init__(self, parent=None, selectionMode=Rect, **kwargs):
        super().__init__(parent, **kwargs)
        self.__mode = selectionMode
        self.__path = None
        self.__item = None

    def setSelectionMode(self, mode):
        """
        Set the selection mode (rectangular or lasso selection).

        Parameters
        ----------
        mode : int
            PlotSelectionTool.Rect or PlotSelectionTool.Lasso

        """
        assert mode in {PlotSelectionTool.Rect, PlotSelectionTool.Lasso}
        if self.__mode != mode:
            if self.__path is not None:
                self.selectionFinished.emit()
            self.__mode = mode
            self.__path = None

    def selectionMode(self):
        """
        Return the current selection mode.
        """
        return self.__mode

    def selectionShape(self):
        """
        Return the current selection shape.

        This is the area selected/drawn by the user.

        Returns
        -------
        shape : QPainterPath
            The selection shape in view coordinates.
        """
        if self.__path is not None:
            shape = QPainterPath(self.__path)
            shape.closeSubpath()
        else:
            shape = QPainterPath()
        viewbox = self.viewBox()

        if viewbox is None:
            return QPainterPath()

        return viewbox.childGroup.mapFromParent(shape)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(pos, pos)
                self.__path = QPainterPath()
                self.__path.addRect(rect)
            else:
                self.__path = QPainterPath()
                self.__path.moveTo(event.pos())
            self.selectionStarted.emit(self.selectionShape())
            self.__updategraphics()
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(event.buttonDownPos(Qt.LeftButton), event.pos())
                self.__path = QPainterPath()
                self.__path.addRect(rect)
            else:
                self.__path.lineTo(event.pos())
            self.selectionUpdated.emit(self.selectionShape())
            self.__updategraphics()
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.__mode == PlotSelectionTool.Rect:
                rect = QRectF(event.buttonDownPos(Qt.LeftButton), event.pos())
                self.__path = QPainterPath()
                self.__path.addRect(rect)
            else:
                self.__path.lineTo(event.pos())
            self.selectionFinished.emit(self.selectionShape())
            self.__path = QPainterPath()
            self.__updategraphics()
            event.accept()
            return True
        else:
            return False

    def __updategraphics(self):
        viewbox = self.viewBox()
        if viewbox is None:
            return

        if self.__path.isEmpty():
            if self.__item is not None:
                self.__item.setParentItem(None)
                viewbox.removeItem(self.__item)
                if self.__item.scene():
                    self.__item.scene().removeItem(self.__item)
                self.__item = None
        else:
            if self.__item is None:
                palette = viewbox.palette()
                color = palette.color(QPalette.Disabled, QPalette.Highlight)
                item = QGraphicsPathItem()
                item.setPen(QPen(color, 0))
                color.setAlpha(50)
                item.setBrush(QBrush(color))
                self.__item = item
                viewbox.addItem(item)

            self.__item.setPath(self.selectionShape())


class PlotZoomTool(PlotTool):
    """
    A zoom tool.

    Allows the user to draw a rectangular region to zoom in.
    """

    zoomStarted = Signal(QRectF)
    zoomUpdated = Signal(QRectF)
    zoomFinished = Signal(QRectF)

    def __init__(self, parent=None, autoZoom=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.__zoomrect = QRectF()
        self.__zoomitem = None
        self.__autozoom = autoZoom

    def zoomRect(self):
        """
        Return the current drawn rectangle (region of interest)

        Returns
        -------
        zoomrect : QRectF
        """
        view = self.viewBox()
        if view is None:
            return QRectF()
        return view.childGroup.mapRectFromParent(self.__zoomrect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__zoomrect = QRectF(event.pos(), event.pos())
            self.zoomStarted.emit(self.zoomRect())
            self.__updategraphics()
            event.accept()
            return True
        elif event.button() == Qt.RightButton:
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.__zoomrect = QRectF(
                event.buttonDownPos(Qt.LeftButton), event.pos()).normalized()
            self.zoomUpdated.emit(self.zoomRect())
            self.__updategraphics()
            event.accept()
            return True
        elif event.buttons() & Qt.RightButton:
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__zoomrect = QRectF(
                event.buttonDownPos(Qt.LeftButton), event.pos()).normalized()

            if self.__autozoom:
                PlotTool.pushViewRect(self.viewBox(), self.zoomRect())

            self.zoomFinished.emit(self.zoomRect())
            self.__zoomrect = QRectF()
            self.__updategraphics()
            event.accept()
            return True
        elif event.button() == Qt.RightButton:
            PlotTool.popViewRect(self.viewBox())
            event.accept()
            return True
        else:
            return False

    def __updategraphics(self):
        viewbox = self.viewBox()
        if viewbox is None:
            return
        if not self.__zoomrect.isValid():
            if self.__zoomitem is not None:
                self.__zoomitem.setParentItem(None)
                viewbox.removeItem(self.__zoomitem)
                if self.__zoomitem.scene() is not None:
                    self.__zoomitem.scene().removeItem(self.__zoomitem)
                self.__zoomitem = None
        else:
            if self.__zoomitem is None:
                self.__zoomitem = QGraphicsRectItem()
                palette = viewbox.palette()
                color = palette.color(QPalette.Disabled,
                                      QPalette.Highlight)
                self.__zoomitem.setPen(QPen(color, 0))
                color.setAlpha(50)
                self.__zoomitem.setBrush(QBrush(color))
                viewbox.addItem(self.__zoomitem)

            self.__zoomitem.setRect(self.zoomRect())


class PlotPanTool(PlotTool):
    """
    Pan/translate tool.
    """
    panStarted = Signal()
    translated = Signal(QPointF)
    panFinished = Signal()

    def __init__(self, parent=None, autoPan=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.__autopan = autoPan
        self.__lastPos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panStarted.emit()
            event.accept()
            return True
        else:
            return False

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            viewbox = self.viewBox()
            if self.__lastPos is None:
                # pg.GraphicsScene will duplicate mouseMoveEvents -> cannot
                # use event.lastPos() directly need to store it
                self.__lastPos = event.lastPos()
            delta = (viewbox.mapToView(event.pos()) -
                     viewbox.mapToView(self.__lastPos))
            if self.__autopan:
                viewbox.translateBy(-delta)
            self.translated.emit(-delta)
            self.__lastPos = event.pos()
            event.accept()
            return True
        else:
            return False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panFinished.emit()
            self.__lastPos = None
            event.accept()
            return True
        else:
            return False


class PlotPinchZoomTool(PlotTool):
    """
    A tool implementing a "Pinch to zoom".
    """
    def gestureEvent(self, event):
        gesture = event.gesture(Qt.PinchGesture)
        if gesture is None:
            return False
        if gesture.state() == Qt.GestureStarted:
            event.accept(gesture)
            return True
        elif gesture.changeFlags() & QPinchGesture.ScaleFactorChanged:
            viewbox = self.viewBox()
            center = viewbox.mapSceneToView(gesture.centerPoint())
            scale = gesture.scaleFactor()
            if scale > 0:
                viewbox.scaleBy((1 / scale, 1 / scale), center)
            event.accept()
            return True
        elif gesture.state() == Qt.GestureFinished:
            viewbox = self.viewBox()
            PlotTool.pushViewRect(viewbox, viewbox.viewRect())
            event.accept()
            return True
        else:
            return False


class PlotToolBox(QObject):
    actionTriggered = Signal(QAction)
    toolActivated = Signal(PlotTool)

    class StandardAction(enum.IntEnum):
        """
        An enum representing standard plot actions
        """
        #: No action
        NoAction = 0
        #: Reset zoom (zoom to fit) action with CTRL + Key_0 shortcut
        ZoomReset = 1
        #: Zoom in action with QKeySequence.ZoomIn shortcut
        ZoomIn = 2
        #: Zoom out action with QKeySequence.ZoomOut shortcut
        ZoomOut = 4
        #: A Select tool action (exclusive with other *Tool)
        SelectTool = 8
        #: A Zoom tool action (exclusive with other *Tool)
        ZoomTool = 16
        #: A Pan tool  (exclusive with other *Tool)
        PanTool = 32

    NoAction, ZoomReset, ZoomIn, ZoomOut, SelectTool, ZoomTool, PanTool = \
        list(StandardAction)

    _DefaultActions = (ZoomReset | ZoomIn | ZoomOut |
                       SelectTool | ZoomTool | PanTool)

    _ExclusiveMask = SelectTool | ZoomTool | PanTool

    _ActionData = {
        ZoomReset: ("Zoom to fit", "zoom_reset",
                    Qt.ControlModifier + Qt.Key_0),
        ZoomIn: ("Zoom in", "", QKeySequence.ZoomIn),
        ZoomOut: ("Zoom out", "", QKeySequence.ZoomOut),
        SelectTool: ("Select", "arrow", Qt.ControlModifier | Qt.Key_1),
        ZoomTool: ("Zoom", "zoom", Qt.ControlModifier | Qt.Key_2),
        PanTool: ("Pan", "pan_hand", Qt.ControlModifier | Qt.Key_3),
    }

    def __init__(self, parent=None, standardActions=_DefaultActions, **kwargs):
        super().__init__(parent, **kwargs)
        self.__standardActions = standardActions
        self.__actions = {}
        self.__tools = {}
        self.__viewBox = None
        self.__currentTool = None
        self.__toolgroup = QActionGroup(self, exclusive=True)
        self.__toolgroup.triggered[QAction].connect(self.__on_toolAction)

        def icon(name):
            path = "icons/Dlg_{}.png".format(name)
            path = pkg_resources.resource_filename(
                "Orange.widgets.widget", path
            )
            return QIcon(path)

        isfirsttool = True
        for flag in PlotToolBox.StandardAction:
            if standardActions & flag:
                _text, _iconname, _keyseq = PlotToolBox._ActionData[flag]
                action = QAction(
                    _text, self, icon=icon(_iconname),
                    shortcut=QKeySequence(_keyseq)
                )
                tooltip = "{} <kbd>{}</kbd>".format(
                    escape(_text),
                    action.shortcut().toString(QKeySequence.NativeText)
                )
                action.setToolTip(tooltip)
                self.__actions[flag] = action
                if flag & PlotToolBox._ExclusiveMask:
                    action.setCheckable(True)
                    self.__toolgroup.addAction(action)
                    if flag == PlotToolBox.SelectTool:
                        tool = PlotSelectionTool(self)
                        tool.cursor = Qt.ArrowCursor
                        action._menu = menu = QMenu()
                        menu.addAction("Rectangular")
                        menu.addAction("Lasso")
                        action.setMenu(menu)
                    elif flag == PlotToolBox.ZoomTool:
                        tool = PlotZoomTool(self)
                        tool.cursor = Qt.ArrowCursor
                    elif flag == PlotToolBox.PanTool:
                        tool = PlotPanTool(self)
                        tool.cursor = Qt.OpenHandCursor

                    self.__tools[flag] = tool
                    action.setData(tool)

                    if isfirsttool:
                        action.setChecked(True)
                        self.__currentTool = tool
                        isfirsttool = False

    def __on_toolAction(self, action):
        tool = action.data()
        if not isinstance(tool, PlotTool):
            return
        if self.__currentTool is not None:
            self.__currentTool.setViewBox(None)
        self.__currentTool = tool
        if tool is not None:
            tool.setViewBox(self.__viewBox)
            if self.__viewBox is not None:
                self.__viewBox.setCursor(tool.cursor)

    def setViewBox(self, box):
        # type: (pg.ViewBox) -> None
        """Set the view box on which to operate."""
        if self.__viewBox is not box and self.__currentTool is not None:
            self.__currentTool.setViewBox(None)
            # TODO: Unset/restore default view box cursor
            self.__viewBox = None

        self.__viewBox = box
        if self.__currentTool is not None:
            self.__currentTool.setViewBox(box)
            if box is not None:
                box.setCursor(self.__currentTool.cursor)

    def viewBox(self):
        # type: () -> pg.ViewBox
        """Return the current view box."""
        return self.__viewBox

    def standardAction(self, action):
        # type: (StandardAction) -> QAction
        """Return the QAction for a specified StandardAction"""
        return self.__actions[action]

    def actions(self):
        # type: () -> List[QAction]
        return list(self.__actions.values())

    def button(self, action, parent=None):
        # type: (StandardAction, Optional[QWidget]) -> QAbstractButton
        action = self.standardAction(action)
        b = QToolButton(parent)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setDefaultAction(action)
        return b

    def toolGroup(self):
        """Return the exclusive tool action button group"""
        return self.__toolgroup

    def plotTool(self, action):
        # type: (StandardAction) -> PlotTool
        return self.__tools[action]
