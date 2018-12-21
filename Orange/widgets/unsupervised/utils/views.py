import enum

from typing import Optional, Any

from AnyQt.QtCore import (
    Qt, QPoint, QSize, QAbstractItemModel, QPersistentModelIndex, QModelIndex
)
from AnyQt.QtGui import QKeyEvent, QContextMenuEvent
from AnyQt.QtWidgets import (
    QWidget, QTreeView, QMenu, QAction, QActionGroup, QComboBox,
    QStyledItemDelegate, QStyleOptionViewItem
)

from Orange.widgets.utils.itemmodels import iter_model, create_list_model

__all__ = [
    "AnalysisRoleView",
    "EnumItemDelegate",
]


# TODO: Could use a more generic name.
class AnalysisRoleView(QTreeView):
    """
    A two column view for assigning exclusive roles to items in a model.

    Allows the user to select multiple items and assign a state using a context
    menu popup, or using left/right arrow keys to switch to next/previous
    state.
    """
    def __init__(self, *args, editRole=Qt.EditRole, **kwargs):
        super().__init__(*args, **kwargs)
        self.__editColumn = 1
        self.__editRole = editRole
        self.__statesModel = None  # type: Optional[QAbstractItemModel]
        self.setHeaderHidden(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        header = self.header()
        header.setStretchLastSection(True)

    def setStateModel(self, model):
        # type: (QAbstractItemModel) -> None
        """
        Set the (list) model used for selection of possible states.

        Qt.UserRole is used as the source for values written back to the
        view's model.

        NOTE
        ----
        Ownership of the model stays with the caller.
        """
        self.__statesModel = model

    def stateModel(self):
        # type: () -> Optional[QAbstractItemModel]
        """Return the state model"""
        return self.__statesModel

    def editRole(self):
        # type: () -> Qt.ItemDataRole
        """
        The edit role in the model (self.model()) from which the states are
        read and written to.
        """
        return self.__editRole

    def setEditRole(self, role):
        # type: (Qt.ItemDataRole) -> None
        """
        Set the edit role
        """
        self.__editRole = role

    def editColumn(self):
        # type: () -> int
        """
        The column in the model (self.model()) from which the states are
        read and written to.
        """
        return self.__editColumn

    def setEditColumn(self, column):
        # type: (int) -> None
        """
        Set the edit column.
        """
        self.__editColumn = column

    def keyPressEvent(self, event):
        # type: (QKeyEvent) -> None
        """Reimplemented."""
        if self.__editKeyEvent(event):
            event.setAccepted(True)
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        # type: (QContextMenuEvent) -> None
        """Reimplemented."""
        self.__popup(event.globalPos())

    def __popup(self, pos):
        # type: (QPoint) -> None
        # Popup a menu for setting the state of the current selection at `pos`.
        # `pos` is in global screen coordinates.
        states = self.__statesModel
        if states is None:
            return
        # find the current common state for the selection
        selection = self.selectionModel().selectedRows(self.editColumn())
        role = self.editRole()
        current = set(filter(None, (idx.data(role) for idx in selection)))
        if len(current) == 1:  # have a common current state
            current = current.pop()
        else:
            current = None
        pselection = [QPersistentModelIndex(idx) for idx in selection]
        menu = QMenu(self)
        menu.setAttribute(Qt.WA_DeleteOnClose)
        group = QActionGroup(menu, exclusive=True)
        for index in iter_model(self.__statesModel):
            text = index.data(Qt.DisplayRole)
            a = QAction(str(text), group, checkable=True)
            a.setData(index.data(Qt.UserRole))
            tip = index.data(Qt.ToolTipRole)
            if isinstance(tip, str):
                a.setToolTip(tip)
            whatsthis = index.data(Qt.WhatsThisRole)
            if isinstance(whatsthis, str):
                a.setWhatsThis(whatsthis)
            state = index.data(Qt.UserRole)
            a.setChecked(state == current)
            menu.addAction(a)

        def setstate(action):  # type: (QAction) -> None
            # write back the state to the model
            data = action.data()
            for pidx in pselection:
                if not pidx.isValid():
                    continue
                idx = pidx.sibling(pidx.row(), pidx.column())  # to QModelIndex
                if idx.isValid():
                    model = idx.model()
                    model.setData(idx, data, role)

        menu.triggered[QAction].connect(setstate)
        menu.popup(pos, group.checkedAction())
        menu.setFocus(Qt.PopupFocusReason)

    def __editKeyEvent(self, event):
        # type: (QKeyEvent) -> bool
        # Handle key event, return True if the event was handled.
        selection = self.selectionModel().selectedRows(self.editColumn())
        model = self.model()
        if not selection:
            return False
        if (event.key() == Qt.Key_Down and event.modifiers() & Qt.AltModifier) \
                or event.key() == Qt.Key_Menu:
            # on Alt + Key_Down open a popup menu
            # ensure current index is visible and map its pos to screen coord.
            current = self.selectionModel().currentIndex()
            self.scrollTo(current, QTreeView.EnsureVisible)
            pos = self.visualRect(current).center()
            pos = self.mapToGlobal(pos)
            self.__popup(pos)
            return True
        if event.key() == Qt.Key_Left:
            direction = 1
        elif event.key() == Qt.Key_Right:
            direction = -1
        else:
            return False

        editrole = self.editRole()
        for i in selection:
            value = model.data(i, editrole)
            model.setData(i, self.nextState(value, direction))
        return True

    def nextState(self, value, direction):
        # type: (Any, int) -> Any
        """
        Return the next state (item) in the `statesModel()`.

        Parameters
        ----------
        value : Any
            An value from Qt.UserRole in the `statesModel()`
        direction : int
            -1 or 1 specifies the search direction

        Returns
        -------
        next : Any
            The next Qt.UserRole value in the `statesModel()`.
        """
        model = self.__statesModel
        if model is None:
            return None

        matches = model.match(
            model.index(0, 0), Qt.UserRole, value, 1, Qt.MatchExactly
        )
        if not matches:
            return None
        match = matches[0]
        index = match.sibling(
            (match.row() + direction) % model.rowCount(), match.column()
        )
        return index.data(Qt.UserRole)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        column = self.editColumn()
        header = self.header()
        if header.count() != 2 or \
                header.visualIndex(column) != header.count() - 1:
            return
        # The last section is fixed but also forced stretch (stretchLastSection)
        # so it always gets extra space. Spill the extra width to to first
        # section instead
        sizes = [header.sectionSize(i) for i in range(header.count())]
        size = self.sizeHintForColumn(column)
        spill = sizes[column] - size
        header.resizeSection(0, sizes[0] + spill)
        header.resizeSection(1, sizes[1] - spill)

    def sizeHintForColumn(self, column):
        if column == self.editColumn():
            delegagte = self.itemDelegateForColumn(column)
            if delegagte is None:
                delegagte = self.itemDelegate()
            states = self.__statesModel
            if states is None:
                return super().sizeHintForColumn(column)
            # infer size hint for the column for all states
            m = create_list_model([{}])
            index = m.index(0, 0)
            option = self.viewOptions()
            sh = QSize()
            for i in range(states.rowCount()):
                item = states.index(i, 0).data(Qt.UserRole)
                m.setData(index, item, self.editRole())
                sh = sh.expandedTo(delegagte.sizeHint(option, index))
            return sh.width()
        else:
            return super().sizeHintForColumn(column)


class EnumItemDelegate(QStyledItemDelegate):
    """
    An item delegate for displaying and selecting values of an `enum.Enum`.
    """
    def displayText(self, value, locale):
        """Reimplemented."""
        if isinstance(value, enum.Enum):
            return value.name
        else:
            return super().displayText(value, locale)

    def createEditor(self, parent, option, index):
        # type: (QWidget, QStyleOptionViewItem, QModelIndex) -> QWidget
        """Reimplemented."""
        data = index.data(Qt.EditRole)
        if not isinstance(data, enum.Enum):
            return super().createEditor(parent, option, index)
        else:
            model = create_list_model([
                {Qt.DisplayRole: str(e.value), Qt.UserRole: e}
                for e in type(data)
            ])

        editor = QComboBox(parent, autoFillBackground=True, frame=False)
        editor.setModel(model)
        editor.setCurrentIndex(editor.findData(data, Qt.UserRole))
        model.setParent(editor)
        return editor

    def setEditorData(self, editor, index):
        # type: (QWidget, QModelIndex) -> None
        """Reimplemented."""
        data = index.data(Qt.EditRole)
        if isinstance(editor, QComboBox):
            editor.setCurrentIndex(editor.findData(data, Qt.UserRole))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        # type: (QWidget, QAbstractItemModel, QModelIndex) -> None
        """Reimplemented."""
        if isinstance(editor, QComboBox):
            data = editor.itemData(editor.currentIndex(), Qt.UserRole)
            model.setData(index, data, Qt.EditRole)
        else:
            super().setModelData(editor, model, index)

    def initStyleOption(self, option, index):
        # type: (QStyleOptionViewItem, QModelIndex) -> None
        """Reimplemented."""
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignRight | Qt.AlignVCenter
        option.textElideMode = Qt.ElideMiddle
        option.text = option.text + "\N{vertical ellipsis}"
