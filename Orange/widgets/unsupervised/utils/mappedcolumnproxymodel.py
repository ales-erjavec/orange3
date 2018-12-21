from typing import Dict

from AnyQt.QtCore import (
    Qt, QModelIndex, QAbstractItemModel, QIdentityProxyModel,
)

__all__ = [
    "MappedColumnProxyModel"
]


class MappedColumnProxyModel(QIdentityProxyModel):
    """
    A proxy model extending a list model with a second column mapping
    select roles back into the original model.

    Can be used to edit auxiliary edit roles in a two column view.
    """
    # This class could be extended to map multiple columns.

    __mappedRoles = ...  # type: Dict[Qt.ItemDataRole, Qt.ItemDataRole]

    def __init__(self, parent=None, **kwargs):
        mappedRoles = kwargs.pop("mappedRoles", {})
        super().__init__(parent, **kwargs)
        self.__mappedRoles = mappedRoles.copy()

    def setMappedRoles(self, mappedRoles):
        # type: (Dict[Qt.ItemDataRole, Qt.ItemDataRole]) -> None
        """
        Set the mapped roles between column 1 and the source model's roles
        """
        self.__mappedRoles = mappedRoles.copy()
        if self.sourceModel() is not None and self.rowCount():
            self.dataChanged.emit(
                self.index(0, 1),
                self.index(self.rowCount() - 1, 1)
            )

    def mappedRoles(self):
        # type: () -> Dict[Qt.ItemDataRole, Qt.ItemDataRole]
        """
        Return the mapped roles.
        """
        return self.__mappedRoles.copy()

    def setSourceModel(self, sourceModel):
        # type: (QAbstractItemModel) -> None
        """Reimplemented."""
        model = self.sourceModel()
        if model is not None:
            model.dataChanged.disconnect(self.__onDataChanged)
        super().setSourceModel(sourceModel)
        if sourceModel is not None:
            sourceModel.dataChanged.connect(self.__onDataChanged)

    def rowCount(self, parent=QModelIndex()):
        """Reimplemented."""
        if parent.isValid() or self.sourceModel() is None:
            return 0
        return self.sourceModel().rowCount()

    def columnCount(self, parent=QModelIndex()):
        """Reimplemented."""
        return 0 if parent.isValid() else 2

    def mapToSource(self, proxyIndex):
        # type: (QModelIndex) -> QModelIndex
        """Reimplemented."""
        row, col = proxyIndex.row(), proxyIndex.column()
        if col == 1:  # map the extra column back
            col = 0
        return self.sourceModel().index(row, col)

    def data(self, index, role=Qt.DisplayRole):
        # type: (QModelIndex, Qt.ItemDataRole) -> Any
        """Reimplemented."""
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        source = self.sourceModel()
        index = self.mapToSource(index)
        if col == 0:
            return source.data(index, role)
        elif col == 1 and role in self.__mappedRoles:
            return source.data(index, self.__mappedRoles[role])
        else:
            return None

    def index(self, row, column, parent=QModelIndex()):
        # type: (int, int, QModelIndex) -> QModelIndex
        """Reimplemented."""
        # MUST not create indexes that are out of bounds or parented
        if self.hasIndex(row, column, parent):
            return self.createIndex(row, column)
        else:
            return QModelIndex()

    def parent(self, child):
        """Reimplemented."""
        return QModelIndex()

    def flags(self, index):
        # type: (QModelIndex) -> Qt.ItemFlags
        """Reimplemented."""
        if not index.isValid():
            return Qt.NoItemFlags
        source = self.sourceModel()
        if index.column() == 1:
            # ?? query source; ItemIsEditable if the source is enabled ??
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        sourceindex = self.mapToSource(index)
        return source.flags(sourceindex)

    def setData(self, index, value, role=Qt.EditRole):
        # type: (QModelIndex, Any, Qt.ItemDataRole) -> bool
        """Reimplemented."""
        if not index.isValid():
            return False
        row, col = index.row(), index.column()
        source = self.sourceModel()
        index = self.mapToSource(index)
        if col == 0:
            return source.setData(index, value, role)
        elif col == 1 and role in self.__mappedRoles:
            return source.setData(index, value, self.__mappedRoles[role])
        else:
            return False

    def buddy(self, index):
        # type: (QModelIndex) -> QModelIndex
        """Reimplemented."""
        if index.column() == 0:
            return index.sibling(index.row(), 1)
        else:
            return index

    def sibling(self, row, column, idx):
        # type: (int, int, QModelIndex) -> QModelIndex
        """Reimplemented."""
        return self.index(row, column, idx.parent())

    def __onDataChanged(self, tl, br, roles=[]):
        tl = self.mapFromSource(tl)
        br = self.mapFromSource(br)
        # change the range to column 1. The base class will emit changes for
        # the original column.
        tl = tl.sibling(tl.row(), 1)
        br = br.sibling(br.row(), 1)
        if tl.isValid() and br.isValid():
            self.dataChanged.emit(tl, br)
