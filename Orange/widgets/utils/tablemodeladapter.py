from typing import Dict, Callable, Any, Tuple

from AnyQt.QtCore import Qt, QModelIndex, QAbstractTableModel


class TableModelDispatcher(QAbstractTableModel):
    __slots__ = (
        "shape", "__data", "__column_header", "__row_header",
        "__row_count", "__col_count",
    )

    Dispatch = Dict[Qt.ItemDataRole, Callable[[int, int], Any]]
    HeaderDispatch = Dict[Qt.ItemDataRole, Callable[[int], Any]]

    def __init__(
            self,
            shape: Tuple[int, int],
            data_dispatch: Dispatch,
            column_header_dispatch: HeaderDispatch = None,
            row_header_dispatch: HeaderDispatch = None,
            parent=None,
            **kwargs
    ) -> None:
        super().__init__(parent, **kwargs)
        self.shape = shape
        self.__row_count = shape[0]
        self.__col_count = shape[1]
        self.__data = data_dispatch
        self.__column_header = column_header_dispatch or {}
        self.__row_header = row_header_dispatch or {}

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return self.__row_count

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        else:
            return self.__col_count

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row = index.row()
        column = index.column()
        N, M = self.shape
        if not 0 <= row < N and 0 <= column < M:
            return None

        delegate = self.__data.get(role, None)
        if delegate is not None:
            return delegate(row, column)
        else:
            return None

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> Any:
        if orientation == Qt.Horizontal:
            delegate = self.__column_header.get(role, None)
            if delegate is not None:
                return delegate(section)
            elif role == Qt.DisplayRole:
                return section + 1
        elif orientation == Qt.Vertical:
            delegate = self.__row_header.get(role, None)
            if delegate is not None:
                return delegate(section)
            elif role == Qt.DisplayRole:
                return section + 1