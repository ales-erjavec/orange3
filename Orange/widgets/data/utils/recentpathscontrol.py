import os
import types
from typing import Optional, Any, Mapping

from PyQt5.QtCore import QFileInfo, Qt, QAbstractItemModel, pyqtSignal as Signal
from PyQt5.QtGui import QIcon, QStandardItem, QColor, QStandardItemModel
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QComboBox, QSizePolicy, QPushButton, QStyle,
    QFileIconProvider
)

from Orange.widgets.utils.combobox import ItemStyledComboBox
from Orange.widgets.utils.pathutils import PathItem, VarPath


def icon_for_path(path: str) -> QIcon:
    """
    Return an icon for the file/directory at `path`.
    """
    iconprovider = QFileIconProvider()
    finfo = QFileInfo(path)
    if finfo.exists():
        return iconprovider.icon(finfo)
    else:
        return iconprovider.icon(QFileIconProvider.File)


class VarPathItem(QStandardItem):
    """
    An QStandardItem
    """
    PathRole = Qt.UserRole + 4502
    VarPathRole = PathRole + 1

    def path(self) -> str:
        """Return the resolved path or '' if unresolved or missing"""
        path = self.data(VarPathItem.PathRole)
        return path if isinstance(path, str) else ""

    def setPath(self, path: str) -> None:
        """Set absolute path."""
        self.setData(PathItem.AbsPath(path), VarPathItem.VarPathRole)

    def varPath(self) -> Optional[PathItem]:
        vpath = self.data(VarPathItem.VarPathRole)
        return vpath if isinstance(vpath, PathItem) else None

    def setVarPath(self, vpath: PathItem) -> None:
        """Set variable path item."""
        self.setData(vpath, VarPathItem.VarPathRole)

    def resolve(self, vpath: PathItem) -> Optional[str]:
        """
        Resolve `vpath` item. This implementation dispatches to parent model's
        (:func:`VarPathItemModel.resolve`)
        """
        model = self.model()
        if isinstance(model, VarPathItemModel):
            return model.resolve(vpath)
        else:
            return vpath.resolve({})

    def data(self, role=Qt.UserRole + 1) -> Any:
        if role == Qt.DisplayRole:
            value = super().data(role)
            if value is not None:
                return value
            vpath = self.varPath()
            if isinstance(vpath, PathItem.AbsPath):
                return os.path.basename(vpath.path)
            elif isinstance(vpath, PathItem.VarPath):
                return os.path.basename(vpath.relpath)
            else:
                return None
        elif role == Qt.DecorationRole:
            return icon_for_path(self.path())
        elif role == VarPathItem.PathRole:
            vpath = self.data(VarPathItem.VarPathRole)
            if isinstance(vpath, PathItem.AbsPath):
                return vpath.path
            elif isinstance(vpath, VarPath):
                path = self.resolve(vpath)
                if path is not None:
                    return path
            return super().data(role)
        elif role == Qt.ToolTipRole:
            vpath = self.data(VarPathItem.VarPathRole)
            if isinstance(vpath, VarPath.AbsPath):
                return vpath.path
            elif isinstance(vpath, VarPath):
                text = f"${{{vpath.name}}}/{vpath.relpath}"
                p = self.resolve(vpath)
                if p is None or not os.path.exists(p):
                    text += " (missing)"
                return text
        elif role == Qt.ForegroundRole:
            vpath = self.data(VarPathItem.VarPathRole)
            if isinstance(vpath, PathItem):
                p = self.resolve(vpath)
                if p is None or not os.path.exists(p):
                    return QColor(Qt.red)
        return super().data(role)


class VarPathItemModel(QStandardItemModel):
    def __init__(self, *args, replacementEnv=types.MappingProxyType({}),
                 **kwargs):
        self.__replacements = types.MappingProxyType(dict(replacementEnv))
        super().__init__(*args, **kwargs)

    def setReplacementEnv(self, env: Mapping[str, str]) -> None:
        self.__replacements = types.MappingProxyType(dict(env))
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1)
        )

    def replacementEnv(self) -> Mapping[str, str]:
        return self.__replacements

    def resolve(self, vpath: PathItem) -> Optional[str]:
        return vpath.resolve(self.replacementEnv())


class RecentPathsControl(QFrame):
    #: Signal emitted when a recent path entry is activated
    activated = Signal(int)
    #: Signal emitted when the index of the current path item changes
    currentIndexChanged = Signal(int)

    def __init__(self, *args, **kwargs):
        frameShape = kwargs.pop("frameShape", QFrame.NoFrame)
        placeholderText = kwargs.pop("placeholderText", "")
        super().__init__(*args, **kwargs)
        self.setFrameShape(frameShape)
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.__label = QLabel("File:")
        self.__model = VarPathItemModel(self)
        self.recent_combo = ItemStyledComboBox(
            self, objectName="recent-combo-box", toolTip="Recent files.",
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            minimumContentsLength=16, placeholderText=placeholderText
        )
        self.recent_combo.currentIndexChanged.connect(self.currentIndexChanged)
        self.recent_combo.setModel(self.__model)
        self.recent_combo.activated.connect(self.activated)
        self.recent_combo.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.browse_button = QPushButton(
            "…", icon=self.style().standardIcon(QStyle.SP_DirOpenIcon),
            toolTip="Browse filesystem", autoDefault=False,
        )
        self.browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.__label)
        layout.addWidget(self.recent_combo)
        layout.addWidget(self.browse_button)

    def setLabelText(self, text: str):
        """Set the label text in front of the recent paths combo box."""
        self.__label.setText(text)

    def labelText(self):
        return self.__label.text()

    def setModel(self, model: QAbstractItemModel) -> None:
        """Set the recent items model."""
        self.recent_combo.setModel(model)

    def model(self) -> QAbstractItemModel:
        return self.recent_combo.model()

    def currentData(self, role=Qt.UserRole) -> Any:
        return self.recent_combo.currentData(role)

    def itemData(self, index: int, role=Qt.UserRole) -> Any:
        return self.recent_combo.itemData(index, role)

    def currentIndex(self) -> int:
        return self.recent_combo.currentIndex()

    def setCurrentIndex(self, index: int) -> None:
        self.recent_combo.setCurrentIndex(index)

    def count(self) -> int:
        return self.recent_combo.count()

    def setPlaceholderText(self, text):
        self.recent_combo.setPlaceholderText(text)

    def placeholderText(self):
        return self.recent_combo.placeholderText()
