from typing import Any

from AnyQt.QtCore import (
    Qt, pyqtSignal as Signal, pyqtSlot as Slot, pyqtProperty as Property
)
from AnyQt.QtWidgets import QComboBox

from orangewidget.utils.combobox import ComboBoxSearch, ComboBox

__all__ = [
    "ComboBoxSearch", "ComboBox", "EnumComboBox",
]


class EnumComboBox(QComboBox):
    """
    A combo box mapping the current `itemData(..., Qt.UserRole)` in the
    `currentValue` property. The values are assumed to be not None and
    unique across all items.
    """
    #: Emitted when the user item data  changes (Qt.UserRole for the current
    #: index). Note that no signal is emitted when current index is  set to -1.
    currentValueChanged = Signal(object)

    def setCurrentValue(self, value):
        # type: (Any) -> None
        """
        Set (select) the current index by value.
        """
        index = self.findData(value, Qt.UserRole, Qt.MatchExactly)
        if index != -1:
            self.setCurrentIndex(index)

    def currentValue(self):
        # type: () -> Any
        """
        Return the current value (the itemData(currentIndex(), Qt.UserRole))
        """
        if self.currentIndex() < 0:
            return None
        else:
            return self.itemData(self.currentIndex(), Qt.UserRole)

    currentValue_ = Property(
        object, currentValue, setCurrentValue, notify=currentValueChanged,
        user=True
    )

    def __init__(self, *args, **kwargs):
        currentIndexChanged = kwargs.pop("currentIndexChanged", None)
        super().__init__(*args, **kwargs)
        # connect first to ensure consistent order of emitted signals
        self.currentIndexChanged.connect(self.__emitCurrentChanged)
        if currentIndexChanged is not None:
            self.currentIndexChanged.connect(currentIndexChanged)

    @Slot(int)
    def __emitCurrentChanged(self, index):
        if index != -1:
            self.currentValueChanged.emit(self.itemData(index, Qt.UserRole))
