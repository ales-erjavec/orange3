# pylint: disable=all
from AnyQt.QtTest import QSignalSpy

from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.combobox import EnumComboBox


class TestEnumComboBox(GuiTest):
    def test_signals(self):
        cb = EnumComboBox()
        spy = QSignalSpy(cb.currentValueChanged)
        cb.addItem("A", userData=ord("A"))
        self.assertEqual(list(spy), [[ord("A")]])
        self.assertEqual(cb.currentValue(), ord("A"))
        cb.addItem("B", userData=ord("B"))
        cb.setCurrentValue(ord("B"))
        self.assertEqual(list(spy), [[ord("A")], [ord("B")]])
        self.assertEqual(cb.currentValue(), ord("B"))
        cb.clear()
        self.assertEqual(list(spy), [[ord("A")], [ord("B")]])
        self.assertEqual(cb.currentValue(), None)
