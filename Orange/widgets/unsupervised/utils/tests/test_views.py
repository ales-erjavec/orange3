import enum

from AnyQt.QtCore import Qt, QItemSelectionModel, QLocale
from AnyQt.QtWidgets import QApplication, QMenu, QStyleOptionViewItem, \
    QComboBox, QWidget
from AnyQt.QtTest import QTest, QSignalSpy

from Orange.widgets.tests.base import GuiTest

from Orange.widgets.unsupervised.utils import (
    AnalysisRoleView, MappedColumnProxyModel,
    EnumItemDelegate)
from Orange.widgets.utils.itemmodels import create_list_model


class Foo(enum.Enum):
    Bar = "Bar"
    Baz = "Baz"


class TestCase(GuiTest):
    def test_view_edit(self):
        view = AnalysisRoleView(
            selectionMode=AnalysisRoleView.ExtendedSelection,
        )
        model = create_list_model([
            {Qt.DisplayRole: "A", Qt.UserRole: Foo.Bar},
            {Qt.DisplayRole: "B", Qt.UserRole: Foo.Bar},
            {Qt.DisplayRole: "Z", Qt.UserRole: None},
        ])
        mapper = MappedColumnProxyModel()
        mapper.mappedRoles()
        mapper.setMappedRoles(
            {Qt.UserRole: Qt.UserRole, Qt.DisplayRole: Qt.UserRole,
             Qt.EditRole: Qt.UserRole}
        )
        mapper.setSourceModel(model)
        index = mapper.index(0, 1)
        self.assertEqual(index.data(Qt.DisplayRole), Foo.Bar)
        self.assertEqual(index.data(Qt.UserRole), Foo.Bar)

        view.setModel(mapper)
        states = create_list_model([
            {Qt.DisplayRole: "Bar", Qt.UserRole: Foo.Bar},
            {Qt.DisplayRole: "Bar", Qt.UserRole: Foo.Baz},
        ])
        view.setStateModel(states)
        view.setEditRole(Qt.UserRole)

        selmodel = view.selectionModel()  # type: QItemSelectionModel
        index = mapper.index(0, 1)
        self.assertEqual(index.data(Qt.UserRole), Foo.Bar)
        selmodel.select(index,
                        QItemSelectionModel.Select | QItemSelectionModel.Rows)
        QTest.keyPress(view.viewport(), Qt.Key_Left)
        self.assertEqual(index.data(Qt.UserRole), Foo.Baz)

        QTest.keyPress(view.viewport(), Qt.Key_Down, Qt.AltModifier)
        f = QApplication.activePopupWidget()
        self.assertIsInstance(f, QMenu)
        f = f  # type: QMenu
        a = f.actions()[0]
        self.assertEqual(a.data(), Foo.Bar)
        a.trigger()
        self.assertEqual(index.data(Qt.UserRole), Foo.Bar)

        view.grab()

    def test_enum_delegate(self):
        delegate = EnumItemDelegate()
        self.assertEqual(delegate.displayText(Foo.Bar, QLocale()), "Bar")
        parent = QWidget()
        opt = QStyleOptionViewItem()
        opt.initFrom(parent)
        model = create_list_model(
            [{Qt.DisplayRole: Foo.Bar, Qt.EditRole: Foo.Bar}]
        )
        w = delegate.createEditor(parent, opt, model.index(0, 0))
        delegate.setEditorData(w, model.index(0, 0))

        assert isinstance(w, QComboBox)
        self.assertEqual(w.count(), len(Foo))

        w.setCurrentIndex(1)
        delegate.setModelData(w, model, model.index(0, 0))

        self.assertEqual(model.index(0, 0).data(Qt.EditRole), Foo.Baz)
