import unittest

from Orange.widgets.unsupervised.utils import (
    MappedColumnProxyModel
)

from AnyQt.QtCore import Qt
from Orange.widgets.utils.itemmodels import create_list_model


class TestCase(unittest.TestCase):
    def test_proxy(self):
        source = create_list_model([
            {Qt.DisplayRole: "A", Qt.UserRole: 1},
            {Qt.DisplayRole: "B", Qt.UserRole: 2},
            {Qt.UserRole: 3},
            {}
        ])
        model = MappedColumnProxyModel()
        self.assertEqual(model.rowCount(), 0)
        self.assertEqual(model.columnCount(), 2)
        model.setSourceModel(source)
        model.setMappedRoles(
            {Qt.EditRole: Qt.UserRole, Qt.UserRole - 1: Qt.DisplayRole}
        )

        self.assertEqual(model.rowCount(), 4)
        self.assertEqual(model.columnCount(), 2)

        self.assertEqual(model.index(0, 0).data(Qt.DisplayRole), "A")
        self.assertEqual(model.index(3, 0).data(Qt.DisplayRole), None)

        self.assertEqual(model.index(0, 1).data(Qt.EditRole), 1)
        self.assertEqual(model.index(0, 1).data(Qt.UserRole - 1), "A")
        self.assertEqual(model.index(3, 1).data(Qt.EditRole), None)
        self.assertEqual(model.index(3, 1).data(Qt.UserRole - 1), None)

        self.assertTrue(model.setData(model.index(0, 1), 2, Qt.EditRole))
        self.assertEqual(model.index(0, 1).data(Qt.EditRole), 2)
        self.assertEqual(model.index(0, 0).data(Qt.UserRole), 2)

        self.assertTrue(model.setData(model.index(3, 1), 3, Qt.EditRole))
        self.assertEqual(model.index(3, 1).data(Qt.EditRole), 3)
        self.assertEqual(model.index(3, 0).data(Qt.UserRole), 3)

        ind = model.index(0, 0)
        buddy = model.buddy(ind)
        self.assertTrue(buddy.isValid())
        self.assertEqual(buddy.column(), 1)
        self.assertEqual(model.buddy(buddy), buddy)

        model.setSourceModel(None)
        model.setMappedRoles({})
