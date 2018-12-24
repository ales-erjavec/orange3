# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import enum
import unittest
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_almost_equal, assert_array_equal

from AnyQt.QtCore import Qt, QLocale, QItemSelectionModel
from AnyQt.QtTest import QTest

from Orange.widgets.utils.itemmodels import create_list_model, select_rows
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owcorrespondence import (
    OWCorrespondenceAnalysis, correspondence, cross_tabulate,
    CATypes, EnumItemDelegate, AnalysisRoleView, MappedColumnProxyModel
)


class TestOWCorrespondence(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCorrespondenceAnalysis)
        self.data = Table("titanic")

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.widget.Inputs.data,
                         Table.from_domain(Table("iris").domain))
        self.assertTrue(self.widget.Error.empty_data.is_shown())
        self.assertIsNone(self.widget.data)

    def test_data_values_in_column(self):
        """
        Check that the widget does not crash when:
        1) Domain has a two or more discrete variables but less than in a table
        2) There is at least one NaN value in a column.
        GH-2066
        """
        table = Table.from_list(
            Domain(
                [ContinuousVariable("a"),
                 DiscreteVariable("b", values=("t", "f")),
                 DiscreteVariable("c", values=("y", "n")),
                 DiscreteVariable("d", values=("k", "l", "z"))]
            ),
            list(zip(
                [42.48, 16.84, 15.23, 23.8],
                ["t", "t", "", "f"],
                "yyyy",
                "klkk"
            )))
        self.send_signal(self.widget.Inputs.data, table)

    def test_data_one_value_zero(self):
        """
        Check that the widget does not crash on discrete attributes with only
        one value.
        GH-2149
        """
        table = Table.from_list(
            Domain(
                [DiscreteVariable("a", values=("0", ))]
            ),
            [(0,), (0,), (0,)]
        )
        self.send_signal(self.widget.Inputs.data, table)

    def test_no_discrete_variables(self):
        """
        Do not crash when there are no discrete (categorical) variable(s).
        GH-2723
        """
        table = Table.from_list(
            Domain(
                [ContinuousVariable("a")]
            ),
            [(1,), (2,), (3,)]
        )
        self.assertFalse(self.widget.Error.no_disc_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, table)
        self.assertTrue(self.widget.Error.no_disc_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_disc_vars.is_shown())

        self.send_signal(self.widget.Inputs.data, table)
        self.assertTrue(self.widget.Error.no_disc_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, Table("iris"))
        self.assertFalse(self.widget.Error.no_disc_vars.is_shown())

    def test_outputs(self):
        w = self.widget

        self.assertIsNone(self.get_output(w.Outputs.coordinates), None)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertTupleEqual(self.get_output(w.Outputs.coordinates).X.shape,
                              (6, 2))
        select_rows(w.varview, [0, 1, 2])
        w.commit()
        self.assertTupleEqual(self.get_output(w.Outputs.coordinates).X.shape,
                              (8, 8))
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.coordinates), None)

    def test_mca(self):
        table = Table("smokers_ct")
        w = self.widget
        w.set_analysis_type(CATypes.CA)
        w.set_analysis_type(CATypes.MCA)
        w.set_data(table)
        w.grab()
        w.handleNewSignals()
        w.grab()

    def test_contingency_input(self):
        table = Table("smokers_ct")
        X = cross_tabulate(table, [table.domain[0]], [table.domain[1]])
        cont = Table.from_numpy(None, X)
        w = self.widget
        w.set_contingency(cont)
        w.handleNewSignals()

        cont = Table.from_numpy(
            None, X, metas=[[v] for v in table.domain[0].values])
        w.set_contingency(cont)
        w.handleNewSignals()

    def test_contingency_input_sparse(self):
        table = Table("smokers_ct")
        X = cross_tabulate(table, [table.domain[0]], [table.domain[1]])
        X = sp.csc_matrix(X)
        cont = Table.from_numpy(
            None, X, metas=[[v] for v in table.domain[0].values])
        assert cont.is_sparse()
        w = self.widget
        w.set_contingency(cont)
        w.handleNewSignals()

    def test_model_edit(self):
        delegate = EnumItemDelegate()
        self.assertEqual(delegate.displayText(Foo.Bar, QLocale()), "Bar")

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
            {Qt.UserRole: Qt.UserRole, Qt.DisplayRole: Qt.UserRole}
        )
        mapper.setSourceModel(model)
        index = mapper.index(0, 1)
        self.assertEqual(index.data(Qt.DisplayRole), Foo.Bar)
        self.assertEqual(index.data(Qt.UserRole), Foo.Bar)

        view.setModel(mapper)
        selmodel = view.selectionModel()  # type: QItemSelectionModel
        index = mapper.index(0, 1)
        self.assertEqual(index.data(Qt.UserRole), Foo.Bar)
        selmodel.select(index, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        QTest.keyPress(view.viewport(), Qt.Key_Left)
        self.assertEqual(index.data(Qt.UserRole), Foo.Baz)


class Foo(enum.Enum):
    Bar = "Bar"
    Baz = "Bar"
