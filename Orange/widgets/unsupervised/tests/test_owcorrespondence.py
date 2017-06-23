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


def reflect(v1, v2):
    """
    Reflect v2 so it matches the sign of `v1` in the first element.
    """
    s1 = np.sign(v1.flat[0])
    s2 = np.sign(v2.flat[0])
    return (v1, -v2) if s1 * s2 == -1 else (v1, v2)


def assert_almost_equal_to_reflection(actual, desired, *args, **kwargs):
    actual, desired = reflect(desired, actual)
    assert_almost_equal(actual, desired, *args, **kwargs)


class TestCA(unittest.TestCase):
    def test_smoke(self):
        smoke = np.array([
            # none light medium heavy
            [    4,    2,     3,     2],  # SM
            [    4,    3,     7,     4],  # JM
            [   25,   10,    12,     4],  # SE
            [   18,   24,    33,    13],  # JE
            [   10,    6,     7,     2],  # SC
        ])
        # > c = ca(smoke)
        # > coo <- plot(ca)
        # > round(coo$rows, 3)
        expected_rows = np.array([
            #  Dim1    Dim2
            [-0.066, -0.194], # SM
            [ 0.259, -0.243], # JM
            [-0.381, -0.011], # SE
            [ 0.233,  0.058], # JE
            [-0.201,  0.079], # SC
        ])

        # > round(roo$cols, 3)
        expected_cols = np.array([
            #  Dim1    Dim2
            [-0.393, -0.030], # none
            [ 0.099,  0.141], # light
            [ 0.196,  0.007], # medium
            [ 0.294, -0.198], # heavy
        ])

        smoke = np.array(smoke, dtype=float)
        ca = correspondence(smoke)
        dims = [0, 1]
        x = np.round(ca.row_factors[:, dims], 3)
        y = np.round(ca.col_factors[:, dims], 3)
        assert_almost_equal_to_reflection(x[:, 0], expected_rows[:, 0])
        assert_almost_equal_to_reflection(x[:, 1], expected_rows[:, 1])
        assert_almost_equal_to_reflection(y[:, 0], expected_cols[:, 0])
        assert_almost_equal_to_reflection(y[:, 1], expected_cols[:, 1])
        # > round(c$rowcoord, 3)[,1:2]
        expected_rows = np.array([
            #  Dim1    Dim2
            [-0.241, -1.936],  # SM
            [ 0.947, -2.431],  # JM
            [-1.392, -0.107],  # SE
            [ 0.852,  0.577],  # JE
            [-0.735,  0.788],  # SC
        ])
        # > round(c$colcoord, 3)[,1:2]
        expected_cols = np.array([
            #  Dim1    Dim2
            [-1.438, -0.305],  # none
            [ 0.364,  1.409],  # light
            [ 0.718,  0.074],  # medium
            [ 1.074, -1.976],  # heavy
        ])
        x = np.round(ca.row_standard_coordinates[:5, dims], 3)
        y = np.round(ca.col_standard_coordinates[:4, dims], 3)
        assert_almost_equal_to_reflection(x[:, 0], expected_rows[:, 0])
        assert_almost_equal_to_reflection(x[:, 1], expected_rows[:, 1])
        assert_almost_equal_to_reflection(y[:, 0], expected_cols[:, 0])
        assert_almost_equal_to_reflection(y[:, 1], expected_cols[:, 1])
        # > round(c$rowinertia, 3)
        expected = np.array([0.003, 0.012, 0.038, 0.026, 0.006,])

        inertia = np.round(ca.row_inertia_, 3)
        assert_almost_equal(inertia, expected)
        # > round(c$colinertia, 3)
        expected = np.array([0.049, 0.007, 0.013, 0.016])
        inertia = np.round(ca.col_inertia_, 3)
        assert_almost_equal(inertia, expected)

        assert_almost_equal(ca.inertia_of_axis, ca.D ** 2)

    def test_punctuation(self):
        # Author punctuation
        # (from 'Correspondence Analysis - Herve Abdi Lynne J. Williams')
        data = np.array([
            #period, comma, other
            [  7836,  13112,  6026],  # "Rousseau"
            [ 53655, 102383, 42413],  # "Chateaubriand"
            [115615, 184541, 59226],  # "Hugo"
            [161926, 340479, 62754],  # "Zola"
            [ 38177, 105101, 12670],  # "Proust"
            [ 46371,  58367, 14299],  # "Giraudoux"
        ])
        # Fig 12.
        expected = np.array([
            [ 0.2398,  0.0741],
            [ 0.1895,  0.1071],
            [ 0.1033, -0.0297],
            [-0.0918,  0.0017],
            [-0.2243,  0.0631],
            [ 0.0475, -0.1963],
        ])
        ca = correspondence(data)
        assert_almost_equal_to_reflection(
            ca.row_factors[:, 0], expected[:, 0], decimal=4
        )
        assert_almost_equal_to_reflection(
            ca.row_factors[:, 1], expected[:, 1], decimal=4
        )
        assert_almost_equal(ca.D[:2] ** 2, [0.0178, 0.0056], decimal=3)

    def test_mca(self):
        pass

    def test_cross_tabulate(self):
        table = Table("zoo")
        hair = table.domain["hair"]
        type_ = table.domain["type"]
        xtab = cross_tabulate(
            table, table.domain.variables, [table.domain.class_var])
        # assert_array_equal(xtab, [0, 1])
        xtab = cross_tabulate(table, [], [])
        self.assertEqual(xtab.shape, (0, 0))

        xtab = cross_tabulate(table, [hair], [])
        self.assertEqual(xtab.size, 0)
        self.assertEqual(xtab.shape, (len(hair.values), 0))
        xtab = cross_tabulate(table, [], [hair])
        self.assertEqual(xtab.shape, (0, len(hair.values)))
        xtab = cross_tabulate(table, [type_], [hair])
        self.assertEqual(xtab.shape, (len(type_.values), len(hair.values)))
        self.assertEqual(xtab.sum(), len(table))
        self.assertSequenceEqual(
            xtab.sum(axis=1).tolist(), [4, 20, 13, 8, 10, 41, 5])

