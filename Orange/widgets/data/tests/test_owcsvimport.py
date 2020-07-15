import unittest
from unittest import mock
from contextlib import ExitStack

import os
import io
import csv
import json
from typing import Type, TypeVar, Optional

import numpy as np
from numpy.testing import assert_array_equal

from AnyQt.QtCore import QSettings
from AnyQt.QtWidgets import QFileDialog

from orangewidget.tests.utils import simulate
from orangewidget.widget import OWBaseWidget

from Orange.data import DiscreteVariable, TimeVariable, ContinuousVariable, \
    StringVariable
from Orange.tests import named_file
from Orange.widgets.tests.base import WidgetTest, GuiTest
from Orange.widgets.data import owcsvimport
from Orange.widgets.data.owcsvimport import (
    pandas_to_table, ColumnType, RowSpec
)
from Orange.widgets.utils.pathutils import PathItem, samepath
from Orange.widgets.utils.settings import QSettings_writeArray
from Orange.widgets.utils.state_summary import format_summary_details

W = TypeVar("W", bound=OWBaseWidget)


class TestOWCSVFileImport(WidgetTest):
    def create_widget(
            self, cls: Type[W], stored_settings: Optional[dict] = None,
            reset_default_settings=True, **kwargs) -> W:
        if reset_default_settings:
            self.reset_default_settings(cls)
        widget = cls.__new__(cls, signal_manager=self.signal_manager,
                             stored_settings=stored_settings, **kwargs)
        widget.__init__()

        def delete():
            widget.onDeleteWidget()
            widget.close()
            widget.deleteLater()

        self._stack.callback(delete)
        return widget

    def setUp(self):
        super().setUp()
        self._stack = ExitStack().__enter__()
        # patch `_local_settings` to avoid side effects, across tests
        fname = self._stack.enter_context(named_file(""))
        s = QSettings(fname, QSettings.IniFormat)
        self._stack.enter_context(mock.patch.object(
            owcsvimport.OWCSVFileImport, "_local_settings", lambda *a: s
        ))
        self.widget = self.create_widget(owcsvimport.OWCSVFileImport)

    def tearDown(self):
        del self.widget
        self._stack.close()
        super().tearDown()

    def test_basic(self):
        w = self.widget
        w.activate_recent(0)
        w.cancel()

    data_regions_options = owcsvimport.Options(
        encoding="ascii", dialect=csv.excel_tab(),
        columntypes=[
            (range(0, 1), ColumnType.Categorical),
            (range(1, 2), ColumnType.Text),
            (range(2, 3), ColumnType.Categorical),
        ], rowspec=[
            (range(0, 1), RowSpec.Header),
            (range(1, 3), RowSpec.Skipped),
        ],
    )
    data_regions_path = os.path.join(
        os.path.dirname(__file__), "data-regions.tab")

    def _check_data_regions(self, table):
        self.assertEqual(len(table), 3)
        self.assertEqual(len(table), 3)
        self.assertTrue(table.domain["id"].is_discrete)
        self.assertTrue(table.domain["continent"].is_discrete)
        self.assertTrue(table.domain["state"].is_string)
        assert_array_equal(table.X, [[0, 1], [1, 1], [2, 0]])
        assert_array_equal(table.metas,
                           np.array([["UK"], ["Russia"], ["Mexico"]], object))

    def test_restore(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "data-regions.tab")

        w = self.create_widget(
            owcsvimport.OWCSVFileImport,
            stored_settings={
                "_session_items": [
                    (path, self.data_regions_options.as_dict())
                ]
            }
        )
        item = w.current_item()
        self.assertTrue(samepath(item.path(), path))
        self.assertEqual(item.options(), self.data_regions_options)
        out = self.get_output("Data", w)
        self._check_data_regions(out)
        self.assertEqual(out.name, "data-regions")

    def test_restore_from_local(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "data-regions.tab")
        s = owcsvimport.OWCSVFileImport._local_settings()
        s.clear()
        QSettings_writeArray(
            s, "recent", [
                {"path": path,
                 "options": json.dumps(self.data_regions_options.as_dict())}]
        )
        w = self.create_widget(
            owcsvimport.OWCSVFileImport,
        )
        item = w.current_item()
        self.assertIsNone(item)
        simulate.combobox_activate_index(w.recent_combo, 0)
        item = w.current_item()
        self.assertTrue(samepath(item.path(), path))
        self.assertEqual(item.options(), self.data_regions_options)
        data = w.settingsHandler.pack_data(w)
        self.assertEqual(
            data['_session_items_v2'], [
                (PathItem.AbsPath(path).as_dict(),
                 self.data_regions_options.as_dict())
            ],
            "local settings item must be recorded in _session_items_v2 when "
            "activated",
        )
        self._check_data_regions(self.get_output("Data", w))

    def test_summary(self):
        """Check if status bar is updated when data is received"""
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "data-regions.tab")
        widget = self.create_widget(
            owcsvimport.OWCSVFileImport,
            stored_settings={
                "_session_items": [
                    (path, self.data_regions_options.as_dict())
                ]
            }
        )
        output_sum = widget.info.set_output_summary = mock.Mock()
        widget.commit()
        self.wait_until_finished(widget)
        output = self.get_output("Data", widget)
        output_sum.assert_called_with(len(output),
                                      format_summary_details(output))

    data_csv_types_options = owcsvimport.Options(
        encoding="ascii", dialect=csv.excel_tab(),
        columntypes=[
            (range(0, 5), ColumnType.Auto),
        ]
    )

    def test_type_guessing(self):
        """ Check if correct column type is guessed when column type auto """
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "data-csv-types.tab")
        widget = self.create_widget(
            owcsvimport.OWCSVFileImport,
            stored_settings={
                "_session_items": [
                    (path, self.data_csv_types_options.as_dict())
                ]
            }
        )
        widget.commit()
        self.wait_until_finished(widget)
        output = self.get_output("Data", widget)
        domain = output.domain

        self.assertIsInstance(domain["time"], TimeVariable)
        self.assertIsInstance(domain["discrete1"], DiscreteVariable)
        self.assertIsInstance(domain["discrete2"], DiscreteVariable)
        self.assertIsInstance(domain["numeric1"], ContinuousVariable)
        self.assertIsInstance(domain["numeric2"], ContinuousVariable)
        self.assertIsInstance(domain["string"], StringVariable)

    def test_browse(self):
        widget = self.widget
        path = self.data_regions_path
        browse_dialog = widget._browse_dialog
        with mock.patch.object(widget, "_browse_dialog") as r:
            dlg = browse_dialog()
            dlg.setOption(QFileDialog.DontUseNativeDialog)
            dlg.selectFile(path)
            dlg.exec_ = dlg.exec = lambda: QFileDialog.Accepted
            r.return_value = dlg
            with mock.patch.object(owcsvimport.CSVImportDialog, "exec_",
                                   lambda _: QFileDialog.Accepted):
                widget.browse()
        cur = widget.current_item()
        self.assertIsNotNone(cur)
        self.assertTrue(samepath(cur.path(), path))

    def test_browse_for_missing(self):
        missing = os.path.dirname(__file__) + "/this file does not exist.csv"
        widget = self.create_widget(
            owcsvimport.OWCSVFileImport, stored_settings={
                "_session_items": [
                    (missing, self.data_regions_options.as_dict())
                ]
            }
        )
        widget.activate_recent(0)
        dlg = widget.findChild(QFileDialog)
        assert dlg is not None
        # calling selectFile when using native (macOS) dialog does not have
        # an effect - at least not immediately;
        dlg.setOption(QFileDialog.DontUseNativeDialog)
        dlg.selectFile(self.data_regions_path)
        dlg.accept()
        self.assertTrue(samepath(
            self.data_regions_path,
            widget.recent_combo.currentData(owcsvimport.ImportItem.PathRole)
        ))
        self.assertEqual(
            self.data_regions_options.as_dict(),
            widget.recent_combo.currentData(owcsvimport.ImportItem.OptionsRole).as_dict()
        )


class TestImportDialog(GuiTest):
    def test_dialog(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "grep_file.txt")
        d = owcsvimport.CSVImportDialog()
        d.setPath(path)
        opts = owcsvimport.Options(
            encoding="utf-8",
            dialect=owcsvimport.textimport.Dialect(
                " ", "\"", "\\", True, True
            ),
            columntypes=[
                (range(0, 2), ColumnType.Numeric),
                (range(2, 3), ColumnType.Categorical)
            ],
            rowspec=[
                (range(0, 4), RowSpec.Skipped),
                (range(4, 5), RowSpec.Header),
                (range(8, 13), RowSpec.Skipped),
            ]
        )
        d.setOptions(opts)
        d.restoreDefaults()
        opts1 = d.options()
        d.reset()
        opts1 = d.options()


class TestUtils(unittest.TestCase):
    def test_load_csv(self):
        contents = (
            b'1/1/1990,1.0,[,one,\n'
            b'1/1/1990,2.0,],two,\n'
            b'1/1/1990,3.0,{,three,'
        )
        opts = owcsvimport.Options(
            encoding="ascii",
            dialect=csv.excel(),
            columntypes=[
                (range(0, 1), ColumnType.Time),
                (range(1, 2), ColumnType.Numeric),
                (range(2, 3), ColumnType.Text),
                (range(3, 4), ColumnType.Categorical),
                (range(4, 5), ColumnType.Auto),
            ],
            rowspec=[]
        )
        df = owcsvimport.load_csv(io.BytesIO(contents), opts)
        self.assertEqual(df.shape, (3, 5))
        self.assertSequenceEqual(
            list(df.dtypes),
            [np.dtype("M8[ns]"), np.dtype(float), np.dtype(object),
             "category", np.dtype(float)],
        )
        opts = owcsvimport.Options(
            encoding="ascii",
            dialect=csv.excel(),
            columntypes=[
                (range(0, 1), ColumnType.Skip),
                (range(1, 2), ColumnType.Numeric),
                (range(2, 3), ColumnType.Skip),
                (range(3, 4), ColumnType.Categorical),
                (range(4, 5), ColumnType.Skip),
            ],
            rowspec=[
                (range(1, 2), RowSpec.Skipped)
            ]
        )
        df = owcsvimport.load_csv(io.BytesIO(contents), opts)
        self.assertEqual(df.shape, (2, 2))
        self.assertSequenceEqual(
            list(df.dtypes), [np.dtype(float), "category"]
        )
        self.assertSequenceEqual(
            list(df.iloc[:, 0]), [1.0, 3.0]
        )
        self.assertSequenceEqual(
            list(df.iloc[:, 1]), ["one", "three"]
        )

    def test_convert(self):
        contents = (
            b'I, J,  K\n'
            b' , A,   \n'
            b'B,  ,  1\n'
            b'?, ., NA'
        )

        class dialect(csv.excel):
            skipinitialspace = True

        opts = owcsvimport.Options(
            encoding="ascii",
            dialect=dialect(),
            columntypes=[
                (range(0, 1), ColumnType.Text),
                (range(1, 2), ColumnType.Categorical),
                (range(2, 3), ColumnType.Text),

            ],
            rowspec=[(range(0, 1), RowSpec.Header)]
        )
        df = owcsvimport.load_csv(io.BytesIO(contents), opts)
        tb = pandas_to_table(df)

        assert_array_equal(tb.metas[:, 0], ["", "B", "?"])
        assert_array_equal(tb.metas[:, 1], ["", "1", "NA"])
        assert_array_equal(tb.X[:, 0], [0.0, np.nan, np.nan])

        opts = owcsvimport.Options(
            encoding="ascii",
            dialect=dialect(),
            columntypes=[
                (range(0, 1), ColumnType.Categorical),
                (range(1, 2), ColumnType.Categorical),
                (range(2, 3), ColumnType.Numeric),
            ],
            rowspec=[(range(0, 1), RowSpec.Header)]
        )
        df = owcsvimport.load_csv(io.BytesIO(contents), opts)
        tb = pandas_to_table(df)

        assert_array_equal(tb.X[:, 0], [np.nan, 0, np.nan])
        assert_array_equal(tb.X[:, 1], [0, np.nan, np.nan])
        assert_array_equal(tb.X[:, 2], [np.nan, 1, np.nan])

    def test_sniff_csv(self):
        f = io.StringIO("A|B|C\n1|2|3\n1|2|3")
        dialect, header = owcsvimport.sniff_csv(f)
        self.assertEqual(dialect.delimiter, "|")
        self.assertTrue(header)
        with self.assertRaises(csv.Error):
            owcsvimport.sniff_csv(f, delimiters=["."])


if __name__ == "__main__":
    unittest.main()
