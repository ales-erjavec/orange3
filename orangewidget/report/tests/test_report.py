import unittest
from unittest.mock import patch
from importlib import import_module
import os
import warnings
import tempfile

import AnyQt
from AnyQt.QtGui import QFont, QBrush
from AnyQt.QtCore import Qt

from orangewidget.report.owreport import OWReport
from orangewidget import gui
from orangewidget.widget import OWWidget
from orangewidget.tests.base import GuiTest


class TstWidget(OWWidget):
    def send_report(self):
        self.report_caption("AA")


class TestReport(GuiTest):
    def test_report(self):
        count = 5
        rep = OWReport()
        for _ in range(count):
            widget = TstWidget()
            widget.create_report_html()
            rep.make_report(widget)
        self.assertEqual(rep.table_model.rowCount(), count)

    def test_save_report_permission(self):
        """
        Permission Error may occur when trying to save report.
        GH-2147
        """
        rep = OWReport()
        filenames = ["f.report", "f.html"]
        for filename in filenames:
            with patch("orangewidget.report.owreport.open",
                       create=True, side_effect=PermissionError),\
                    patch("AnyQt.QtWidgets.QFileDialog.getSaveFileName",
                          return_value=(filename, 'HTML (*.html)')),\
                    patch("AnyQt.QtWidgets.QMessageBox.exec_",
                          return_value=True), \
                    patch("orangewidget.report.owreport.log.error") as log:
                rep.save_report()
                log.assert_called()

    def test_save_report(self):
        rep = OWReport()
        widget = TstWidget()
        widget.create_report_html()
        rep.make_report(widget)
        temp_dir = tempfile.mkdtemp()
        temp_name = os.path.join(temp_dir, "f.report")
        try:
            with patch("AnyQt.QtWidgets.QFileDialog.getSaveFileName",
                       return_value=(temp_name, 'Report (*.report)')), \
                    patch("AnyQt.QtWidgets.QMessageBox.exec_",
                          return_value=True):
                rep.save_report()
        finally:
            os.remove(temp_name)
            os.rmdir(temp_dir)


if __name__ == "__main__":
    unittest.main()
