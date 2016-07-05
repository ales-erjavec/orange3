import os
import sys
import unittest
from unittest import mock

from PyQt4.QtCore import Qt, QObject, QCoreApplication
from .. import updater as updatemanager
from ... import config


class _BaseCoreTest(unittest.TestCase):
    def setUp(self):
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        self.app = app

    def tearDown(self):
        del self.app

    def test(self):
        updater = updatemanager.UpdateManager()
        with mock.patch.object(updater, "fetch_latest_version",
                               return_value="42.42.42")
            updater.start()
            res = updater.latest_version_f.results()

        updater.fetch_latest_version.assert_called_once_with()
        # updater

    def test_update_history(self):
        updatemanager.UpdateManager()
        p = updatemanager.version_record
