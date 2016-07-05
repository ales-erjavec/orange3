import os
import tempfile
from datetime import date, timedelta
from types import SimpleNamespace

import unittest
from unittest import mock

from AnyQt.QtCore import QCoreApplication
try:
    from AnyQt.QtTest import QSignalSpy
    HAS_QSIGNALSPY = True
except ImportError:
    HAS_QSIGNALSPY = False

from .. import updater as updatemanager

date_to_string = updatemanager._date_to_string


class mocksettings(object):
    def __init__(self, settings):
        self._settings = settings

    def value(self, key, defaultValue=None, type=None):
        try:
            val = self._settings[key]
        except KeyError:
            return defaultValue
        if type is not None and not isinstance(val, type):
            # this is not exactly how PyQt's QSettings.value works,
            # make sure to use consistent types in passed settings.
            try:
                return type(val)
            except (TypeError, ValueError):
                return None
        else:
            return val

    def setValue(self, key, val):
        self._settings[key] = val

    def sync(self):
        pass


def n_days_ago(n=1):
    return date_to_string(date.today() - timedelta(days=n))


class TestUpdater(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix=__name__)
        self.tempdir = self._tempdir.name

        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        self.app = app

    def tearDown(self):
        self._tempdir.cleanup()
        del self._tempdir
        del self.app

    def _prepare_updater(self, configoverride):
        config = {
            "enabled": True,
            "check-period": 1,
            "last-notification-date": n_days_ago(n=1),
            "remind-period": 1,
        }
        config.update(**configoverride)
        updater = updatemanager.UpdateManager()
        updater._settings = lambda: mocksettings(config)
        updater.cachePath = lambda: os.path.join(self.tempdir, "cache")
        updater.fetchUpdateInfo = mock.MagicMock(
            return_value=[
                SimpleNamespace(
                    name="Orange3",
                    version="42.42.42",
                    display_name=None,
                    category="core",
                    download_url="https://example.org/",
                    release_notes_url="")
            ]
        )
        return updater

    def test_autostart(self):
        updater = self._prepare_updater(
            {"check-period": 0,
             "last-check-date": "1970-01-01",
             "last-notification-date": n_days_ago(n=2),
             "remind-period": 1}
        )
        self.assertSequenceEqual(updater.updateCandidates(), [])
        recordrequest = mock.MagicMock()
        didstart = updater.autoStart()
        self.assertTrue(didstart)
        self.assertTrue(updater.reason() & updater.AutoStart)
        self.assertTrue(not updater.reason() & updater.NormalStart)

        updater.updateNotificationRequested.connect(recordrequest)
        if HAS_QSIGNALSPY:
            spy = QSignalSpy(updater.finished)
            self.assertTrue(spy.wait(1000))
        else:
            updater.finished.connect(self.app.quit)
            self.app.exec_()

        if updater.exception():
            raise updater.exception()

        updater.fetchUpdateInfo.assert_called_once_with()
        recordrequest.assert_called_once_with()

        settings = updater._settings()._settings
        self.assertEqual(settings.get("last-check-date", ""),
                         n_days_ago(n=0))
        self.assertEqual(settings.get("last-notification-date", ""),
                         n_days_ago(n=0))

        updater = self._prepare_updater(
            {"check-period": 1,
             "last-check-date": n_days_ago(n=0),
             "remind-period": 1,
             "last-notification-date": n_days_ago(n=0)}
        )
        didstart = updater.autoStart()
        self.assertFalse(didstart)

    def test_remind(self):
        updater = self._prepare_updater(
            {"check-period": 7,
             "last-check-date": n_days_ago(n=1),
             "last-notification-date": n_days_ago(n=2),
             "remind-period": 1}
        )

        recordrequest = mock.MagicMock()
        updater.updateNotificationRequested.connect(recordrequest)

        with mock.patch.object(
                updater, "updateCandidates",
                return_value=[
                    updatemanager.UpdateCandidate(
                        name="Orange Canvas",
                        latest_version="42.42.42",
                        installed_version="1.1.1",
                        category="core",
                        skipped=False,
                        meta={
                            "download-url": "https://example.org/",
                            "release-notes-url": ""})
                    ]
                ):
            didstart = updater.autoStart()
            self.assertTrue(didstart)

            QCoreApplication.processEvents()

            updater.fetchUpdateInfo.assert_not_called()
            recordrequest.assert_called_once_with()
