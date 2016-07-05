"""
Update checker
--------------

Check for updates on an (un)timely basis

"""
import os
import logging
import enum
import pickle

from datetime import date, datetime, timedelta

from PyQt4.QtCore import QObject, QSettings, QTimer
from PyQt4.QtCore import pyqtSlot as Slot, pyqtSignal as Signal

from .. import config

from ..utils import concurrent as qconcurrent

import pkg_resources

parse_version = pkg_resources.parse_version

log = logging.getLogger(__name__)


class State(enum.Enum):
    NoState = "NoState"
    Ready = "Ready"
    Running = "Running"
    Finished = "Finished"
    Error = "Error"


class NotificationFlag(enum.IntEnum):
    #: The notification is a new notification, that has never before been
    #: presented to the user (e.g. an new update has been found)
    New = 1
    #: This notification is a result of an action triggered by an
    #: automatic scheduler (i.e. not a result of an explicit user request).
    Spontaneous = 2

_DATEFORMAT = "%Y-%m-%d"


def _date_parse(string):
    """Parse a YYYY-MM-DD formatted date and return it as `datatime.date`."""
    try:
        return datetime.strptime(string, _DATEFORMAT).date()
    except ValueError:
        return None


def _date_to_string(date):
    """Return a YYYY-MM-DD formatted date string."""
    return date.strftime(_DATEFORMAT)


class UpdateManager(QObject):
    """
    A utility class fo running periodic (configurable) update checks
    and notifications.
    """
    #: Update check has started
    started = Signal()
    #: Update check has finished
    finished = Signal()

    #: A visible update notification should be displayed to the user
    #: This signal is emitted as a result of a completed update check
    #: or is a rescheduled notification of a previously dismissed
    #: update
    updateNotificationRequested = Signal(object)  # NotificationFlag)

    class StartReason(enum.IntEnum):
        """Update check reason."""
        NoReason = 0     #: Did not start
        AutoStart = 1    #: Started as a result of `autoStart`
        NormalStart = 2  #: Started as a result of `start`

    NoReason, AutoStart, NormalStart = StartReason

    def __init__(self, *args, **kwargs):
        super(UpdateManager, self).__init__(*args, **kwargs)
        self.__startreason = UpdateManager.NoReason
        self.__state = State.Ready
        self.__items = []

    def cachePath(self):
        """
        Returns
        -------
        path : str
            A path to the local cache store.
        """
        datadir = config.data_dir()
        return os.path.join(datadir, "UpdateManager1.cache")

    def autoStart(self):
        """
        Start the update check in the background if scheduled.

        Return False if no checks are scheduled; i.e if updates are disabled
        or not enough time has passed since the last check.
        """
        settings = QSettings()
        settings.beginGroup("application/update")
        updateperiod = settings.value("check-period", defaultValue=1, type=int)
        if updateperiod < 0:
            return False

        lastcheck = settings.value("last-check-date", "", type=str)
        lastcheck = _date_parse(lastcheck)
        if lastcheck is None:
            lastcheck = date.fromtimestamp(0)

        lastdelta = date.today() - lastcheck
        log.debug("Time from last update check: %s (%s)", lastdelta, lastcheck)
        if lastdelta >= timedelta(days=updateperiod):
            reason = self.__startreason
            self.start()
            self.__startreason = reason | UpdateManager.AutoStart
            return True
        else:
            QTimer.singleShot(0, self.__emitNotification)

    def __emitNotification(self):
        if self.__state == State.Running:
            # start was called again; will emit with updated information
            # when it completes.
            return

        spontaneous = not bool(self.__startreason & UpdateManager.NormalStart)

        s = QSettings()
        s.beginGroup("application/update")

        lastnotified = s.value(
            "last-notification-date", defaultValue="", type=str)
        lastnotified = _date_parse(lastnotified)

        if lastnotified is None:
            lastnotified = date.fromtimestamp(0)

        nagperiod = s.value("nag-period", defaultValue=1, type=int)

        if date.today() - lastnotified < timedelta(days=nagperiod):
            do_not_nag = True
        else:
            do_not_nag = False

        lastnotifiedver = s.value(
            "last-notified-version", defaultValue="", type=str)
        lastnotifiedver = parse_version(lastnotifiedver)
        current = config.application_version()
        latest = self.latesetVersion()
        # is this a new version for which we have never before emitted a
        # notification
        isnew = lastnotifiedver < parse_version(latest)
        if isnew:
            # ignore the nag suppression period when first receiving a new
            # latest version string
            do_not_nag = False

        skipped = latest in self.skippedVersions()

        flags = 0
        if spontaneous:
            flags |= NotificationFlag.Spontaneous
        if isnew:
            flags |= NotificationFlag.New

        if (not spontaneous or  # user requested check
                (not skipped and not do_not_nag and
                 parse_version(latest) > parse_version(current))):
            self.updateNotificationRequested.emit(flags)

            s.setValue("last-notification-date", _date_to_string(date.today()))
            s.setValue("last-notified-version", latest)

    def start(self):
        """
        Start the update check in a background thread.

        Return True if a check was scheduled, or False if one is already in
        progress.
        """
        if self.__state != State.Running:
            self.__state = State.Running
            f = qconcurrent.submit(config.fetch_updates1)
            w = qconcurrent.FutureWatcher(parent=self)
            w.done.connect(self.__ondone)
            w.setFuture(f)

            self.__startreason |= UpdateManager.NormalStart
            self.started.emit()
            return True
        else:
            return False

    def reason(self):
        """
        Reason for running the check.

        Returns
        -------
        reason : UpdateManager.Reason
            The reason the update check was run (by `autoStart` or `start`)
        """
        return self.__startreason

    @Slot(object)
    def __ondone(self, future):
        # future with the latest application version has completed
        assert future.done()
        try:
            items = future.result()
        except qconcurrent.CancelledError:
            self.__state = State.Ready
            self.finished.emit()
            return
        except Exception as ex:
            self.__state = State.Error
            self.__r = ex
            log.exception("An update task failed with an error")
            self.finished.emit()
            return
        else:
            self.__state = State.Ready

        self.__items = items
        try:
            os.makedirs(os.path.dirname(self.cachePath()), exist_ok=True)
            with open(self.cachePath(), "wb") as f:
                pickle.dump(self.__items, f)
        except OSError:
            pass
        s = QSettings()
        s.beginGroup("application/update")
        s.setValue("last-check-date", _date_to_string(date.today()))
        s.sync()
        self.finished.emit()
        self.__emitNotification()

    def updateItems(self):
        try:
            with open(self.cachePath(), "rb") as f:
                items = pickle.load(f)
        except FileNotFoundError:
            items = []
        except pickle.UnpicklingError:
            items = []
        return items

    def latesetVersion(self):
        """
        Return the latest (cached) available application version.

        Returns
        -------
        version : str
        """
        items = self.updateItems()
        item = [item for item in items
                if item.name == config.application_name()]
        if item:
            return item[0].version
        else:
            return config.application_version()

    def noteSkipped(self, version):
        """
        Mark the version as skipped, and never again emit spontaneous update
        notifications for it.

        Parameters
        ----------
        version : str
        """
        fname = os.path.join(config.cache_dir(), "update-skip.txt")
        try:
            with open(fname, "x") as f:
                f.write("# Versions which the user explicitly skipped "
                        "(one per line)\n")
        except FileExistsError:
            pass

        with open(fname, "a") as f:
            f.write(version + '\n')

    def skippedVersions(self):
        """
        Return a list of all versions which are noted as skipped.

        Returns
        -------
        versions : List[str]
        """
        # Did the user previously select the 'Skip this version'
        fname = os.path.join(config.cache_dir(), "update-skip.txt")
        try:
            with open(fname, "r") as f:
                return [ver.rstrip(" \n") for ver in f
                        if not ver.startswith("#")]
        except FileNotFoundError:
            return []
