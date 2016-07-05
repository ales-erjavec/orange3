"""
Update checker
--------------

Check for updates on an periodic basis

"""
# TODO: the useful library code should be in a package/module lower then this
# TODO: extend to report all updateable items (i.e. add-ons)
# TODO: unification with add-ons management

import os
import logging
import enum
import pickle

from collections import namedtuple
from datetime import date, datetime, timedelta

from AnyQt.QtCore import QObject, QSettings, QTimer
from AnyQt.QtCore import pyqtSlot as Slot, pyqtSignal as Signal

from .. import config

from ..utils import qconcurrent

import pkg_resources

parse_version = pkg_resources.parse_version

log = logging.getLogger(__name__)

_DATEFORMAT = "%Y-%m-%d"


def _date_parse(string):
    """
    Parse a date string in YYYY-MM-DD format and return it as `datetime.date`.

    If the string is not valid, return `None`
    """
    try:
        return datetime.strptime(string, _DATEFORMAT).date()
    except ValueError:
        return None


def _date_to_string(date):
    """
    Return a YYYY-MM-DD formatted date string.
    """
    return date.strftime(_DATEFORMAT)


class UpdateCandidate(
        namedtuple(
            "UpdateCandidate",
            ["name",
             "latest_version",
             "installed_version",
             "category",
             "display_name",
             "skipped",
             "meta",
            ])):
    def __new__(cls, name, latest_version, installed_version, category, *,
                display_name=None, skipped=False, meta=None):
        if meta is None:
            meta = {}
        return super().__new__(cls, name, latest_version, installed_version,
                               category, display_name or name, skipped, meta)


class State(enum.Enum):
    Ready = "Ready"
    Running = "Running"
    Finished = "Finished"
    Error = "Error"


class UpdateManager(QObject):
    """
    A utility class for running periodic (configurable) update checks
    and notifications.
    """
    #: Update check has started
    started = Signal()
    #: Update check has finished
    finished = Signal()

    #: A visible update notification should be displayed to the user.
    #: This signal is emitted as a result of a completed update check
    #: or is a rescheduled notification of a previously dismissed
    #: update notification
    updateNotificationRequested = Signal()

    class StartReason(enum.IntEnum):
        """Update check reason."""
        NoReason = 0     #: Did not start
        AutoStart = 1    #: Started as a result of `autoStart`
        NormalStart = 2  #: Started as a result of `start`

    NoReason, AutoStart, NormalStart = StartReason

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__startreason = UpdateManager.NoReason
        self.__state = State.Ready
        # An error (if any) encountered during the update check
        self.__exception = None  # type: Optional[Exception]

    def _settings(self):
        """
        Return a QSettings instance initialized at the appropriate group

        Returns
        -------
        settings : QSettings
        """
        s = QSettings()
        s.beginGroup("application.update")
        return s

    def cachePath(self):
        """
        Returns
        -------
        path : str
            A path to the local cache store.
        """
        datadir = config.cache_dir()
        return os.path.join(datadir, "UpdateManager.cache.pck")

    def autoStart(self):
        """
        Start the update check in the background if scheduled.

        Return False if no checks are scheduled; i.e if updates are disabled
        or not enough time has passed since the last check.
        """
        settings = self._settings()
        updateperiod = settings.value("check-period", defaultValue=1, type=int)
        enabled = settings.value("enabled", defaultValue=False, type=bool)
        if not enabled:
            log.debug("UpdateManager.autoStart: Not enabled for this env")
            return False
        if updateperiod < 0:
            log.debug("UpdateManager.autoStart: Disabled by user preference")
            return False

        lastcheck = settings.value("last-check-date",
                                   defaultValue="", type=str)
        lastcheck = _date_parse(lastcheck)
        if lastcheck is None:
            lastcheck = date.fromtimestamp(0)

        lastdelta = date.today() - lastcheck
        log.debug("UpdateManager.autoStart: Time from last check: %s (%s)",
                  lastdelta, lastcheck)
        if lastdelta >= timedelta(days=updateperiod):
            self.__doStart(UpdateManager.AutoStart)
            return True
        elif self.__state == State.Running:
            # If already running update the start reason and return True
            self.__startreason |= UpdateManager.AutoStart
            return True
        elif self.__shouldEmitSpontaneousNotification():
            # If any non-skipped updates available and the no-remind
            # period is up
            QTimer.singleShot(0, self.__emitNotification)
            return True
        else:
            return False

    def start(self):
        """
        Start the update check in a background thread.

        Return `True` if a check was scheduled, or `False` if one is
        already in progress. In both cases the `UpdateManager.reason`
        property will be updated with the `UpdateManager.NormalStart`
        flag.
        """
        return self.__doStart(UpdateManager.NormalStart)

    def __doStart(self, reason):
        self.__startreason |= reason
        if self.__state != State.Running:
            self.__state = State.Running
            log.debug("UpdateManager.start: running check (%s)", reason)
            f = qconcurrent.submit(self.fetchUpdateInfo)
            w = qconcurrent.FutureWatcher(parent=self)
            w.done.connect(self.__ondone)
            w.setFuture(f)
            self.started.emit()
            return True
        else:
            return False

    def fetchUpdateInfo(self):
        """
        Fetch and return available updates for the application.

        Note
        ----
        This method will be called in a separate thread.

        Returns
        -------
        updates : List[config.Installable]
        """
        return config.fetch_latest()

    @Slot(object)
    def __ondone(self, future):
        # future with the latest application version has completed
        assert future.done()
        self.__exception = None
        log.debug("UpdateManager: Update check completed (%s)", future)
        if future.cancelled():
            self.__state = State.Ready
            self.finished.emit()
            return
        elif future.exception():
            self.__state = State.Error
            self.__exception = future.exception()
            log.exception("UpdateManager: An error occurred while "
                          "retrieving update information",
                          exc_info=self.__exception)
            self.finished.emit()
            return
        else:
            self.__state = State.Ready

        items = future.result()
        try:
            os.makedirs(os.path.dirname(self.cachePath()), exist_ok=True)
            with open(self.cachePath(), "wb") as f:
                pickle.dump((0, 0, 0), f)  # write data version first
                pickle.dump(items, f)
        except OSError:
            pass
        except pickle.PicklingError:
            log.exception("UpdateManager: Error writing to cache",
                          exc_info=True)

        s = self._settings()
        s.setValue("last-check-date", _date_to_string(date.today()))
        s.sync()
        self.finished.emit()
        self.__emitNotification()

    def __shouldEmitSpontaneousNotification(self):
        # Determine if a spontaneous notification request should be emitted
        s = self._settings()
        lastnotified = s.value(
            "last-notification-date", defaultValue="", type=str)
        lastnotified = _date_parse(lastnotified)
        if lastnotified is None:
            lastnotified = date.fromtimestamp(0)
        remindperiod = s.value("remind-period", defaultValue=3, type=int)
        noremind = date.today() - lastnotified < timedelta(days=remindperiod)
        updatable = self.updateCandidates()
        notskipped = [item for item in updatable if not item.skipped]
        return bool(notskipped and not noremind)

    def __emitNotification(self):
        if self.__state == State.Running:
            # start was called again; will emit with updated information
            # when it completes.
            return
        spontaneous = not self.__startreason & UpdateManager.NormalStart
        if not spontaneous or self.__shouldEmitSpontaneousNotification():
            self.updateNotificationRequested.emit()
            s = self._settings()
            s.setValue("last-notification-date",
                       _date_to_string(date.today()))

    def reason(self):
        """
        Reason for running the check.

        Returns
        -------
        reason : UpdateManager.Reason
            The reason the update check was run (by `autoStart` or `start`)
        """
        return self.__startreason

    def allItems(self):
        """
        Return a list of *all* current (cached) `UpdateCandidate` items.

        Note that this includes items that are up-to date.

        Returns
        -------
        item: List[UpdateCandidates]
        """
        fname = self.cachePath()
        try:
            with open(fname, "rb") as f:
                v1, v2, v3 = pickle.load(f)
                if (v1, v2, v3) == (0, 0, 0):
                    items = pickle.load(f)
                else:
                    items = []
        except FileNotFoundError:
            items = []
        except OSError as err:
            log.warning("UpdateManager: Error reading '%s': '%r'", fname, err)
            items = []
        except pickle.UnpicklingError as err:
            log.warning("UpdateManager: Unpickling error: %r", err)
            items = []
        except Exception:  # pylint: disable=broad-except
            log.exception("UpdateManager: Exception raised", exc_info=True)
            items = []

        ws = pkg_resources.WorkingSet()
        updateitems = []
        for item in items:
            try:
                req = pkg_resources.Requirement.parse(item.name)
            except ValueError:
                # item.name is not a valid distribution name
                continue
            try:
                dist = ws.find(req)
            except (pkg_resources.VersionConflict,
                    pkg_resources.DistributionNotFound):
                # There is no distribution installed (possibly uninstalled
                # since last check)
                continue

            updateitems.append(
                UpdateCandidate(
                    name=item.name,
                    latest_version=item.version,
                    installed_version=dist.version,
                    category=item.category,
                    display_name=item.display_name,
                    skipped=item.version in self.skippedVersions(item.name),
                    meta={
                        "release-notes-url": item.release_notes_url,
                        "download-url": item.download_url
                    }
                )
            )
        return updateitems

    def updateCandidates(self):
        """
        Return all update candidates.

        This is a subset of `allItems()` whose `latest_version` is larger
        then their `installed_version`.

        Returns
        -------
        items : List[UpdateCandidate]
        """
        return [item for item in self.allItems()
                if (parse_version(item.installed_version) <
                    parse_version(item.latest_version))]

    def exception(self):
        """
        Return the exception that occurred in fetchUpdateInfo (if any)

        Returns
        -------
        exception : Optional[Exception]
        """
        return self.__exception

    def latestVersion(self, name):
        """
        Return the latest (cached) available application version.

        Returns
        -------
        version : str
        """
        items = self.allItems()
        for item in items:
            if item.name == name:
                return item.latest_version
        return "0.0.0"

    def noteSkipped(self, item):
        """
        Mark the version as skipped, and never again emit spontaneous update
        notifications for it.

        Parameters
        ----------
        item : UpdateCandidate
        """
        if item.latest_version in self.skippedVersions(item.name):
            return

        fname = os.path.join(config.data_dir(), "update-skip.txt")
        try:
            with open(fname, "x") as f:
                f.write("# Versions which the user explicitly skipped "
                        "(one per line)\n")
        except FileExistsError:
            pass

        with open(fname, "a") as f:
            f.write("{0.name}=={0.latest_version}\n".format(item))

    def skippedVersions(self, name):
        """
        Return a list of all versions which are noted as skipped.

        Returns
        -------
        versions : List[str]
        """
        fname = os.path.join(config.data_dir(), "update-skip.txt")
        try:
            with open(fname, "r") as f:
                return [ver.split("==", 1)[-1].strip()
                        for ver in f if ver.startswith(name + "==")]
        except FileNotFoundError:
            return []
