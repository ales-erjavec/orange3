"""
Orange Canvas Configuration

"""

import os
import sys
import logging
import pickle as pickle
import itertools
import sysconfig

from urllib.request import urlopen
from collections import namedtuple

import pkg_resources

from AnyQt.QtGui import (
    QPainter, QFont, QFontMetrics, QColor, QPixmap, QIcon
)
from AnyQt.QtCore import Qt, QCoreApplication, QPoint, QRect, QSettings

from .utils.settings import Settings, config_slot

log = logging.getLogger(__name__)


def init():
    """
    Initialize the QCoreApplication.organizationDomain, applicationName,
    applicationVersion and the default settings format. Will only run once.

    .. note:: This should not be run before QApplication has been initialized.
              Otherwise it can break Qt's plugin search paths.

    """
    if not QCoreApplication.instance():
        raise RuntimeError(
            "config.init must not be called before QApplication has "
            "been initialized")

    dist = pkg_resources.get_distribution("Orange3")
    version = dist.version
    # Use only major.minor
    version = ".".join(version.split(".", 2)[:2])

    QCoreApplication.setOrganizationDomain("biolab.si")
    QCoreApplication.setApplicationName("Orange Canvas")
    QCoreApplication.setApplicationVersion(version)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    # Set $PREFIX/etc/xdg/ as the system config dir.
    path = os.path.join(sysconfig.get_path("data",), "etc", "xdg")
    QSettings.setPath(QSettings.IniFormat, QSettings.SystemScope, path)

    # Make it a null op.
    global init
    init = lambda: None

rc = {}


spec = \
    [("startup/show-splash-screen", bool, True,
      "Show splash screen at startup"),

     ("startup/show-welcome-screen", bool, True,
      "Show Welcome screen at startup"),

     ("stylesheet", str, "orange",
      "QSS stylesheet to use"),

     ("schemeinfo/show-at-new-scheme", bool, True,
      "Show Workflow Properties when creating a new Workflow"),

     ("mainwindow/scheme-margins-enabled", bool, False,
      "Show margins around the workflow view"),

     ("mainwindow/show-scheme-shadow", bool, True,
      "Show shadow around the workflow view"),

     ("mainwindow/toolbox-dock-exclusive", bool, True,
      "Should the toolbox show only one expanded category at the time"),

     ("mainwindow/toolbox-dock-floatable", bool, False,
      "Is the canvas toolbox floatable (detachable from the main window)"),

     ("mainwindow/toolbox-dock-movable", bool, True,
      "Is the canvas toolbox movable (between left and right edge)"),

     ("mainwindow/toolbox-dock-use-popover-menu", bool, True,
      "Use a popover menu to select a widget when clicking on a category "
      "button"),

     ("mainwindow/number-of-recent-schemes", int, 15,
      "Number of recent workflows to keep in history"),

     ("schemeedit/show-channel-names", bool, True,
      "Show channel names"),

     ("schemeedit/show-link-state", bool, True,
      "Show link state hints."),

     ("schemeedit/enable-node-animations", bool, True,
      "Enable node animations."),

     ("schemeedit/freeze-on-load", bool, False,
      "Freeze signal propagation when loading a workflow."),

     ("quickmenu/trigger-on-double-click", bool, True,
      "Show quick menu on double click."),

     ("quickmenu/trigger-on-right-click", bool, True,
      "Show quick menu on right click."),

     ("quickmenu/trigger-on-space-key", bool, True,
      "Show quick menu on space key press."),

     ("quickmenu/trigger-on-any-key", bool, False,
      "Show quick menu on double click."),

     ("logging/level", int, 1, "Logging level"),

     ("logging/show-on-error", bool, True, "Show log window on error"),

     ("logging/dockable", bool, True, "Allow log window to be docked"),

     ("help/open-in-external-browser", bool, False,
      "Open help in an external browser"),

     ("error-reporting/machine-id", str, '',
     "Report custom name instead of machine ID"),

     ("application.update/enabled", bool, False,
      "Enable periodic 'Check for updates' functionality"),

     ("application.update/check-period", int, 1,
      "Check for updates every #N days (0 means check at every start, "
      "negative value means never)"),

     ("application.update/remind-period", int, 3,
      "Remind the user every #N days after an update is discovered"),
     ]

spec = [config_slot(*t) for t in spec]


def settings():
    init()
    store = QSettings()
    settings = Settings(defaults=spec, store=store)
    return settings


def data_dir():
    """Return the application data directory. If the directory path
    does not yet exists then create it.

    """
    from Orange.misc import environ
    path = os.path.join(environ.data_dir(), "canvas")

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def cache_dir():
    """Return the application cache directory. If the directory path
    does not yet exists then create it.

    """
    from Orange.misc import environ
    path = os.path.join(environ.cache_dir(), "canvas")

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def log_dir():
    """
    Return the application log directory.
    """
    init()
    if sys.platform == "darwin":
        name = str(QCoreApplication.applicationName())
        logdir = os.path.join(os.path.expanduser("~/Library/Logs"), name)
    else:
        logdir = data_dir()

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir


def widget_settings_dir():
    """
    Return the widget settings directory.
    """
    from Orange.misc import environ
    return environ.widget_settings_dir()


def open_config():
    global rc
    app_dir = data_dir()
    filename = os.path.join(app_dir, "canvas-rc.pck")
    if os.path.exists(filename):
        with open(os.path.join(app_dir, "canvas-rc.pck"), "rb") as f:
            rc.update(pickle.load(f))


def save_config():
    app_dir = data_dir()
    with open(os.path.join(app_dir, "canvas-rc.pck"), "wb") as f:
        pickle.dump(rc, f)


def recent_schemes():
    """Return a list of recently accessed schemes.
    """
    app_dir = data_dir()
    recent_filename = os.path.join(app_dir, "recent.pck")
    recent = []
    if os.path.isdir(app_dir) and os.path.isfile(recent_filename):
        with open(recent_filename, "rb") as f:
            recent = pickle.load(f)

    # Filter out files not found on the file system
    recent = [(title, path) for title, path in recent \
              if os.path.exists(path)]
    return recent


def save_recent_scheme_list(scheme_list):
    """Save the list of recently accessed schemes
    """
    app_dir = data_dir()
    recent_filename = os.path.join(app_dir, "recent.pck")

    if os.path.isdir(app_dir):
        with open(recent_filename, "wb") as f:
            pickle.dump(scheme_list, f)


WIDGETS_ENTRY = "orange.widgets"


# This could also be achieved by declaring the entry point in
# Orange's setup.py, but that would not guaranty this entry point
# is the first in a list.

def default_entry_point():
    """
    Return a default orange.widgets entry point for loading
    default Orange Widgets.

    """
    dist = pkg_resources.get_distribution("Orange3")
    ep = pkg_resources.EntryPoint("Orange Widgets", "Orange.widgets",
                                  dist=dist)
    return ep


def widgets_entry_points():
    """
    Return an `EntryPoint` iterator for all 'orange.widget' entry
    points plus the default Orange Widgets.

    """
    ep_iter = pkg_resources.iter_entry_points(WIDGETS_ENTRY)
    chain = [[default_entry_point()],
             ep_iter
             ]
    return itertools.chain(*chain)

#: Parameters for searching add-on packages in PyPi using xmlrpc api.
ADDON_KEYWORD = 'orange3 add-on'
ADDON_PYPI_SEARCH_SPEC = {"keywords": ADDON_KEYWORD}
#: Entry points by which add-ons register with pkg_resources.
ADDON_ENTRY = "orange3.addon"


def splash_screen():
    """
    """
    pm = QPixmap(
        pkg_resources.resource_filename(
            __name__, "icons/orange-splash-screen.png")
    )

    version = QCoreApplication.applicationVersion()
    size = 21 if len(version) < 5 else 16
    font = QFont("Helvetica")
    font.setPixelSize(size)
    font.setBold(True)
    font.setItalic(True)
    font.setLetterSpacing(QFont.AbsoluteSpacing, 2)
    metrics = QFontMetrics(font)
    br = metrics.boundingRect(version).adjusted(-5, 0, 5, 0)
    br.moveCenter(QPoint(436, 224))

    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    p.setFont(font)
    p.setPen(QColor("#231F20"))
    p.drawText(br, Qt.AlignCenter, version)
    p.end()
    return pm, QRect(88, 193, 200, 20)


def application_icon():
    """
    Return the main application icon.
    """
    path = pkg_resources.resource_filename(
        __name__, "icons/orange-canvas.svg"
    )
    return QIcon(path)


def application_name():
    """
    Return the application name
    """
    return "Orange Canvas"


def application_distribution():
    """
    Return the `pkg_resources.Distribution` of the python project defining
    this application

    Returns
    -------
    dist: pkg_resources.Distribution
    """
    return pkg_resources.get_distribution("Orange3")


def application_version():
    """
    Return the application version string
    """
    dist = pkg_resources.get_distribution("Orange3")
    return dist.version


def latest_application_version_url():
    return "http://orange.biolab.si/version"


def download_url():
    """
    Return the download landing page url
    """
    return "http://orange.biolab.si/download"


# An example application configuration
exampleconfig = """
[application.update]

# Enable disable periodic update checking (default is disabled)
#enabled = false

# Check period expressed in days
#check-period = 1

# Remind the user every #N days after an update is discovered
#remind-period = 3
"""


def updateconf():
    """
    Return periodic update check configuration for the current environment

    Updates must be specifically enabled on an per environment basis.

    Returns
    -------
    conf : dict
        A dictionary with at least one key `"enabled"` and corresponding
        boolean value. Other entries include `"check-period"` and
        `"remind-period"` if specified in the configuration file (see
        `etc/xdg/biolab.si/Orange Canvas.ini` example in the source root).

    """
    init()
    s = QSettings()
    s.beginGroup("application.update")
    conf = {
        "enabled": s.value("enabled", defaultValue=False, type=bool),
        "check-period": s.value("check-period", defaultValue=7, type=int),
        "remind-period": s.value("remind-period", defaultValue=3, type=int),

    }
    conf.update({
        "version-check-url": latest_application_version_url(),
        "download-url": download_url()
    })
    return conf


_Installable = namedtuple(
    "Installable",
    ["name",
     "version",
     "category",
     "display_name",
     "download_url",
     "release_notes_url"]
)


class Installable(_Installable):
    """
    Attributes
    ----------
    name : str
        A python distribution/project name of the installable item
    version : str
        Version string of the available item.
    category : str
        "core" or "add-on"
    display_name : Optional[str]
        A human friendlier name for use in GUI (if applicable; otherwise
        `name` is used)
    download_url : Optional[str]
    release_notes_url : Optional[str]
    """
    def __new__(cls, name, version, category, display_name=None,
                download_url=None, release_notes_url=None):
        return super().__new__(
            cls, name, version, category, display_name, download_url,
            release_notes_url)


def fetch_latest():
    """
    Fetch the latest installable application and/or add-ons meta info

    Returns
    -------
    info : List[Installable]
        A list of namespace like objects with (at least) the following fields:

        * `name` : str
        * `version` : str
        * `category`: str
        * `display_name` : Optional[str]
        * `download_url` : Optional[str]
        * `release_notes_url` : Optional[str]

    `category` is one of "core" or "add-on"

    """
    cfg = updateconf()
    if not cfg["enabled"]:
        return []

    with urlopen(cfg["version-check-url"], timeout=10) as s:
        version = s.read().decode("ascii")

    latest = [
        Installable(
            name=application_distribution().project_name,
            version=version,
            category="core",
            display_name=application_name(),
            download_url=cfg["download-url"],
            release_notes_url=
                "https://github.com/biolab/orange3/blob/master/CHANGELOG.md")
    ]
    return latest
