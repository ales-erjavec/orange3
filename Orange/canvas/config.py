"""
Orange Canvas Configuration

"""

import os
import sys
import logging
import pickle as pickle
import itertools
import sysconfig
import configparser

from urllib.request import urlopen

import pkg_resources

from PyQt4.QtGui import (
    QPainter, QFont, QFontMetrics, QColor, QPixmap, QIcon
)

from PyQt4.QtCore import Qt, QCoreApplication, QPoint, QRect

from .utils.settings import Settings, config_slot

# Import QSettings from qtcompat module (compatibility with PyQt < 4.8.3
from .utils.qtcompat import QSettings

log = logging.getLogger(__name__)


def init():
    """
    Initialize the QCoreApplication.organizationDomain, applicationName,
    applicationVersion and the default settings format. Will only run once.

    .. note:: This should not be run before QApplication has been initialized.
              Otherwise it can break Qt's plugin search paths.

    """
    dist = pkg_resources.get_distribution("Orange3")
    version = dist.version
    # Use only major.minor
    version = ".".join(version.split(".", 2)[:2])

    QCoreApplication.setOrganizationDomain("biolab.si")
    QCoreApplication.setApplicationName("Orange Canvas")
    QCoreApplication.setApplicationVersion(version)
    QSettings.setDefaultFormat(QSettings.IniFormat)

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

     ("output/redirect-stderr", bool, True,
      "Redirect and display standard error output"),

     ("output/redirect-stdout", bool, True,
      "Redirect and display standard output"),

     ("output/stay-on-top", bool, True, ""),

     ("output/show-on-error", bool, True, "Show output window on error"),

     ("output/dockable", bool, True, "Allow output window to be docked"),

     ("help/stay-on-top", bool, True, ""),

     ("help/dockable", bool, True, "Allow help window to be docked"),

     ("help/open-in-external-browser", bool, False,
      "Open help in an external browser"),

     ("application/update/check-period", int, 1,
      "Check for updates every #N days (0 means check at every start;"
      "-1 means disabled"),

     ("application/update/nag-period", int, 1,
      "Nag the user every #N days when an update is available"),
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


def application_version():
    """
    Return the application version string
    """
    dist = pkg_resources.get_distribution("Orange3")
    return dist.version


def fetch_latest_application_version():
    with urlopen("http://orange.biolab.si/version", timeout=5) as r:
        return r.read().decode("ascii")


def download_url():
    """
    Return the download landing page url
    """
    return "http://orange.biolab.si/download"


from .utils import pypiquery

# A list of known Orange3 Add-ons hardcoded here due to pypi's flakiness
# https://github.com/pypa/pypi-legacy/issues/397
# https://bitbucket.org/pypa/pypi/issues/326
OFFICIAL_ADDONS = [
    "Orange-Bioinformatics",
    "Orange3-DataFusion",
    "Orange3-Prototypes",
    "Orange3-Text",
    "Orange3-Network",
    "Orange3-Associate",
]


# TODO: Try the 'https://pypi.io/pypi' as a xmlrpc entry point

def _query_orange_addons():
    # Search Orange3 add ons on PyPi
    # return a list of project names and their latest version
    client = pypiquery.default_pypi_client(timeout=10)
    addons = pypiquery.pypi_search(ADDON_PYPI_SEARCH_SPEC, client)

    addons = [addon.name for addon in addons] + OFFICIAL_ADDONS
    addons = list(set(addons))
    return pypiquery.pypi_json_query_project_meta(addons)


def query_addons():
    return _query_orange_addons()


def pkgconfg_iter_installed():
    # TODO: get_dist_meta and other utils from manager into utils
    from .help.manager import get_dist_meta
    rval = []
    ws = pkg_resources.WorkingSet(sys.path)
    for key, dist in ws.by_key.items():
        meta = get_dist_meta(dist)
        keywords = meta.get("Keywords", "")
        if "orange3 add-on" in keywords:
            rval.append(dist)
    for ep in ws.iter_entry_points(ADDON_ENTRY):
        if ep.dist not in rval:
            rval.append(ep.dist)
    return rval


updateconfig = """
[update-components]
items =
    Orange Canvas

[Orange Canvas]
name = Orange
type = ApplicationInstaller
category = "core/application"
"""

# An example application configuration
updateconfig = """
[application.update]

# Enable disable update/checking (default is disabled)
#enabled = false

# Update type; can be 'Manual' (the user must download and run an
# installer) or 'Auto' (the application can apply/install an update by
# itself). Further type specific configurations are listed bellow
type = Manual
# type = Auto


[application.update.manual]

# An url returning the latest application version string. The version must
# be be PEP-440 compliant
versionurl = http://orange.biolab.si/version

# Application download landing page
downloadurl = http://orange.biolab.si/download


[application.update.auto]

# Base url of a simple repository api (specified by PEP 503). If not supplied
# then the default https://pypi.python.org/simple/ will be used
#repourl = https://pypi.io/simple/


# The project name used to query the pypi index
project = Orange3

"""


def updateconf():
    fname = os.path.join(
        sysconfig.get_path("data"), "etc", "Orange3", "update.conf")
    cfg = configparser.ConfigParser()
    cfg.read([fname])
    items = cfg.get("update-components", "items", fallback="")
    items = [s.strip() for s in items.splitlines()]
    # ignore missing sections
    items = [s for s in items if s in cfg]
    config = {}

    for item in items:
        name = cfg.get(item, "name", fallback=item)
        cat = cfg.get(item, "category", fallback=None)
        type = cfg.get(item, "type", fallback=None)
        if cat is not None and type is not None:
            config[item] = {"name": name, "category": cat, "type": type}
    return config


def updateconf1():
    fname = os.path.join(
        sysconfig.get_path("data"), "etc", "orange3", "canvas.conf")

    cfg = configparser.ConfigParser()
    cfg.read([fname])
    config = {}

    enabled = cfg.getboolean("application.update", "enabled", fallback=False)
    config["enabled"] = enabled
    if not enabled:
        return {"enabled": False}

    updatetype = cfg.get("application.update", "type", fallback=None)
    if updatetype is None:
        return {"enabled": False}
    elif updatetype not in {"Auto", "Manual"}:
        return {"enabled": False}

    config["updatetype"] = updatetype

    if updatetype == "Manual":
        section = "application.update.manual"
        versionurl = cfg.get(section, "versionurl", fallback=None)
        downloadurl = cfg.get(section, "downloadurl", fallback=None)
        if not versionurl and downloadurl:
            return {"enabled": False}
        config["manual"] = {"versionurl": versionurl,
                            "downloadurl": downloadurl}
        return config
    elif updatetype == "Auto":
        section = "application.update.auto"
        repourl = cfg.get(section, "repourl",
                           fallback=pypiquery.PYPI_INDEX_URL)
        projectname = cfg.get(section, "project", fallback=None)
        if not repourl and projectname:
            return {"enabled": False}
        config["auto"] = {"repourl": repourl, "projectname": projectname}
    return config


def fetch_updates():
    from .application.addons import Installable  ## is not installable and not from adddons!!!!!
    cfg = updateconf()
    item = cfg.get(application_name(), None)
    if item is None:
        return []
    elif item["type"] == "ApplicationInstaller":
        latest = fetch_latest_application_version()
        item = Installable(
            name=application_name(),
            version=latest,
            summary="",
            description="",
            package_url=download_url(),
            release_urls=[]
        )
        return [item]
    else:
        return []


def fetch_updates1():
    from types import SimpleNamespace as namespace
    cfg = updateconf1()
    if not cfg["enabled"]:
        return []
    elif cfg["updatetype"] == "Manual":
        cfg = cfg["manual"]
        with urlopen(cfg["versionurl"]) as s:
            version = s.read().decode("ascii")
        return [namespace(name=application_name(),
                          updatetype="manual",
                          version=version,
                          download_url=cfg["downloadurl"],
                          package_url=cfg["downloadurl"])]

    elif cfg["updatetype"] == "Auto":
        cfg = cfg["auto"]
        indexurl = cfg["repourl"]
        projectname = cfg["projectname"]
        releases = pypiquery.simple_index_query(projectname, indexurl)
        if not releases:
            return []
        else:
            latest = releases[-1]
            return [namespace(name=application_name(),
                              updatetype="pip",
                              project_name=projectname,
                              version=latest.version,
                              index_url=indexurl,
                              package_url=indexurl + projectname,
                              download_url=indexurl + projectname,
                              release_urls=latest.urls)]
