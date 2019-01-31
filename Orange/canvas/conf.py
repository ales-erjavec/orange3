import itertools
import pkg_resources

from orangecanvas import config

from . import discovery
from . import widgetsscheme

WIDGETS_ENTRY = "orange.widgets"
ADDONS_ENTRY = "orange3.addon"


class orangeconfig(config.default):
    ApplicationName = "Orange Canvas"
    dist = pkg_resources.get_distribution("Orange3")
    try:
        ApplicationVersion = dist.version
    except ValueError:
        ApplicationVersion = ""

    @staticmethod
    def widgets_entry_points():
        """
        Return an `EntryPoint` iterator for all 'orange.widget' entry
        points plus the default Orange Widgets.

        """
        dist = pkg_resources.get_distribution("Orange3")
        default_ep = pkg_resources.EntryPoint(
            "Orange Widgets", "Orange.widgets", dist=dist)
        return itertools.chain(
            (default_ep,), pkg_resources.iter_entry_points(WIDGETS_ENTRY))

    @staticmethod
    def addon_entry_points():
        return pkg_resources.iter_entry_points(ADDONS_ENTRY)

    @staticmethod
    def addon_pypi_search_spec():
        return {"keywords": ["orange3", "add-on"]}

    @staticmethod
    def tutorials_entry_points():
        default_ep = pkg_resources.EntryPoint(
            "Orange3", "Orange.canvas.workflows",
            dist=pkg_resources.get_distribution("Orange3"))

        return itertools.chain(
            (default_ep,),
            pkg_resources.iter_entry_points("orange.widgets.tutorials"))

    widget_discovery = discovery.WidgetDiscovery
    workflow_constructor = widgetsscheme.WidgetsScheme

    APPLICATION_URLS = {
        #: Submit a bug report action in the Help menu
        "Bug Report": "https://github.com/biolab/orange3/issues",
        #: A url quick tour/getting started url
        "Quick Start": "https://orange.biolab.si/getting-started/",
        #: The 'full' documentation, should be something like current /docs/
        #: but specific for 'Visual Programing' only
        "Documentation": "https://orange.biolab.si/toolbox/",
        #: YouTube tutorials
        "Screencasts":
            "https://www.youtube.com/watch"
            "?v=HXjnDIgGDuI&list=PLmNPvQr9Tf-ZSDLwOzxpvY-HrE0yv-8Fy&index=1",
        #: Used for 'Submit Feedback' action in the help menu
        "Feedback": "https://orange.biolab.si/survey/long.html",
    }
