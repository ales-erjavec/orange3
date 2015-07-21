import itertools
import pkg_resources

from orangecanvas import config

from . import discovery
from . import widgetsscheme

WIDGETS_ENTRY = "orange.widgets"
ADDONS_ENTRY = "orange3.addon"


class orangeconfig(config.default):
    ApplicationName = "Orange Canvas"
    dist = pkg_resources.get_distribution("Orange")
    ApplicationVersion = "{}.{}".format(*dist.version.split(".", 2)[:2])

    @staticmethod
    def widgets_entry_points():
        """
        Return an `EntryPoint` iterator for all 'orange.widget' entry
        points plus the default Orange Widgets.

        """
        dist = pkg_resources.get_distribution("Orange")
        default_ep = pkg_resources.EntryPoint(
            "Orange Widgets", "Orange.widgets", dist=dist)
        return itertools.chain(
            (default_ep,), pkg_resources.iter_entry_points(WIDGETS_ENTRY))

    @staticmethod
    def addon_entry_points():
        return pkg_resources.iter_entry_points(ADDONS_ENTRY)

    @staticmethod
    def addon_pypi_search_spec():
        return {"keywords": ["orange3 add-on"]}

    @staticmethod
    def tutorials_entry_points():
        default_ep = pkg_resources.EntryPoint(
            "Orange", "Orange.canvas.application.tutorials",
            dist=pkg_resources.get_distribution("Orange"))

        return itertools.chain(
            (default_ep,), pkg_resources.iter_entry_points(""))

    widget_discovery = discovery.WidgetDiscovery
    workflow_constructor = widgetsscheme.WidgetsScheme
