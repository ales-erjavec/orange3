import sys
import argparse
import logging

from AnyQt.QtCore import QEvent
from AnyQt.QtGui import QPainter
from AnyQt.QtWidgets import QTabWidget, QApplication

from orangecanvas import utils
from orangecanvas.application.application import CanvasApplication
from orangecanvas.registry import WidgetRegistry
from orangewidget.workflow import widgetsscheme
from orangecanvas.canvas import scene, view


def main(argv=None):
    app = CanvasApplication(list(argv) if argv else [])
    argv = app.arguments()

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", "-l", metavar="LEVEL", type=int,
                        default=logging.CRITICAL, dest="log_level")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not show GUI")
    parser.add_argument("--config", default="orangewidget.workflow.config.Config",
                        type=str)
    parser.add_argument("file")
    args = parser.parse_args(argv[1:])

    log_level = args.log_level
    filename = args.file
    logging.basicConfig(level=log_level)
    cfg_class = utils.name_lookup(args.config)
    cfg = cfg_class()
    cfg.init()
    reg = WidgetRegistry()
    widget_discovery = cfg.widget_discovery(reg)
    widget_discovery.run(cfg.widgets_entry_points())
    model = cfg.workflow_constructor()
    sigprop = model.findChild(widgetsscheme.WidgetsSignalManager)
    sigprop.pause()  # Pause signal propagation during load

    with open(filename, "rb") as f:
        model.load_from(f, registry=reg)

    # Workflow graph display
    sc = scene.CanvasScene()
    sc.set_scheme(model)
    scview = view.CanvasView(sc)
    scview.setRenderHint(QPainter.Antialiasing)
    sc.setParent(scview)

    # Put all the widgets in a tab widget
    mainwidget = QTabWidget()
    for w in map(model.widget_for_node, model.nodes):
        mainwidget.addTab(w, w.captionTitle)

    mainwidget.addTab(scview, "Workflow view")
    if not args.no_display:
        mainwidget.show()
        mainwidget.raise_()

    sigprop.resume()  # Resume inter-widget signal propagation
    rval = app.exec_()
    model.clear()
    # Notify the workflow model to 'close'.
    QApplication.sendEvent(model, QEvent(QEvent.Close))
    app.processEvents()
    del mainwidget
    return rval


if __name__ == "__main__":
    sys.exit(main(sys.argv))
