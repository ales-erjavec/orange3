"""
Tests for WidgetsScheme
"""
import copy
import io

from PyQt4.QtCore import QObject, QEventLoop, QTimer

from Orange.widgets.data.owfile import OWFile
from Orange.widgets.data.owdiscretize import OWDiscretize

from ...gui import test
from ...registry.tests import small_testing_registry

from ..widgetsscheme import WidgetsScheme, WidgetManager


class QSignalSpy(QObject):
    """
    QSignalSpy-like object (missing in PyQt4, but present in PyQt5)
    """
    def __init__(self, boundsig):
        self.__boundsig = boundsig
        self.__boundsig.connect(lambda *args: self.__record(*args))
        self.__recorded = []
        self.__loop = QEventLoop()
        self.__timer = QTimer(singleShot=True)
        self.__timer.timeout.connect(self.__loop.quit)

    def __record(self, *args):
        self.__recorded.append(list(args))
        if self.__loop.isRunning():
            self.__loop.quit()

    def wait(self, timeout):
        count = len(self)
        self.__timer.stop()
        self.__timer.setInterval(timeout)
        self.__timer.start()
        self.__loop.exec_()
        self.__timer.stop()
        return len(self) != count

    def __getitem__(self, index):
        return self.__recorded[index]

    def __len__(self):
        return len(self.__recorded)


class TestWidgetsScheme(test.QAppTestCase):
    def test_scheme(self):
        reg = small_testing_registry()
        base = "Orange.widgets"
        file_desc = reg.widget(base + ".data.owfile.OWFile")
        discretize_desc = reg.widget(base + ".data.owdiscretize.OWDiscretize")

        scheme = WidgetsScheme()

        manager = scheme.widget_manager
        manager.set_creation_policy(WidgetManager.Immediate)

        # all current node/widget pairs in WidgetManager
        current = []  # type: List[Tuple[SchemeNode, OWWidget]]

        def record_add(node, widget):
            current.append((node, widget))

        def record_remove(node, widget):
            current.remove((node, widget))

        manager.widget_for_node_added.connect(record_add)
        manager.widget_for_node_removed.connect(record_remove)

        node = scheme.new_node(
            file_desc, properties={
                "recent_paths": OWFile.recent_paths.default[:1]})

        self.assertEqual(len(current), 1)
        node_1, widget = current[0]

        self.assertIs(node_1, node)
        self.assertIsInstance(widget, OWFile)
        self.assertIs(scheme.widget_for_node(node), widget)

        manager.set_creation_policy(WidgetManager.OnDemand)
        node = scheme.new_node(discretize_desc)

        self.assertEqual(len(current), 1)

        # Force widget creation
        widget = manager.widget_for_node(node)
        self.assertIsInstance(widget, OWDiscretize)

        self.assertEqual(len(current), 2)

        node_1, widget = current[1]

        self.assertIs(node_1, node)
        self.assertIsInstance(widget, OWDiscretize)
        self.assertIs(scheme.widget_for_node(node), widget)

        scheme.clear()

        self.assertEqual(current, [])

        node_file = scheme.new_node(
            file_desc, properties={
                "recent_paths": OWFile.recent_paths.default[:1]})

        node_disc = scheme.new_node(discretize_desc)
        self.assertEqual(current, [])

        # saving should not touch/force create widgets if not initialized
        # (due to sync_node_properties call in save_to)
        stream = io.BytesIO()
        scheme.save_to(stream)
        self.assertEqual(current, [])

        manager.set_creation_policy(WidgetManager.Delayed)
        spy = QSignalSpy(manager.widget_for_node_added)
        didemit = spy.wait(2000)
        self.assertTrue(didemit)
        self.assertEqual(len(current), 1)
        didemit = spy.wait(2000)
        self.assertTrue(didemit)
        self.assertEqual(len(spy), 2)
        self.assertEqual(len(current), 2)

        [(n1, w1), (n2, w2)] = spy

        self.assertIs(n1, node_file)
        self.assertIs(n2, node_disc)
