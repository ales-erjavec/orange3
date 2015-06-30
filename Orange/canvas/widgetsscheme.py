"""
Widgets Scheme
==============

A Scheme for Orange Widgets Scheme (.ows).

This is a subclass of the general :class:`Scheme`. It is responsible for
the construction and management of OWBaseWidget instances corresponding
to the scheme nodes, as well as delegating the signal propagation to a
companion :class:`WidgetsSignalManager` class.

.. autoclass:: WidgetsScheme
   :bases:

.. autoclass:: WidgetsSignalManager
  :bases:

"""
import sys
import logging
import warnings
from types import SimpleNamespace as namespace

import sip

from PyQt4.QtCore import Qt, QCoreApplication, QTimer, QEvent

from OrangeCanvas.scheme.signalmanager import \
    SignalManager, compress_signals, can_enable_dynamic
from OrangeCanvas.scheme import Scheme, SchemeNode, WorkflowEvent
from OrangeCanvas.scheme.widgetsscheme import WidgetManager
from OrangeCanvas.scheme.node import UserMessage
from OrangeCanvas.utils import name_lookup
from OrangeCanvas.resources import icon_loader

log = logging.getLogger(__name__)


class WidgetsScheme(Scheme):
    """
    A Scheme containing Orange Widgets managed with a `WidgetsSignalManager`
    instance.

    Extends the base `Scheme` class to handle the lifetime
    (creation/deletion, etc.) of `OWWidget` instances corresponding to
    the nodes in the scheme. It also delegates the interwidget signal
    propagation to an instance of `WidgetsSignalManager`.

    """
    def __init__(self, parent=None, title=None, description=None):
        Scheme.__init__(self, parent, title, description)
        self.__started = False
        self.signal_manager = WidgetsSignalManager(self)
        self.widget_manager = WidgetManager(self)

    def sync_node_properties(self):
        """
        Sync the widget settings/properties with the SchemeNode.properties.
        Return True if there were any changes in the properties (i.e. if the
        new node.properties differ from the old value) and False otherwise.

        """
        changed = False
        if not self.__started:
            return False

        for node in self.nodes:
            widget = self.widget_manager.widget_for_node(node)
            settings = widget.settingsHandler.pack_data(widget)
            if settings != node.properties:
                node.properties = settings
                changed = True
        log.debug("Scheme node properties sync (changed: %s)", changed)
        return changed

    def start(self):
        if not self.__started:
            self.signal_manager.set_scheme(self)
            self.signal_manager.start()
            self.widget_manager.set_scheme(self)
            self.__started = True

    def stop(self):
        if self.__started:
            self.signal_manager.set_scheme(None)
            self.widget_manager.close()
            self.widget_manager.set_scheme(None)
            self.__started = False

    def customEvent(self, event):
        if event.type() == WorkflowEvent.Activate:
            self.start()
            event.accept()
        elif event.type() == WorkflowEvent.Deactivate:
            self.stop()
            event.accept()
        else:
            super(WidgetsScheme, self).customEvent(event)


class WidgetManager(WidgetManager):
    """
    OWWidget instance manager class.

    This class handles the lifetime of OWWidget instances in a
    :class:`WidgetsScheme`. It closely cooperates with WidgetSignalManager
    to abstract away the widget.OWWidget's `peculiarities`.

    """
    #: Widget processing state flags:
    #:   * InputUpdate - WidgetsSignalManager is updating/setting the
    #:     widget's inputs.
    #:   * BlockingUpdate - widget has entered a blocking state (explicit
    #:     blocking by a call to OWWidget.setBlocking)
    #:   * ProcessingUpdate - widget has entered processing state (implicit
    #:     blocking due to a progressBarInit/Finishes calls).
    NoUpdate, InputUpdate, BlockingUpdate, ProcessingUpdate = 0, 1, 2, 4

    def __init__(self, parent=None):
        super(WidgetManager, self).__init__(parent)

        # Widgets that were 'removed' from the scheme but were at
        # the time in an input update loop and could not be deleted
        # immediately
        self.__delay_delete = set()

        # processing state flags for all nodes (including the ones
        # in __delay_delete).
        self.__widget_processing_state = {}

        # Tracks the widget in the update loop by the SignalManager
        self.__updating_widget = None

    def signal_manager(self):
        """
        Return the signal manager in use on the :func:`scheme`.
        """
        scheme = self.scheme()
        if scheme is not None:
            return scheme.findChild(SignalManager)
        else:
            return None

    def set_scheme(self, scheme):
        super(WidgetManager, self).set_scheme(scheme)

        signalmanager = self.signal_manager()
        if signalmanager is not None:
            signalmanager.processingStarted[SchemeNode].connect(
                self.__on_processing_started
            )
            signalmanager.processingFinished[SchemeNode].connect(
                self.__on_processing_finished)

    def close(self):
        signalmanager = self.signal_manager()
        if signalmanager is not None:
            signalmanager.processingStarted[SchemeNode].disconnect(
                self.__on_processing_started
            )
            signalmanager.processingFinished[SchemeNode].disconnect(
                self.__on_processing_finished)
        super(WidgetManager, self).close()

    def remove_widget(self, widget):
        node = self.node_for_widget(widget)

        node.title_changed.disconnect(widget.setCaption)
        widget.progressBarValueChanged.disconnect(node.set_progress)

        widget.processingStateChanged.disconnect(
            self.__on_processing_state_changed)
        widget.widgetStateChanged.disconnect(
            self.__on_widget_state_changed)
        widget.blockingStateChanged.disconnect(
            self.__on_blocking_state_changed)

        widget.close()

        # Save settings to user global settings.
        widget.saveSettings()

        # Notify the widget it will be deleted.
        widget.onDeleteWidget()

        if self.__widget_processing_state[widget] != 0:
            # If the widget is in an update loop and/or blocking we
            # delay the scheduled deletion until the widget is done.
            self.__delay_delete.add(widget)
        else:
            widget.deleteLater()
            del self.__widget_processing_state[widget]

    def create_widget_instance(self, node):
        """
        Create a OWWidget instance for the node.
        """
        desc = node.description
        klass = name_lookup(desc.qualified_name)

        log.info("Creating %r instance.", klass)
        fakesm = namespace()
        fakesm.captured = []
        fakesm.send = \
            lambda widget, name, value, id: \
                fakesm.captured.append((widget, name, value, id))

        widget = klass.__new__(
            klass,
            None,
            signal_manager=fakesm,
            stored_settings=node.properties
        )
        widget.__init__()
        widget.signalManager = self.signal_manager()
        self.__widget_processing_state[widget] = 0

        if fakesm.captured:
            warnings.warn(
                "{} sends output from __init__!".format(klass.__name__))

        widget.setCaption(node.title)
        widget.setWindowIcon(
            icon_loader.from_description(desc).get(desc.icon))
        widget.setVisible(node.properties.get("visible", False))

        node.title_changed.connect(widget.setCaption)

        # Widget's info/warning/error messages.
        widget.widgetStateChanged.connect(self.__on_widget_state_changed)

        # Widget's statusTip
        node.set_status_message(widget.statusMessage())
        widget.statusMessageChanged.connect(node.set_status_message)

        # Widget's progress bar value state.
        widget.progressBarValueChanged.connect(node.set_progress)

        # Widget processing state (progressBarInit/Finished)
        # and the blocking state.
        widget.processingStateChanged.connect(
            self.__on_processing_state_changed
        )
        widget.blockingStateChanged.connect(self.__on_blocking_state_changed)

        if widget.isBlocking():
            # A widget can already enter blocking state in __init__
            self.__widget_processing_state[widget] |= self.BlockingUpdate

        if widget.processingState != 0:
            # It can also start processing (initialization of resources, ...)
            self.__widget_processing_state[widget] |= self.ProcessingUpdate
            node.set_processing_state(1)
            node.set_progress(widget.progressBarValue)

        if self.signal_manager() is not None and fakesm.captured:
            def delayedsend():
                sm = self.signal_manager()
                if sm is not None:
                    for args in fakesm.captured:
                        sm.send(*args)
            QTimer.singleShot(0, delayedsend)

        return widget

    def node_processing_state(self, node):
        """
        Return the processing state flags for the node.

        Same as `manager.widget_processing_state(manger.widget_for_node(node))`
        """
        widget = self.widget_for_node(node)
        return self.__widget_processing_state[widget]

    def widget_processing_state(self, widget):
        """
        Return the processing state flags for the widget.
        """
        return self.__widget_processing_state[widget]

    def __initialize_widget_state(self, node, widget):
        """
        Initialize the tracked info/warning/error message state.
        """
        for message_type, state in widget.widgetState.items():
            for message_id, message_value in state.items():
                message = user_message_from_state(
                    widget, message_type, message_id, message_value)

                node.set_state_message(message)

    def __on_widget_state_changed(self, message_type, message_id,
                                  message_value):
        """
        The OWBaseWidget info/warning/error state has changed.

        message_type is one of "Info", "Warning" or "Error" string depending
        of which method (information, warning, error) was called. message_id
        is the first int argument if supplied, and message_value the message
        text.

        """
        widget = self.sender()
        try:
            node = self.node_for_widget(widget)
        except KeyError:
            pass
        else:
            message = user_message_from_state(
                widget, str(message_type), message_id, message_value)

            node.set_state_message(message)

    def __on_processing_state_changed(self, state):
        """
        A widget processing state has changed (progressBarInit/Finished)
        """
        widget = self.sender()
        try:
            node = self.node_for_widget(widget)
        except KeyError:
            return

        if state:
            self.__widget_processing_state[widget] |= self.ProcessingUpdate
        else:
            self.__widget_processing_state[widget] &= ~self.ProcessingUpdate
        self.__update_node_processing_state(node)

    def __on_processing_started(self, node):
        """
        Signal manager entered the input update loop for the node.
        """
        widget = self.widget_for_node(node)
        # Remember the widget instance. The node and the node->widget mapping
        # can be removed between this and __on_processing_finished.
        self.__updating_widget = widget
        self.__widget_processing_state[widget] |= self.InputUpdate
        self.__update_node_processing_state(node)

    def __on_processing_finished(self, node):
        """
        Signal manager exited the input update loop for the node.
        """
        widget = self.__updating_widget
        self.__widget_processing_state[widget] &= ~self.InputUpdate

        if widget in self.__node_for_widget:
            self.__update_node_processing_state(node)
        elif widget in self.__delay_delete:
            self.__try_delete(widget)
        else:
            raise ValueError("%r is not managed" % widget)

        self.__updating_widget = None

    def __on_blocking_state_changed(self, state):
        """
        OWWidget blocking state has changed.
        """
        if not state:
            # schedule an update pass.
            self.signal_manager()._update()

        widget = self.sender()
        if state:
            self.__widget_processing_state[widget] |= self.BlockingUpdate
        else:
            self.__widget_processing_state[widget] &= ~self.BlockingUpdate

        if widget in self.__node_for_widget:
            node = self.node_for_widget(widget)
            self.__update_node_processing_state(node)

        elif widget in self.__delay_delete:
            self.__try_delete(widget)

    def __update_node_processing_state(self, node):
        """
        Update the `node.processing_state` to reflect the widget state.
        """
        state = self.node_processing_state(node)
        node.set_processing_state(1 if state else 0)

    def __try_delete(self, widget):
        if self.__widget_processing_state[widget] == 0:
            self.__delay_delete.remove(widget)
            widget.deleteLater()
            del self.__widget_processing_state[widget]


def user_message_from_state(widget, message_type, message_id, message_value):
    message_type = str(message_type)
    if message_type == "Info":
        contents = widget.widgetStateToHtml(True, False, False)
        level = UserMessage.Info
    elif message_type == "Warning":
        contents = widget.widgetStateToHtml(False, True, False)
        level = UserMessage.Warning
    elif message_type == "Error":
        contents = widget.widgetStateToHtml(False, False, True)
        level = UserMessage.Error
    else:
        raise ValueError("Invalid message_type: %r" % message_type)

    if not contents:
        contents = None

    message = UserMessage(contents, severity=level,
                          message_id=message_type,
                          data={"content-type": "text/html"})
    return message


class WidgetsSignalManager(SignalManager):
    """
    A signal manager for a WidgetsScheme.
    """
    def __init__(self, scheme):
        SignalManager.__init__(self, scheme)

        scheme.installEventFilter(self)

        self.__scheme_deleted = False

        scheme.destroyed.connect(self.__on_scheme_destroyed)

    def send(self, widget, channelname, value, signal_id):
        """
        send method compatible with OWWidget.
        """
        scheme = self.scheme()
        try:
            node = scheme.widget_manager.node_for_widget(widget)
        except KeyError:
            # The Node/Widget was already removed from the scheme.
            log.debug("Node for %r is not in the scheme.", widget)
            return

        try:
            channel = node.output_channel(channelname)
        except ValueError:
            log.error("%r is not valid signal name for %r",
                      channelname, node.description.name)
            return

        # Expand the signal_id with the unique widget id and the
        # channel name. This is needed for OWBaseWidget's input
        # handlers (Multiple flag).
        signal_id = (widget.widget_id, channelname, signal_id)

        SignalManager.send(self, node, channel, value, signal_id)

    def is_blocking(self, node):
        return self.scheme().widget_manager.node_processing_state(node) != 0

    def send_to_node(self, node, signals):
        """
        Implementation of `SignalManager.send_to_node`.

        Deliver input signals to an OWBaseWidget instance.

        """
        scheme = self.scheme()
        if scheme is not None:
            widget = scheme.widget_manager.widget_for_node(node)
            self.process_signals_for_widget(node, widget, signals)

    def compress_signals(self, signals):
        """
        Reimplemented from :func:`SignalManager.compress_signals`.
        """
        return compress_signals(signals)

    def process_signals_for_widget(self, node, widget, signals):
        """
        Process new signals for the OWBaseWidget.
        """
        # This replaces the old OWBaseWidget.processSignals method

        if sip.isdeleted(widget):
            log.critical("Widget %r was deleted. Cannot process signals",
                         widget)
            return

        app = QCoreApplication.instance()

        for signal in signals:
            link = signal.link
            value = signal.value

            # Check and update the dynamic link state
            if link.is_dynamic():
                link.dynamic_enabled = can_enable_dynamic(link, value)
                if not link.dynamic_enabled:
                    # Send None instead
                    value = None

            handler = link.sink_channel.handler
            if handler.startswith("self."):
                handler = handler.split(".", 1)[1]

            handler = getattr(widget, handler)

            if link.sink_channel.single:
                args = (value,)
            else:
                args = (value, signal.id)

            log.debug("Process signals: calling %s.%s (from %s with id:%s)",
                      type(widget).__name__, handler.__name__, link, signal.id)

            app.setOverrideCursor(Qt.WaitCursor)
            try:
                handler(*args)
            except Exception:
                sys.excepthook(*sys.exc_info())
                log.exception("Error calling '%s' of '%s'",
                              handler.__name__, node.title)
            finally:
                app.restoreOverrideCursor()

        app.setOverrideCursor(Qt.WaitCursor)
        try:
            widget.handleNewSignals()
        except Exception:
            sys.excepthook(*sys.exc_info())
            log.exception("Error calling 'handleNewSignals()' of '%s'",
                          node.title)
        finally:
            app.restoreOverrideCursor()

    def event(self, event):
        if event.type() == QEvent.UpdateRequest:
            if self.__scheme_deleted:
                log.debug("Scheme has been/is being deleted. No more "
                          "signals will be delivered to any nodes.")
                event.setAccepted(True)
                return True
        # Retain a reference to the scheme until the 'process_queued' finishes
        # in SignalManager.event.
        scheme = self.scheme()
        return SignalManager.event(self, event)

    def eventFilter(self, receiver, event):
        if event.type() == QEvent.DeferredDelete and receiver is self.scheme():
            try:
                state = self.runtime_state()
            except AttributeError:
                # If the scheme (which is a parent of this object) is
                # already being deleted the SignalManager can also be in
                # the process of destruction (noticeable by its __dict__
                # being empty). There is nothing really to do in this
                # case.
                state = None

            if state == SignalManager.Processing:
                log.info("Deferring a 'DeferredDelete' event for the Scheme "
                         "instance until SignalManager exits the current "
                         "update loop.")
                event.setAccepted(False)
                self.processingFinished.connect(self.scheme().deleteLater)
                self.__scheme_deleted = True
                return True

        return SignalManager.eventFilter(self, receiver, event)

    def __on_scheme_destroyed(self, obj):
        self.__scheme_deleted = True
