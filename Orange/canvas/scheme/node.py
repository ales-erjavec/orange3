"""
===========
Scheme Node
===========

"""
import enum

from AnyQt.QtCore import QObject
from AnyQt.QtCore import pyqtSignal as Signal, pyqtProperty as Property


class UserMessage(object):
    """
    A user message that should be displayed in a scheme view.

    Paramaters
    ----------
    contents : str
        Message text.
    severity : int
        Message severity.
    message_id : A hashable object
        Message id.
    data : dict
        A dictionary with optional extra data.

    """
    #: Severity flags
    Info, Warning, Error = 1, 2, 3

    def __init__(self, contents, severity=Info, message_id=None, data={}):
        self.contents = contents
        self.severity = severity
        self.message_id = message_id
        self.data = dict(data)


class SchemeNode(QObject):
    """
    A node in a :class:`.Scheme`.

    Parameters
    ----------
    description : :class:`WidgetDescription`
        Node description instance.
    title : str, optional
        Node title string (if None `description.name` is used).
    position : tuple
        (x, y) tuple of floats for node position in a visual display.
    properties : dict
        Additional extra instance properties (settings, widget geometry, ...)
    parent : :class:`QObject`
        Parent object.

    """

    class RuntimeState(enum.IntEnum):
        """
        Node runtime state flags.
        """
        NoState = 0
        #: Set after any runtime resources are initialized
        Created = 1

        Initialized = 2
        #: The node's input's are in the process of being set.
        #: This flag is set/cleared by the workflow execution engine
        UpdatingInputs = 4
        #: The node is in processing state.
        #: TODO: Processing or Active, Working,
        Processing = 8
        #: Waiting for user input before the node can proceed
        WaitingForUserInput = 16

        #: This flag is set/cleared from the UI. It means that the
        #: settings/parameters of the node were changed, but the changes
        #: are still un-committed.
        #: Note: Corresponds to the setModified|isModified
        HasUncommitedChanges = 32  # Does this imply WaitingForUserInput?
        #: The node parameters are marked as changed and have not yet been
        #: persisted to disk
        Modified = 64

        #: When a node receives any new inputs and enters processing state:
        #: its outputs are monitored for changes and:
        #: * if at the end of the update it sends changed outputs it enters
        #:   DidUpdate state
        DidUpdate = 4
        #: * if at the end of an update it sends only some of it's outputs
        #:   it enters the DidPartialUpdate
        DidPartialUpdate = 8

    # NOTE: Should the runtime state be more like:
    #       Stopped, Waiting, Running, Paused, Cancelled.
    #       RunState vs RuntimeState?

    class RunState(enum.IntEnum):
        NotRunning = 0
        Running = 1
        Paused = 2
        Stopped = 3
        Cancelled = 4
        Error = 5

    class State(enum.IntEnum):
        NoState = 0
        Initialized = 1
        #: A Node is activated when When
        Activated = 2

    class WorkflowState(enum.IntEnum):
        #: The node is scheduled to receive updated inputs.
        InputPending = 1
        #: ...


    # TODO: Convenience signals
    #: Emitted when node transitions from Processing to ~Processing
    finished = Signal()  # maybe done = Signal()
    #: Emitted when the node transitions to WaitingForUserInput state
    requestUserInput = Signal()

    def __init__(self, description, title=None, position=None,
                 properties=None, parent=None):
        QObject.__init__(self, parent)
        self.description = description

        if title is None:
            title = description.name

        self.__title = title
        self.__position = position or (0, 0)
        self.__progress = -1
        self.__processing_state = 0
        self.__tool_tip = ""
        self.__status_message = ""
        self.__state_messages = {}
        self.__state = SchemeNode.RuntimeState.NoState
        self.properties = properties or {}

    def input_channels(self):
        """
        Return a list of input channels (:class:`InputSignal`) for the node.
        """
        return list(self.description.inputs)

    def output_channels(self):
        """
        Return a list of output channels (:class:`OutputSignal`) for the node.
        """
        return list(self.description.outputs)

    def input_channel(self, name):
        """
        Return the input channel matching `name`. Raise a `ValueError`
        if not found.

        """
        for channel in self.input_channels():
            if channel.name == name:
                return channel
        raise ValueError("%r is not a valid input channel name for %r." % \
                         (name, self.description.name))

    def output_channel(self, name):
        """
        Return the output channel matching `name`. Raise an `ValueError`
        if not found.

        """
        for channel in self.output_channels():
            if channel.name == name:
                return channel
        raise ValueError("%r is not a valid output channel name for %r." % \
                         (name, self.description.name))

    #: The title of the node has changed
    title_changed = Signal(str)

    def set_title(self, title):
        """
        Set the node title.
        """
        if self.__title != title:
            self.__title = str(title)
            self.title_changed.emit(self.__title)

    def title(self):
        """
        The node title.
        """
        return self.__title

    title = Property(str, fset=set_title, fget=title)

    #: Position of the node in the scheme has changed
    position_changed = Signal(tuple)

    def set_position(self, pos):
        """
        Set the position (``(x, y)`` tuple) of the node.
        """
        if self.__position != pos:
            self.__position = pos
            self.position_changed.emit(pos)

    def position(self):
        """
        ``(x, y)`` tuple containing the position of the node in the scheme.
        """
        return self.__position

    position = Property(tuple, fset=set_position, fget=position)

    #: Node's progress value has changed.
    progress_changed = Signal(float)

    def set_progress(self, value):
        """
        Set the progress value.
        """
        if self.__progress != value:
            self.__progress = value
            self.progress_changed.emit(value)

    def progress(self):
        """
        The current progress value. -1 if progress is not set.
        """
        return self.__progress

    progress = Property(float, fset=set_progress, fget=progress)

    #: Node's processing state has changed.
    processing_state_changed = Signal(int)

    def set_processing_state(self, state):
        """
        Set the node processing state.
        """
        if self.__processing_state != state:
            self.__processing_state = state
            self.processing_state_changed.emit(state)

    def processing_state(self):
        """
        The node processing state, 0 for not processing, 1 the node is busy.
        """
        return self.__processing_state

    processing_state = Property(int, fset=set_processing_state,
                                fget=processing_state)

    def set_tool_tip(self, tool_tip):
        if self.__tool_tip != tool_tip:
            self.__tool_tip = tool_tip

    def tool_tip(self):
        return self.__tool_tip

    tool_tip = Property(str, fset=set_tool_tip,
                        fget=tool_tip)

    #: The node's status tip has changes
    status_message_changed = Signal(str)

    def set_status_message(self, text):
        if self.__status_message != text:
            self.__status_message = text
            self.status_message_changed.emit(text)

    def status_message(self):
        return self.__status_message

    #: The node's state message has changed
    state_message_changed = Signal(UserMessage)

    def set_state_message(self, message):
        """
        Set a message to be displayed by a scheme view for this node.
        """
        if message.message_id in self.__state_messages and \
                not message.contents:
            del self.__state_messages[message.message_id]

        self.__state_messages[message.message_id] = message
        self.state_message_changed.emit(message)

    def state_messages(self):
        """
        Return a list of all state messages.
        """
        return self.__state_messages.values()

    def set_state_flag(self, flag, state=True):
        state = bool(state)
        if bool(self.__state & flag) != state:
            if state:
                self.__state |= flag
            else:
                self.__state &= ~flag

            assert bool(self.__state & flag) == state
            self.runtime_state_changed.emit(self.__state)

    def runtime_state(self):
        return self.__state

    def test_runtime_state(self, state):
        return bool(self.__state & state)

    runtime_state_changed = Signal(int)

    # def set_blocking(self, state):
    #     self.set_state_flag(SchemeNode.RuntimeState.)

    def __str__(self):
        return "SchemeNode(description_id=%s, title=%r, ...)" % \
                (str(self.description.id), self.title)

    def __repr__(self):
        return str(self)
