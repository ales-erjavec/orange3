"""
Utilities for cooperating between Qt and `concurrent.futures`s
"""
import weakref
import concurrent.futures

from AnyQt.QtCore import (
    QObject,  QThreadPool, QRunnable, QEvent, QCoreApplication
)
from AnyQt.QtCore import pyqtSignal as Signal


class TaskRunnable(QRunnable):
    """
    A QRunnable to fulfil a `Future` in a QThreadPool managed thread.

    Parameters
    ----------
    future : concurrent.futures.Future
        Future whose contents will be set with the result of executing
        `func(*args, **kwargs)`.
    func : Callable
        Function to invoke in a thread
    args : tuple
        Positional arguments for `func`
    kwargs : dict
        Keyword arguments for `func`

    Example
    -------
    >>> f = concurrent.futures.Future()
    >>> task = TaskRunnable(f, time.sleep, (1,), {})
    >>> QThreadPool.globalInstance().start(task)
    >>> f.result()
    """
    def __init__(self, future, func, args, kwargs):
        super().__init__()
        self.future = future
        self.task = (func, args, kwargs)

    def run(self):
        """
        Reimplemented from `QRunnable.run`
        """
        if not self.future.set_running_or_notify_cancel():
            # Was cancelled
            return
        func, args, kwargs = self.task
        try:
            result = func(*args, **kwargs)
        except BaseException as ex:
            self.future.set_exception(ex)
        else:
            self.future.set_result(result)


def submit(func, *args, **kwargs):
    """
    Schedule a callable `func` to run in a global `QThreadPool`.

    Parameters
    ----------
    func : callable
    args : tuple
        Positional arguments for `func`
    kwargs : dict
        Keyword arguments for `func`

    Returns
    -------
    future : concurrent.futures.Future
        Future with the (eventual) result of `func(*args, **kwargs)`
    """
    f = concurrent.futures.Future()
    task = TaskRunnable(f, func, args, kwargs)
    QThreadPool.globalInstance().start(task)
    return f


class FutureWatcher(QObject):
    """
    An `QObject` watching the state changes of a `concurrent.futures.Future`.

    Note
    ----
    The state change notification signals (`done`, `finished`, ...)
    are always emitted when the control flow reaches the event loop
    (even if the future is already completed when set).

    Note
    ----
    An event loop must be running in the watcher's `thread`, otherwise the
    notifier signals will not be emitted.

    Parameters
    ----------
    parent : QObject
        Parent object.
    future : concurrent.futures.Future
        The future instance to watch.

    Example
    -------
    >>> assert QCoreApplication.instance() is not None
    >>> watcher = FutureWatcher()
    >>> watcher.done.connect(lambda f: print(f.result()))
    >>> f = submit(lambda i, j: i ** j, 10, 3)
    >>> watcher.setFuture(f)
    >>> QTest.qWait(100)
    1000
    """
    #: Emitted when the future is done (cancelled or finished)
    done = Signal(concurrent.futures.Future)

    #: Emitted when the future is finished (i.e. returned a result
    #: or raised an exception)
    finished = Signal(concurrent.futures.Future)

    #: Emitted when the future was cancelled
    cancelled = Signal(concurrent.futures.Future)

    #: Emitted with the future's result when successfully finished.
    resultReady = Signal(object)

    #: Emitted with the future's exception when finished with an exception.
    exceptionReady = Signal(BaseException)

    # A private event type used to notify the watcher of a Future's completion
    __FutureDone = QEvent.registerEventType()

    def __init__(self, parent=None, future=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__future = None  # type: concurrent.futures.Future
        if future is not None:
            self.setFuture(future)

    def setFuture(self, future):
        """
        Set the future to watch.

        Raise a `RuntimeError` if a future is already set.

        Parameters
        ----------
        future : concurrent.futures.Future
        """
        if self.__future is not None:
            raise RuntimeError("Future is already set")

        self.__future = future
        selfweakref = weakref.ref(self)

        def on_done(f):
            assert f is future
            selfref = selfweakref()

            if selfref is None:
                return

            try:
                QCoreApplication.postEvent(
                    selfref, QEvent(FutureWatcher.__FutureDone))
            except RuntimeError:
                # Ignore RuntimeErrors (when C++ side of QObject is deleted)
                # (? Use QObject.destroyed and remove the done callback ?)
                pass

        future.add_done_callback(on_done)

    def future(self):
        """
        Return the future.
        """
        return self.__future

    def result(self):
        """
        Return the future's result.

        Note
        ----
        This method is non-blocking. If the future has not yet completed
        it will raise an RuntimeError.
        """
        try:
            return self.__future.result(timeout=0)
        except TimeoutError:
            raise RuntimeError("Result is not ready.")

    def exception(self):
        """
        Return the future's exception.

        Note
        ----
        This method is non-blocking. If the future has not yet completed
        it will raise an RuntimeError.
        """
        try:
            return self.__future.exception(timeout=0)
        except TimeoutError:
            raise RuntimeError("Exception is not ready.")

    def __emitSignals(self):
        assert self.__future is not None
        assert self.__future.done()
        if self.__future.cancelled():
            self.cancelled.emit(self.__future)
            self.done.emit(self.__future)
        elif self.__future.done():
            self.finished.emit(self.__future)
            self.done.emit(self.__future)
            if self.__future.exception():
                self.exceptionReady.emit(self.__future.exception())
            else:
                self.resultReady.emit(self.__future.result())
        else:
            assert False

    def customEvent(self, event):
        # Reimplemented.
        if event.type() == FutureWatcher.__FutureDone:
            self.__emitSignals()
        super().customEvent(event)
