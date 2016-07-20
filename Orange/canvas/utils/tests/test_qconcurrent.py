import unittest
import unittest.mock
import threading

from concurrent import futures

from AnyQt.QtCore import QCoreApplication

try:
    from AnyQt.QtTest import QSignalSpy
    HAS_QSIGNALSPY = True
except ImportError:
    HAS_QSIGNALSPY = False

from .. import qconcurrent


class TestFutureWatcher(unittest.TestCase):
    def setUp(self):
        self.app = QCoreApplication.instance()
        if self.app is None:
            self.app = QCoreApplication([])

    def tearDown(self):
        del self.app

    def test_watcher(self):
        executor = futures.ThreadPoolExecutor(max_workers=1)

        def flush():
            # 'flush' the executor (ensure all scheduled tasks are completed)
            # executor must have exactly 1 worker
            f = executor.submit(lambda: None)
            f.result()

        event = threading.Event()
        event.clear()
        # Block the worker thread to ensure subsequent future can be cancelled
        # before it starts
        executor.submit(event.wait)

        cancelled = unittest.mock.MagicMock()
        watcher = qconcurrent.FutureWatcher()
        watcher.cancelled.connect(cancelled)

        f = executor.submit(lambda: None)
        watcher.setFuture(f)

        self.assertTrue(f.cancel())

        # Unblock the work thread
        event.set()

        with self.assertRaises(futures.CancelledError):
            f.result()

        # ensure the waiters/watchers were notified (by
        # set_running_or_notify_cancelled in the worker thread)
        flush()

        QCoreApplication.sendPostedEvents(watcher, 0)

        cancelled.assert_called_with(f)

        finished = unittest.mock.MagicMock()
        result = unittest.mock.MagicMock()
        exception = unittest.mock.MagicMock()

        watcher = qconcurrent.FutureWatcher()

        watcher.finished.connect(finished)
        watcher.resultReady.connect(result)
        watcher.exceptionReady.connect(exception)

        watcher.setFuture(executor.submit(lambda: 42))
        if HAS_QSIGNALSPY:
            self.assertTrue(QSignalSpy(watcher.done).wait(1000))
        else:
            watcher.done.connect(self.app.quit)
            self.app.exec_()

        self.assertEqual(watcher.result(), 42)

        finished.assert_called_with(watcher.future())
        result.assert_called_with(42)
        exception.assert_not_called()

        finished = unittest.mock.MagicMock()
        result = unittest.mock.MagicMock()
        exception = unittest.mock.MagicMock()

        watcher = qconcurrent.FutureWatcher()

        watcher.finished.connect(finished)
        watcher.resultReady.connect(result)
        watcher.exceptionReady.connect(exception)

        watcher.setFuture(executor.submit(lambda: 1 / 0))
        if HAS_QSIGNALSPY:
            self.assertTrue(QSignalSpy(watcher.done).wait(1000))
        else:
            watcher.done.connect(self.app.quit)
            self.app.exec_()

        with self.assertRaises(ZeroDivisionError):
            watcher.result()

        self.assertIsInstance(watcher.exception(), ZeroDivisionError)
        finished.assert_called_with(watcher.future())
        result.assert_not_called()
        exception.assert_called_with(watcher.exception())

        executor.shutdown()


class TestAnonymousSubmit(unittest.TestCase):
    def test_qconcurrent_submit(self):
        f = qconcurrent.submit(lambda i: (i ** i ** i ** i), 2)
        f.result()
