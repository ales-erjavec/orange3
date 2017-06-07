import sys
import copy
import concurrent.futures
import logging
from collections import namedtuple

try:  # pylint: disable=unused-import
    from typing import List
except ImportError:
    pass

import numpy as np

from AnyQt.QtWidgets import (
    QGroupBox, QRadioButton, QPushButton, QHBoxLayout,
    QVBoxLayout, QStackedWidget, QComboBox,
    QButtonGroup, QStyledItemDelegate, QListView, QDoubleSpinBox
)
from AnyQt.QtCore import Qt
from AnyQt.QtCore import pyqtSlot as Slot

import Orange.data
from Orange.preprocess import impute
from Orange.base import Learner
from Orange.widgets import gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils import concurrent as qconcurrent
from Orange.widgets.utils.concurrent import methodinvoke
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, Msg
from Orange.classification import SimpleTreeLearner
from Orange.widgets.evaluate.owtestlearners import Try


class DisplayFormatDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        method = index.data(Qt.UserRole)
        var = index.model()[index.row()]
        if method:
            option.text = method.format_variable(var)

            if not method.supports_variable(var):
                option.palette.setColor(option.palette.Text, Qt.darkRed)

            if isinstance(getattr(method, 'method', method), impute.DoNotImpute):
                option.palette.setColor(option.palette.Text, Qt.darkGray)


class AsDefault(impute.BaseImputeMethod):
    name = "Default (above)"
    short_name = ""
    format = "{var.name}"
    columns_only = True

    method = impute.DoNotImpute()

    def __getattr__(self, item):
        return getattr(self.method, item)

    def supports_variable(self, variable):
        return self.method.supports_variable(variable)

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)


class SparseNotSupported(ValueError):
    pass


class VariableNotSupported(ValueError):
    pass


RowMask = namedtuple("RowMask", ["mask"])


class Task:
    futures = []    # type: List[Future]
    watcher = ...   # type: FutureSetWatcher
    cancelled = False

    def __init__(self, futures, watcher):
        self.futures = futures
        self.watcher = watcher

    def cancel(self):
        self.cancelled = True
        for f in self.futures:
            f.cancel()


class OWImpute(OWWidget):
    name = "Impute"
    description = "Impute missing values in the data table."
    icon = "icons/Impute.svg"
    priority = 2130

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Learner", Learner, "set_learner")]
    outputs = [("Data", Orange.data.Table)]

    class Error(OWWidget.Error):
        imputation_failed = Msg("Imputation failed for '{}'")
        model_based_imputer_sparse = Msg("Model based imputer does not work for sparse data")

    DEFAULT_LEARNER = SimpleTreeLearner()  # Does not release the gil (is not thread safe)
    METHODS = [AsDefault(), impute.DoNotImpute(), impute.Average(),
               impute.AsValue(), impute.Model(DEFAULT_LEARNER), impute.Random(),
               impute.DropInstances(), impute.Default()]
    DEFAULT, DO_NOT_IMPUTE, MODEL_BASED_IMPUTER, AS_INPUT = 0, 1, 4, 7

    settingsHandler = settings.DomainContextHandler()

    _default_method_index = settings.Setting(DO_NOT_IMPUTE)
    variable_methods = settings.ContextSetting({})
    autocommit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        # copy METHODS (some are modified by the widget)
        self.methods = copy.deepcopy(OWImpute.METHODS)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.controlArea.layout().addLayout(main_layout)

        box = QGroupBox(title=self.tr("Default Method"), flat=False)
        box_layout = QVBoxLayout(box)
        main_layout.addWidget(box)

        button_group = QButtonGroup()
        button_group.buttonClicked[int].connect(self.set_default_method)
        for i, method in enumerate(self.methods):
            if not method.columns_only:
                button = QRadioButton(method.name)
                button.setChecked(i == self.default_method_index)
                button_group.addButton(button, i)
                box_layout.addWidget(button)

        self.default_button_group = button_group

        box = QGroupBox(title=self.tr("Individual Attribute Settings"),
                        flat=False)
        main_layout.addWidget(box)

        horizontal_layout = QHBoxLayout(box)
        main_layout.addWidget(box)

        self.varview = QListView(
            selectionMode=QListView.ExtendedSelection
        )
        self.varview.setItemDelegate(DisplayFormatDelegate())
        self.varmodel = itemmodels.VariableListModel()
        self.varview.setModel(self.varmodel)
        self.varview.selectionModel().selectionChanged.connect(
            self._on_var_selection_changed
        )
        self.selection = self.varview.selectionModel()

        horizontal_layout.addWidget(self.varview)

        method_layout = QVBoxLayout()
        horizontal_layout.addLayout(method_layout)

        button_group = QButtonGroup()
        for i, method in enumerate(self.methods):
            button = QRadioButton(text=method.name)
            button_group.addButton(button, i)
            method_layout.addWidget(button)

        self.value_combo = QComboBox(
            minimumContentsLength=8,
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLength,
            activated=self._on_value_selected
            )
        self.value_double = QDoubleSpinBox(
            editingFinished=self._on_value_selected,
            minimum=-1000., maximum=1000., singleStep=.1, decimals=3,
            )
        self.value_stack = value_stack = QStackedWidget()
        value_stack.addWidget(self.value_combo)
        value_stack.addWidget(self.value_double)
        method_layout.addWidget(value_stack)

        button_group.buttonClicked[int].connect(
            self.set_method_for_current_selection
        )

        method_layout.addStretch(2)

        reset_button = QPushButton(
                "Restore All to Default", checked=False, checkable=False,
                clicked=self.reset_variable_methods, default=False,
                autoDefault=False)
        method_layout.addWidget(reset_button)

        self.variable_button_group = button_group

        box = gui.auto_commit(
            self.controlArea, self, "autocommit", "Apply",
            orientation=Qt.Horizontal, checkbox_label="Apply automatically")
        box.layout().insertSpacing(0, 80)
        box.layout().insertWidget(0, self.report_button)

        self.data = None
        self.learner = None
        self.modified = False
        self.default_method = self.methods[self.default_method_index]
        self.executor = qconcurrent.ThreadExecutor(self)
        self.__task = None

    @property
    def default_method_index(self):
        return self._default_method_index

    @default_method_index.setter
    def default_method_index(self, index):
        if self._default_method_index != index:
            self._default_method_index = index
            self.default_button_group.button(index).setChecked(True)
            self.default_method = self.methods[self.default_method_index]
            self.methods[self.DEFAULT].method = self.default_method

            # update variable view
            for index in map(self.varmodel.index, range(len(self.varmodel))):
                method = self.variable_methods.get(
                    index.row(), self.methods[self.DEFAULT])
                self.varmodel.setData(index, method, Qt.UserRole)
            self._invalidate()

    def set_default_method(self, index):
        """Set the current selected default imputation method.
        """
        self.default_method_index = index

    @check_sql_input
    def set_data(self, data):
        self.closeContext()
        self.varmodel[:] = []
        self.variable_methods = {}
        self.modified = False
        self.data = data

        if data is not None:
            self.varmodel[:] = data.domain.variables
            self.openContext(data.domain)

        self.update_varview()
        self.unconditional_commit()

    def set_learner(self, learner):
        self.learner = learner or self.DEFAULT_LEARNER
        # TODO: KILL IT WITH FIRE! ITS THE ONLY WAY TO BE SURE.
        #       (revert some of 30ffb381c3e)
        imputer = self.methods[self.MODEL_BASED_IMPUTER]
        imputer.learner = self.learner

        button = self.default_button_group.button(self.MODEL_BASED_IMPUTER)
        button.setText(imputer.name)

        variable_button = self.variable_button_group.button(self.MODEL_BASED_IMPUTER)
        variable_button.setText(imputer.name)

        if learner is not None:
            self.default_method_index = self.MODEL_BASED_IMPUTER

        self.update_varview()

        if self.default_method_index == self.MODEL_BASED_IMPUTER:
            self.unconditional_commit()

    def get_method_for_column(self, column_index):
        """Returns the imputation method for column by its index.
        """
        if not isinstance(column_index, int):
            column_index = column_index.row()

        return self.variable_methods.get(column_index,
                                         self.methods[self.DEFAULT])

    def _invalidate(self):
        self.modified = True
        if self.__task is not None:
            self.cancel()
        self.commit()

    def commit(self):
        self.cancel()
        self.warning()
        self.Error.imputation_failed.clear()
        self.Error.model_based_imputer_sparse.clear()

        if self.data is None or len(self.data) == 0:
            self.send("Data", self.data)
            self.modified = False
            return

        data = self.data
        impute_state = [
            (i, var, self.variable_methods.get(i, self.default_method))
            for i, var in enumerate(self.varmodel)
        ]
        def impute_one(index, method, var, data):
            # type: (int, Impute, Variable, Table) -> Any
            if isinstance(method, impute.Model) and data.is_sparse():
                return index, Try.Fail(SparseNotSupported())
            elif isinstance(method, impute.DropInstances):
                return index, Try(method, data, var).map(RowMask)
            elif not method.supports_variable(var):
                return index, Try.Fail(VariableNotSupported(var.name))
            else:
                return index, Try(method, data, var)

        futures = []
        for i, var, method in impute_state:
            f = self.executor.submit(
                impute_one, i, copy.deepcopy(method), var, data)
            futures.append(f)

        w = FutureSetWatcher(futures)
        w.doneAll.connect(self.__commit_finish)
        w.progressChanged.connect(self.__progress_changed)
        # TODO: Record/cache the results as they come in and update var. view.
        self.__task = Task(futures, w)
        self.progressBarInit(processEvents=False)
        self.setBlocking(True)
        # cancel the whole thing at the first problem
        w.exceptionReadyAt.connect(partial(self.__task.cancel))

    @Slot(object)
    def __commit_finish(self, futures):
        assert QThread.currentThread() is self.thread()
        assert len(futures) == len(self.varmodel)
        assert self.__task is not None
        assert self.__task.futures == futures
        assert self.data is not None

        self.setBlocking(False)
        self.progressBarFinished()
        task, self.__task = self.__task, None

        data = self.data
        attributes = []
        class_vars = []
        drop_mask = np.zeros(len(self.data), bool)

        for i, (var, fut) in enumerate(zip(self.varmodel, futures)):
            assert fut.done()
            _i, res = fut.result()
            assert i == _i
            newvar = []
            if isinstance(res, Try.Fail):
                try:
                    raise res.exception
                except SparseNotSupported:
                    self.Error.model_based_imputer_sparse()
                    # ?? break
                except VariableNotSupported:
                    self.warning("Default method can not handle '{}'".
                                 format(var.name))
                except Exception as err:
                    log = logging.getLogger(__name__)
                    log.info("Error for %s", var, exc_info=True)
                    self.Error.imputation_failed(var.name)
                    attributes = class_vars = None
                    break
            elif isinstance(res, Try.Success):
                res = res.value
                if isinstance(res, RowMask):
                    drop_mask |= res.mask
                else:
                    newvar = res

            if isinstance(newvar, Orange.data.Variable):
                newvar = [newvar]

            if i < len(data.domain.attributes):
                attributes.extend(newvar)
            else:
                class_vars.extend(newvar)

        if attributes is None:
            data = None
        else:
            domain = Orange.data.Domain(attributes, class_vars,
                                        data.domain.metas)
            try:
                data = self.data.from_table(domain, data[~drop_mask])
            except Exception as ex:
                self.Error.imputation_failed("Unknown")
                data = None

        self.send("Data", data)
        self.modified = False

    @Slot(int, int)
    def __progress_changed(self, n, d):
        assert QThread.currentThread() is self.thread()
        assert self.__task is not None
        self.progressBarSet(100. * n / d)

    def cancel(self):
        if self.__task is not None:
            task, self.__task = self.__task, None
            task.cancel()
            task.watcher.doneAll.disconnect(self.__commit_finish)
            task.watcher.progressChanged.disconnect(self.__progress_changed)
            concurrent.futures.wait(task.futures)
            task.watcher.flush()
            self.progressBarFinished()
            self.setBlocking(False)

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()

    def send_report(self):
        specific = []
        for i, var in enumerate(self.varmodel):
            method = self.variable_methods.get(i, None)
            if method is not None:
                specific.append("{} ({})".format(var.name, str(method)))

        default = self.default_method.name
        if specific:
            self.report_items((
                ("Default method", default),
                ("Specific imputers", ", ".join(specific))
            ))
        else:
            self.report_items((("Method", default),))

    def _on_var_selection_changed(self):
        indexes = self.selection.selectedIndexes()
        methods = [self.get_method_for_column(i.row()) for i in indexes]

        def method_key(method):
            """
            Decompose method into its type and parameters.
            """
            # The return value should be hashable and  __eq__ comparable
            if isinstance(method, AsDefault):
                return AsDefault, (method.method,)
            elif isinstance(method, impute.Model):
                return impute.Model, (method.learner,)
            elif isinstance(method, impute.Default):
                return impute.Default, (method.default,)
            else:
                return type(method), None

        methods = set(method_key(m) for m in methods)
        selected_vars = [self.varmodel[index.row()] for index in indexes]
        has_discrete = any(var.is_discrete for var in selected_vars)
        fixed_value = None
        value_stack_enabled = False
        current_value_widget = None

        if len(methods) == 1:
            method_type, parameters = methods.pop()
            for i, m in enumerate(self.methods):
                if method_type == type(m):
                    self.variable_button_group.button(i).setChecked(True)

            if method_type is impute.Default:
                (fixed_value,) = parameters

        elif self.variable_button_group.checkedButton() is not None:
            # Uncheck the current button
            self.variable_button_group.setExclusive(False)
            self.variable_button_group.checkedButton().setChecked(False)
            self.variable_button_group.setExclusive(True)
            assert self.variable_button_group.checkedButton() is None

        for method, button in zip(self.methods,
                                  self.variable_button_group.buttons()):
            enabled = all(method.supports_variable(var) for var in
                          selected_vars)
            button.setEnabled(enabled)

        if not has_discrete:
            value_stack_enabled = True
            current_value_widget = self.value_double
        elif len(selected_vars) == 1:
            value_stack_enabled = True
            current_value_widget = self.value_combo
            self.value_combo.clear()
            self.value_combo.addItems(selected_vars[0].values)
        else:
            value_stack_enabled = False
            current_value_widget = None
            self.variable_button_group.button(self.AS_INPUT).setEnabled(False)

        self.value_stack.setEnabled(value_stack_enabled)
        if current_value_widget is not None:
            self.value_stack.setCurrentWidget(current_value_widget)
            if fixed_value is not None:
                if current_value_widget is self.value_combo:
                    self.value_combo.setCurrentIndex(fixed_value)
                elif current_value_widget is self.value_double:
                    self.value_double.setValue(fixed_value)
                else:
                    assert False

    def set_method_for_current_selection(self, method_index):
        indexes = self.selection.selectedIndexes()
        self.set_method_for_indexes(indexes, method_index)

    def set_method_for_indexes(self, indexes, method_index):
        if method_index == self.DEFAULT:
            for index in indexes:
                self.variable_methods.pop(index.row(), None)
        elif method_index == OWImpute.AS_INPUT:
            current = self.value_stack.currentWidget()
            if current is self.value_combo:
                value = self.value_combo.currentIndex()
            else:
                value = self.value_double.value()
            for index in indexes:
                method = impute.Default(default=value)
                self.variable_methods[index.row()] = method
        else:
            method = self.methods[method_index].copy()
            for index in indexes:
                self.variable_methods[index.row()] = method

        self.update_varview(indexes)
        self._invalidate()

    def update_varview(self, indexes=None):
        if indexes is None:
            indexes = map(self.varmodel.index, range(len(self.varmodel)))

        for index in indexes:
            self.varmodel.setData(index, self.get_method_for_column(index.row()), Qt.UserRole)

    def _on_value_selected(self):
        # The fixed 'Value' in the widget has been changed by the user.
        self.variable_button_group.button(self.AS_INPUT).setChecked(True)
        self.set_method_for_current_selection(self.AS_INPUT)

    def reset_variable_methods(self):
        indexes = list(map(self.varmodel.index, range(len(self.varmodel))))
        self.set_method_for_indexes(indexes, self.DEFAULT)
        self.variable_button_group.button(self.DEFAULT).setChecked(True)


import weakref
from functools import partial

from AnyQt.QtCore import QObject, QCoreApplication, QThread, QEvent
from AnyQt.QtCore import pyqtSignal as Signal

Future = concurrent.futures.Future


class FutureSetWatcher(QObject):
    """
    An `QObject` watching the state changes of a list of
    `concurrent.futures.Future` instances

    Note
    ----
    The state change notification signals (`doneAt`, `finishedAt`, ...)
    are always emitted when the control flow reaches the event loop
    (even if the future is already completed when set).

    Note
    ----
    An event loop must be running, otherwise the notifier signals will
    not be emitted.

    Parameters
    ----------
    parent : QObject
        Parent object.
    futures : List[Future]
        A list of future instance to watch.

    Example
    -------
    >>> app = QCoreApplication.instance() or QCoreApplication([])
    >>> fs = [submit(lambda i, j: i ** j, 10, 3) for i in range(10)]
    >>> watcher = FutureSetWatcher(fs)
    >>> watcher.resultReadyAt.connect(
    ...     lambda i, res: print("Result at {}: {}".format(i, res))
    ... )
    >>> watcher.doneAll.connect(app.quit)
    >>> _ = app.exec()
    Result at 1: 1000 ...
    """
    #: Signal emitted when the future at `index` is done (cancelled or
    #: finished)
    doneAt = Signal([int, Future])

    #: Signal emitted when the future at index is finished (i.e. returned
    #: a result)
    finishedAt = Signal(int, Future)

    #: Signal emitted when the future at `index` was cancelled.
    cancelledAt = Signal(int, Future)

    #: Signal emitted with the future's result when successfully
    #: finished.
    resultReadyAt = Signal([int, object])

    #: Signal emitted with the future's exception when finished with an
    #: exception.
    exceptionReadyAt = Signal([int, BaseException])

    #: Signal emitted when a future is done reporting the current completed
    #: count
    progressChanged = Signal([int, int])

    #: Signal emitted when all the futures have completed.
    doneAll = Signal(object)

    def __init__(self, futures=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__futures = None
        self.__countdone = 0
        if futures:
            self.setFutures(futures)

    def setFutures(self, futures):
        # type: (List[Future]) -> None
        """
        Set the future instances to watch.

        Raise a `RuntimeError` if a future is already set.

        Parameters
        ----------
        futures : List[Future]
        """
        if self.__futures is not None:
            raise RuntimeError("already set")
        self.__futures = []
        selfweakref = weakref.ref(self)
        schedule_emit = methodinvoke(self, "__emitpending", (int, Future))

        for i, future in enumerate(futures):
            self.__futures.append(future)

            def on_done(index, f):
                selfref = selfweakref()  # not safe really
                if selfref is None:
                    return
                try:
                    schedule_emit(index, f)
                except RuntimeError:
                    # Ignore RuntimeErrors (when C++ side of QObject is deleted)
                    # (? Use QObject.destroyed and remove the done callback ?)
                    pass

            future.add_done_callback(partial(on_done, i))

    def futures(self):
        # type: () -> List[Future]
        """
        Return a list of all the watched futures.

        Returns
        -------
        futures: List[Futures]
        """
        return list(self.__futures) if self.__futures else []

    @Slot(int, Future)
    def __emitpending(self, index, future):
        # type: (int, Future) -> None
        assert QThread.currentThread() is self.thread()
        assert self.__futures[index] is future
        assert future.done()
        assert self.__countdone < len(self.__futures)
        self.__countdone += 1

        if future.cancelled():
            self.cancelledAt.emit(index, future)
            self.doneAt.emit(index, future)
        elif future.done():
            self.finishedAt.emit(index, future)
            self.doneAt.emit(index, future)
            if future.exception():
                self.exceptionReadyAt.emit(index, future.exception())
            else:
                self.resultReadyAt.emit(index, future.result())
        else:
            assert False

        self.progressChanged.emit(self.__countdone, len(self.__futures))

        if self.__countdone == len(self.__futures):
            self.doneAll.emit(list(self.__futures))

    def flush(self):
        """
        Flush all pending signal emits currently enqueued.
        """
        assert QThread.currentThread() is self.thread()
        QCoreApplication.sendPostedEvents(self, QEvent.MetaCall)


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    logging.basicConfig()
    app = QApplication(list(argv) if argv else [])
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"

    w = OWImpute()
    w.show()
    w.raise_()

    data = Orange.data.Table(filename)
    w.set_data(data)
    w.handleNewSignals()
    app.exec_()
    w.set_data(None)
    w.set_learner(None)
    w.handleNewSignals()
    w.onDeleteWidget()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
