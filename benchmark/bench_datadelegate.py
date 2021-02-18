import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication

from Orange.data import Table
from Orange.widgets.utils.itemdelegates import DataDelegate
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.tableview import TableView
from .base import benchmark, Benchmark
from orangewidget.tests.base import GuiTest


class BenchDataDelegate(GuiTest, Benchmark):
    def setUp(self) -> None:
        super().setUp()
        data = Table("brown-selected")
        self.view = TableView()
        self.delegate = DataDelegate(
            self.view, roles=(Qt.DisplayRole, ) #Qt.BackgroundRole)
        )
        self.view.setItemDelegate(self.delegate)
        self.model = TableModel(data)
        self.view.setModel(self.model)
        self.view.resize(1024, 760)

    def tearDown(self) -> None:
        super().tearDown()
        self.view.setModel(None)
        del self.view
        del self.model
        del self.delegate

    @benchmark(number=3, warmup=1, repeat=10)
    def bench_paint(self):
        _ = self.view.grab()

    @benchmark(number=3, warmup=1, repeat=10)
    def bench_init_style_option(self):
        delegate = self.delegate
        opt = self.view.viewOptions()
        index = self.model.index(0, 0)
        delegate.initStyleOption(opt, index)
