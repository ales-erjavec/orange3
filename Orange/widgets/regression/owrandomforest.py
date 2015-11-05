import sys

from PyQt4 import QtGui
from PyQt4.QtGui import QLabel, QGridLayout
from PyQt4.QtCore import Qt

import Orange.data

from Orange.preprocess.preprocess import Preprocess
from Orange.regression import random_forest

from Orange.widgets import widget, settings, gui
from Orange.widgets.utils.sql import check_sql_input


class OWRandomForest(widget.OWWidget):
    name = "Random Forest Regression"
    description = "Random forest regression algorithm."
    icon = "icons/RandomForestRegression.svg"
    priority = 320

    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Preprocessor", Preprocess, "set_preprocessor")]
    outputs = [("Learner", random_forest.RandomForestRegressionLearner),
               ("Classifier", random_forest.RandomForestRegressionModel)]

    want_main_area = False
    resizing_enabled = False

    learner_name = settings.Setting("Random Forest Regression Learner")
    n_estimators = settings.Setting(10)
    max_features = settings.Setting(5)
    use_max_features = settings.Setting(False)
    random_state = settings.Setting(0)
    use_random_state = settings.Setting(False)
    max_depth = settings.Setting(3)
    use_max_depth = settings.Setting(False)
    max_leaf_nodes = settings.Setting(5)
    use_max_leaf_nodes = settings.Setting(True)
    index_output = settings.Setting(0)

    #: Split Control (number of features to consider)
    All, Log2, Sqrt, FixedP, FixedN = 0, 1, 2, 3, 4
    max_features_type = 0

    def __init__(self):
        super().__init__()

        self.data = None
        self.preprocessors = None

        # Learner name
        box = gui.widgetBox(self.controlArea, self.tr("Name"))
        gui.lineEdit(box, self, "learner_name")

        # Basic properties
        form = QGridLayout()
        form.setColumnStretch(0, 100)
        basic_box = gui.widgetBox(
            self.controlArea, "Basic properties", orientation=form)

        form.addWidget(QLabel(self.tr("Number of trees in the forest: ")),
                       0, 0, Qt.AlignLeft)
        spin = gui.spin(basic_box, self, "n_estimators", minv=1, maxv=1e4,
                        callback=self.settingsChanged, addToLayout=False,
                        #controlWidth=50
                        )
        form.addWidget(spin, 0, 1)

        form.addWidget(QtGui.QLabel("At each split, consider:"),
                       1, 0)
#         layout = QtGui.QHBoxLayout()
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.addWidget(QtGui.QLabel("At each split, consider"))
#         layout.addWidget(
        form.addWidget(
            gui.comboBox(box, self, "max_features_type",
                         items=["all features",
                                "log2(# features)",
                                "sqrt(# features)",
                                "a proportion of features",
                                "a fixed number of features"],
                         callback=self._max_features_type_changed,
                         addToLayout=False),
#         )
            2, 0, Qt.AlignRight
        )
        self.max_features_spin = max_features_spin = gui.spin(
            basic_box, self, "max_features", 2, 50, addToLayout=False,
            callback=self.settingsChanged, #controlWidth=50
        )
        self.max_features_spin.setSuffix("%")
        _st = QtGui.QStackedWidget()
        _st.addWidget(self.max_features_spin)
        _st.addWidget(QtGui.QWidget())
        print(_st.getContentsMargins())
        print(_st.layout().getContentsMargins())
        _st.layout().setContentsMargins(0, 0, 0, 0)
        _st.setSizePolicy(
            QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Preferred)
        print(_st.layout().getContentsMargins())
#         layout.addWidget(self.max_features_spin)
#         form.addLayout(layout, 1, 0, 1, 2)
#         form.addWidget(self.max_features_spin, 2, 1)
        form.addWidget(_st, 2, 1,)

#         max_features_cb = gui.checkBox(
#             basic_box, self, "use_max_features",
#             callback=self.settingsChanged, addToLayout=False,
#             label="Consider a number of best attributes at each split")
# 
#         max_features_spin = gui.spin(
#             basic_box, self, "max_features", 2, 50, addToLayout=False,
#             callback=self.settingsChanged, controlWidth=50)

#         form.addWidget(max_features_cb, 1, 0, Qt.AlignLeft)
#         form.addWidget(max_features_spin, 1, 1, Qt.AlignRight)

        random_state_cb = gui.checkBox(
            basic_box, self, "use_random_state", callback=self.settingsChanged,
            addToLayout=False, label="Use seed for random generator:")
        random_state_spin = gui.spin(
            basic_box, self, "random_state", 0, 2 ** 31 - 1, addToLayout=False,
            callback=self.settingsChanged, #controlWidth=50
            )
        random_state_spin.setSizePolicy(
            QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Preferred)

        form.addWidget(random_state_cb, 3, 0, Qt.AlignLeft)
        form.addWidget(random_state_spin, 3, 1, )
        self._max_features_spin = max_features_spin
        self._random_state_spin = random_state_spin

        # Growth control
        form = QGridLayout()
        growth_box = gui.widgetBox(
            self.controlArea, "Growth control", orientation=form)

        max_depth_cb = gui.checkBox(
            growth_box, self, "use_max_depth",
            label="Set maximal depth of individual trees",
            callback=self.settingsChanged,
            addToLayout=False)

        max_depth_spin = gui.spin(
            growth_box, self, "max_depth", 2, 50, addToLayout=False,
            callback=self.settingsChanged)

        form.addWidget(max_depth_cb, 3, 0, Qt.AlignLeft)
        form.addWidget(max_depth_spin, 3, 1, Qt.AlignRight)

        max_leaf_nodes_cb = gui.checkBox(
            growth_box, self, "use_max_leaf_nodes",
            label="Stop splitting nodes with maximum instances: ",
            callback=self.settingsChanged, addToLayout=False)

        max_leaf_nodes_spin = gui.spin(
            growth_box, self, "max_leaf_nodes", 0, 100, addToLayout=False,
            callback=self.settingsChanged)

        form.addWidget(max_leaf_nodes_cb, 4, 0, Qt.AlignLeft)
        form.addWidget(max_leaf_nodes_spin, 4, 1, Qt.AlignRight)
        self._max_depth_spin = max_depth_spin
        self._max_leaf_nodes_spin = max_leaf_nodes_spin

        gui.button(self.controlArea, self, "&Apply",
                   callback=self.apply, default=True)

        self.settingsChanged()
        self._ensure_state()
        self.apply()

    @check_sql_input
    def set_data(self, data):
        """Set the input train data set."""
        self.data = data
        if data is not None:
            self.apply()

    def set_preprocessor(self, preproc):
        if preproc is None:
            self.preprocessors = None
        else:
            self.preprocessors = (preproc,)
        self.apply()

    def apply(self):
        common_args = dict()
        common_args["n_estimators"] = self.n_estimators
        if self.use_max_features:
            common_args["max_features"] = self.max_features
        if self.use_random_state:
            common_args["random_state"] = self.random_state
        if self.use_max_depth:
            common_args["max_depth"] = self.max_depth
        if self.use_max_leaf_nodes:
            common_args["max_leaf_nodes"] = self.max_leaf_nodes

        learner = random_forest.RandomForestRegressionLearner(
            preprocessors=self.preprocessors, **common_args
        )
        learner.name = self.learner_name
        classifier = None

        if self.data is not None:
            self.error(0)
            if not learner.check_learner_adequacy(self.data.domain):
                self.error(0, learner.learner_adequacy_err_msg)
            else:
                classifier = learner(self.data)
                classifier.name = self.learner_name

        self.send("Learner", learner)
        self.send("Classifier", classifier)

    def settingsChanged(self):
        self._max_features_spin.setEnabled(self.use_max_features)
        self._random_state_spin.setEnabled(self.use_random_state)
        self._max_depth_spin.setEnabled(self.use_max_depth)
        self._max_leaf_nodes_spin.setEnabled(self.use_max_leaf_nodes)
        self._ensure_state()

    def _max_features_type_changed(self):
        self._ensure_state()

    def _ensure_state(self):
        enabled = (self.max_features_type == OWRandomForest.FixedN or
                   self.max_features_type == OWRandomForest.FixedP)

        self.max_features_spin.setVisible(enabled)
#             self.max_features_type == OWRandomForest.FixedN or
#             self.max_features_type == OWRandomForest.FixedP)

        self._max_features_spin.setEnabled(enabled)

        if self.max_features_type == OWRandomForest.FixedN:
            self.max_features_spin.setSuffix("")
            self.max_features_spin.setMaximum(100)
        elif self.max_features_type == OWRandomForest.FixedP:
            self.max_features_spin.setSuffix(" \N{FULLWIDTH PERCENT SIGN}")
            self.max_features_spin.setMaximum(100)


def main(argv=sys.argv):
    app = QtGui.QApplication(list(argv))
    argv = app.argv()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "housing"

    w = OWRandomForest()
    w.set_preprocessor(None)
    w.set_data(None)
    w.handleNewSignals()
    w.set_data(Orange.data.Table(filename))
    w.show()
    w.raise_()
    app.exec_()
    w.set_data(None)
    w.set_preprocessor(None)
    w.handleNewSignals()
    return 0

if __name__ == "__main__":
    sys.exit(main())
