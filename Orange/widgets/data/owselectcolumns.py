import sys
from functools import partial
from typing import Optional  # pylint: disable=unused-import

from AnyQt.QtWidgets import QWidget, QGridLayout
from AnyQt.QtWidgets import QListView  # pylint: disable=unused-import
from AnyQt.QtCore import (
    Qt, QTimer, QSortFilterProxyModel, QItemSelection, QItemSelectionModel,
    QMimeData
)

from Orange.widgets import gui, widget
from Orange.widgets.data.contexthandlers import \
    SelectAttributesDomainContextHandler
from Orange.widgets.settings import ContextSetting, Setting
from Orange.widgets.utils.listfilter import VariablesListItemView, slices, variables_filter
from Orange.widgets.widget import Input, Output
from Orange.data.table import Table
from Orange.widgets.utils import vartype
from Orange.widgets.utils.itemmodels import VariableListModel
import Orange


def source_model(view):
    """ Return the source model for the Qt Item View if it uses
    the QSortFilterProxyModel.
    """
    if isinstance(view.model(), QSortFilterProxyModel):
        return view.model().sourceModel()
    else:
        return view.model()


def source_indexes(indexes, view):
    """ Map model indexes through a views QSortFilterProxyModel
    """
    model = view.model()
    if isinstance(model, QSortFilterProxyModel):
        return list(map(model.mapToSource, indexes))
    else:
        return indexes


class VariablesListItemModel(VariableListModel):
    """
    An Variable list item model specialized for Drag and Drop.
    """
    MIME_TYPE = "application/x-Orange-VariableListModelData"

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid():
            flags |= Qt.ItemIsDragEnabled
        else:
            flags |= Qt.ItemIsDropEnabled
        return flags

    def supportedDropActions(self):
        return Qt.MoveAction  # pragma: no cover

    def supportedDragActions(self):
        return Qt.MoveAction  # pragma: no cover

    def mimeTypes(self):
        return [self.MIME_TYPE]

    def mimeData(self, indexlist):
        """
        Reimplemented.

        For efficiency reasons only the variable instances are set on the
        mime data (under `'_items'` property)
        """
        items = [self[index.row()] for index in indexlist]
        mime = QMimeData()
        # the encoded 'data' is empty, variables are passed by properties
        mime.setData(self.MIME_TYPE, b'')
        mime.setProperty("_items", items)
        return mime

    def dropMimeData(self, mime, action, row, column, parent):
        """
        Reimplemented.
        """
        if action == Qt.IgnoreAction:
            return True  # pragma: no cover
        if not mime.hasFormat(self.MIME_TYPE):
            return False  # pragma: no cover
        variables = mime.property("_items")
        if variables is None:
            return False  # pragma: no cover
        if row < 0:
            row = self.rowCount()

        self[row:row] = variables
        return True


class OWSelectAttributes(widget.OWWidget):
    # pylint: disable=too-many-instance-attributes
    name = "Select Columns"
    description = "Select columns from the data table and assign them to " \
                  "data features, classes or meta variables."
    icon = "icons/SelectColumns.svg"
    priority = 100
    keywords = ["filter"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)
        features = Output("Features", widget.AttributeList, dynamic=False)

    want_main_area = False
    want_control_area = True

    settingsHandler = SelectAttributesDomainContextHandler()
    domain_role_hints = ContextSetting({})
    auto_commit = Setting(True)

    def __init__(self):
        super().__init__()
        # Schedule interface updates (enabled buttons) using a coalescing
        # single shot timer (complex interactions on selection and filtering
        # updates in the 'available_attrs_view')
        self.__interface_update_timer = QTimer(self, interval=0, singleShot=True)
        self.__interface_update_timer.timeout.connect(
            self.__update_interface_state)
        # The last view that has the selection for move operation's source
        self.__last_active_view = None  # type: Optional[QListView]

        def update_on_change(view):
            # Schedule interface state update on selection change in `view`
            self.__last_active_view = view
            self.__interface_update_timer.start()

        self.controlArea = QWidget(self.controlArea)
        self.layout().addWidget(self.controlArea)
        layout = QGridLayout()
        self.controlArea.setLayout(layout)
        layout.setContentsMargins(4, 4, 4, 4)
        box = gui.vBox(self.controlArea, "Available Variables",
                       addToLayout=False)

        self.available_attrs = VariablesListItemModel()
        filter_edit, self.available_attrs_view = variables_filter(
            parent=self, model=self.available_attrs)
        box.layout().addWidget(filter_edit)

        def dropcompleted(action):
            if action == Qt.MoveAction:
                self.commit()

        self.available_attrs_view.selectionModel().selectionChanged.connect(
            partial(update_on_change, self.available_attrs_view))
        self.available_attrs_view.dragDropActionDidComplete.connect(dropcompleted)

        box.layout().addWidget(self.available_attrs_view)
        layout.addWidget(box, 0, 0, 3, 1)

        box = gui.vBox(self.controlArea, "Features", addToLayout=False)
        self.used_attrs = VariablesListItemModel()
        self.used_attrs_view = VariablesListItemView(
            acceptedType=(Orange.data.DiscreteVariable,
                          Orange.data.ContinuousVariable))

        self.used_attrs_view.setModel(self.used_attrs)
        self.used_attrs_view.selectionModel().selectionChanged.connect(
            partial(update_on_change, self.used_attrs_view))
        self.used_attrs_view.dragDropActionDidComplete.connect(dropcompleted)
        box.layout().addWidget(self.used_attrs_view)
        layout.addWidget(box, 0, 2, 1, 1)

        box = gui.vBox(self.controlArea, "Target Variable", addToLayout=False)
        self.class_attrs = VariablesListItemModel()
        self.class_attrs_view = VariablesListItemView(
            acceptedType=(Orange.data.DiscreteVariable,
                          Orange.data.ContinuousVariable))
        self.class_attrs_view.setModel(self.class_attrs)
        self.class_attrs_view.selectionModel().selectionChanged.connect(
            partial(update_on_change, self.class_attrs_view))
        self.class_attrs_view.dragDropActionDidComplete.connect(dropcompleted)
        self.class_attrs_view.setMaximumHeight(72)
        box.layout().addWidget(self.class_attrs_view)
        layout.addWidget(box, 1, 2, 1, 1)

        box = gui.vBox(self.controlArea, "Meta Attributes", addToLayout=False)
        self.meta_attrs = VariablesListItemModel()
        self.meta_attrs_view = VariablesListItemView(
            acceptedType=Orange.data.Variable)
        self.meta_attrs_view.setModel(self.meta_attrs)
        self.meta_attrs_view.selectionModel().selectionChanged.connect(
            partial(update_on_change, self.meta_attrs_view))
        self.meta_attrs_view.dragDropActionDidComplete.connect(dropcompleted)
        box.layout().addWidget(self.meta_attrs_view)
        layout.addWidget(box, 2, 2, 1, 1)

        bbox = gui.vBox(self.controlArea, addToLayout=False, margin=0)
        layout.addWidget(bbox, 0, 1, 1, 1)

        self.up_attr_button = gui.button(bbox, self, "Up",
                                         callback=partial(self.move_up, self.used_attrs_view))
        self.move_attr_button = gui.button(bbox, self, ">",
                                           callback=partial(self.move_selected,
                                                            self.used_attrs_view)
                                          )
        self.down_attr_button = gui.button(bbox, self, "Down",
                                           callback=partial(self.move_down, self.used_attrs_view))

        bbox = gui.vBox(self.controlArea, addToLayout=False, margin=0)
        layout.addWidget(bbox, 1, 1, 1, 1)

        self.up_class_button = gui.button(bbox, self, "Up",
                                          callback=partial(self.move_up, self.class_attrs_view))
        self.move_class_button = gui.button(bbox, self, ">",
                                            callback=partial(self.move_selected,
                                                             self.class_attrs_view,
                                                             exclusive=False)
                                           )
        self.down_class_button = gui.button(bbox, self, "Down",
                                            callback=partial(self.move_down, self.class_attrs_view))

        bbox = gui.vBox(self.controlArea, addToLayout=False, margin=0)
        layout.addWidget(bbox, 2, 1, 1, 1)
        self.up_meta_button = gui.button(bbox, self, "Up",
                                         callback=partial(self.move_up, self.meta_attrs_view))
        self.move_meta_button = gui.button(bbox, self, ">",
                                           callback=partial(self.move_selected,
                                                            self.meta_attrs_view)
                                          )
        self.down_meta_button = gui.button(bbox, self, "Down",
                                           callback=partial(self.move_down, self.meta_attrs_view))

        autobox = gui.auto_commit(None, self, "auto_commit", "Send")
        layout.addWidget(autobox, 3, 0, 1, 3)
        reset = gui.button(None, self, "Reset", callback=self.reset, width=120)
        autobox.layout().insertWidget(0, reset)
        autobox.layout().insertStretch(1, 20)

        layout.setRowStretch(0, 4)
        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 2)
        layout.setHorizontalSpacing(0)
        self.controlArea.setLayout(layout)

        self.data = None
        self.output_data = None
        self.original_completer_items = []

        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)
        self.resize(500, 600)

    @Inputs.data
    def set_data(self, data=None):
        self.update_domain_role_hints()
        self.closeContext()
        self.data = data

        if data is not None:
            self.openContext(data)
            all_vars = data.domain.variables + data.domain.metas

            var_sig = lambda attr: (attr.name, vartype(attr))

            domain_hints = {var_sig(attr): ("attribute", i)
                            for i, attr in enumerate(data.domain.attributes)}

            domain_hints.update({var_sig(attr): ("meta", i)
                                 for i, attr in enumerate(data.domain.metas)})

            if data.domain.class_vars:
                domain_hints.update(
                    {var_sig(attr): ("class", i)
                     for i, attr in enumerate(data.domain.class_vars)})

            # update the hints from context settings
            domain_hints.update(self.domain_role_hints)

            attrs_for_role = lambda role: [
                (domain_hints[var_sig(attr)][1], attr)
                for attr in all_vars if domain_hints[var_sig(attr)][0] == role]

            attributes = [
                attr for place, attr in sorted(attrs_for_role("attribute"),
                                               key=lambda a: a[0])]
            classes = [
                attr for place, attr in sorted(attrs_for_role("class"),
                                               key=lambda a: a[0])]
            metas = [
                attr for place, attr in sorted(attrs_for_role("meta"),
                                               key=lambda a: a[0])]
            available = [
                attr for place, attr in sorted(attrs_for_role("available"),
                                               key=lambda a: a[0])]

            self.used_attrs[:] = attributes
            self.class_attrs[:] = classes
            self.meta_attrs[:] = metas
            self.available_attrs[:] = available
        else:
            self.used_attrs[:] = []
            self.class_attrs[:] = []
            self.meta_attrs[:] = []
            self.available_attrs[:] = []

        if data is not None:
            self.info.set_input_summary(
                summarize(data.domain),
                summarize_domain_long(data.domain),
                format=Qt.RichText
            )
        else:
            self.info.set_input_summary(self.info.NoInput)
        self.unconditional_commit()

    def update_domain_role_hints(self):
        """ Update the domain hints to be stored in the widgets settings.
        """
        hints_from_model = lambda role, model: [
            ((attr.name, vartype(attr)), (role, i))
            for i, attr in enumerate(model)]
        hints = dict(hints_from_model("available", self.available_attrs))
        hints.update(hints_from_model("attribute", self.used_attrs))
        hints.update(hints_from_model("class", self.class_attrs))
        hints.update(hints_from_model("meta", self.meta_attrs))
        self.domain_role_hints = hints

    def selected_rows(self, view):
        """ Return the selected rows in the view.
        """
        rows = view.selectionModel().selectedRows()
        model = view.model()
        if isinstance(model, QSortFilterProxyModel):
            rows = [model.mapToSource(r) for r in rows]
        return [r.row() for r in rows]

    def move_rows(self, view, rows, offset):
        model = view.model()
        newrows = [min(max(0, row + offset), len(model) - 1) for row in rows]

        for row, newrow in sorted(zip(rows, newrows), reverse=offset > 0):
            model[row], model[newrow] = model[newrow], model[row]

        selection = QItemSelection()
        for nrow in newrows:
            index = model.index(nrow, 0)
            selection.select(index, index)
        view.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect)

        self.commit()

    def move_up(self, view):
        selected = self.selected_rows(view)
        self.move_rows(view, selected, -1)

    def move_down(self, view):
        selected = self.selected_rows(view)
        self.move_rows(view, selected, 1)

    def move_selected(self, view, exclusive=False):
        if self.selected_rows(view):
            self.move_selected_from_to(view, self.available_attrs_view)
        elif self.selected_rows(self.available_attrs_view):
            self.move_selected_from_to(self.available_attrs_view, view,
                                       exclusive)

    def move_selected_from_to(self, src, dst, exclusive=False):
        self.move_from_to(src, dst, self.selected_rows(src), exclusive)

    def move_from_to(self, src, dst, rows, exclusive=False):
        src_model = source_model(src)
        attrs = [src_model[r] for r in rows]

        for s1, s2 in reversed(list(slices(rows))):
            del src_model[s1:s2]

        dst_model = source_model(dst)

        dst_model.extend(attrs)

        self.commit()

    def __update_interface_state(self):
        last_view = self.__last_active_view
        if last_view is not None:
            self.update_interface_state(last_view)

    def update_interface_state(self, focus=None, selected=None, deselected=None):
        for view in [self.available_attrs_view, self.used_attrs_view,
                     self.class_attrs_view, self.meta_attrs_view]:
            if view is not focus and not view.hasFocus() \
                    and view.selectionModel().hasSelection():
                view.selectionModel().clear()

        def selected_vars(view):
            model = source_model(view)
            return [model[i] for i in self.selected_rows(view)]

        available_selected = selected_vars(self.available_attrs_view)
        attrs_selected = selected_vars(self.used_attrs_view)
        class_selected = selected_vars(self.class_attrs_view)
        meta_selected = selected_vars(self.meta_attrs_view)

        available_types = set(map(type, available_selected))
        all_primitive = all(var.is_primitive()
                            for var in available_types)

        move_attr_enabled = (available_selected and all_primitive) or \
                             attrs_selected

        self.move_attr_button.setEnabled(bool(move_attr_enabled))
        if move_attr_enabled:
            self.move_attr_button.setText(">" if available_selected else "<")

        move_class_enabled = (all_primitive and available_selected) or class_selected

        self.move_class_button.setEnabled(bool(move_class_enabled))
        if move_class_enabled:
            self.move_class_button.setText(">" if available_selected else "<")
        move_meta_enabled = available_selected or meta_selected

        self.move_meta_button.setEnabled(bool(move_meta_enabled))
        if move_meta_enabled:
            self.move_meta_button.setText(">" if available_selected else "<")

        self.__last_active_view = None
        self.__interface_update_timer.stop()

    def commit(self):
        self.update_domain_role_hints()
        if self.data is not None:
            attributes = list(self.used_attrs)
            class_var = list(self.class_attrs)
            metas = list(self.meta_attrs)

            domain = Orange.data.Domain(attributes, class_var, metas)
            newdata = self.data.transform(domain)
            self.output_data = newdata
            self.Outputs.data.send(newdata)
            self.Outputs.features.send(widget.AttributeList(attributes))
            self.info.set_output_summary(
                summarize(newdata.domain),
                summarize_domain_long(newdata.domain),
                format=Qt.RichText
            )
        else:
            self.output_data = None
            self.Outputs.data.send(None)
            self.Outputs.features.send(None)
            self.info.set_output_summary(self.info.NoOutput)

    def reset(self):
        if self.data is not None:
            self.available_attrs[:] = []
            self.used_attrs[:] = self.data.domain.attributes
            self.class_attrs[:] = self.data.domain.class_vars
            self.meta_attrs[:] = self.data.domain.metas
            self.update_domain_role_hints()
            self.commit()

    def send_report(self):
        if not self.data or not self.output_data:
            return
        in_domain, out_domain = self.data.domain, self.output_data.domain
        self.report_domain("Input data", self.data.domain)
        if (in_domain.attributes, in_domain.class_vars, in_domain.metas) == (
                out_domain.attributes, out_domain.class_vars, out_domain.metas):
            self.report_paragraph("Output data", "No changes.")
        else:
            self.report_domain("Output data", self.output_data.domain)
            diff = list(set(in_domain.variables + in_domain.metas) -
                        set(out_domain.variables + out_domain.metas))
            if diff:
                text = "%i (%s)" % (len(diff), ", ".join(x.name for x in diff))
                self.report_items((("Removed", text),))


import collections


def summarize(domain, categorize_types=True):
    """
    Parameters
    ----------
    domain
    categorize_types

    Returns
    -------

    Examples
    --------
    """
    nfeat = len(domain.attributes)
    ntarg = len(domain.class_vars)
    nmeta = len(domain.metas)

    part_feat, part_target, part_meta = None, None, None
    part_feat = str(nfeat)

    if categorize_types:
        part_feat = summarize_var_type_counts(domain.attributes)

    if ntarg:
        if categorize_types:
            part_target = summarize_var_type_counts(domain.class_vars)
        else:
            part_target = str(ntarg)

    if nmeta:
        if categorize_types:
            part_meta = summarize_var_type_counts(domain.metas)
        else:
            part_meta = str(nmeta)

    text = part_feat
    if part_target:
        text = text + " | " + part_target
    if part_meta:
        text = text + " : " + part_meta
    return text


def var_type_name(vartype):
    # type: (Union[Type[Orange.data.Variable], Orange.data.Variable]) -> str
    """
    Return the variable type name for display (i.e. 'categorical', 'numeric',
    ...)

    Parameters
    ----------
    vartype: Orange.data.Variable

    Returns
    -------
    name: str
        Display name for the variable type
    """
    if not isinstance(vartype, type):
        vartype = type(vartype)

    if issubclass(vartype, Orange.data.DiscreteVariable):
        return "categorical"
    elif issubclass(vartype, Orange.data.TimeVariable):
        return "time"
    elif issubclass(vartype, Orange.data.ContinuousVariable):
        return "numeric"
    elif issubclass(vartype, Orange.data.StringVariable):
        return "string"
    else:
        return vartype.__qualname__.lower()


def summarize_var_type_counts(vars):
    t = collections.Counter([var_type_name(v)[0].upper() for v in vars])
    N = len(vars)
    if len(t) == 0:
        return "0"
    elif len(t) == 1:
        return "{1}{0}".format(*next(iter(t.items())))
    else:
        return "{} ({})".format(
            N, " + ".join("{}{}".format(count, typecode)
                          for typecode, count in t.items())
        )


def summarize_domain_long_items(domain, categorize_types=True,
                                categorize_roles=True, drop_empty=False):
    # type: (Orange.data.Domain, bool, bool, str) -> List[Tuple[str, str]]
    """
    Return a 'field list' (name, body) pairs describing the domain.

    Parameters
    ----------
    domain : Orange.data.Domain
        Domain to describe.
    categorize_types : bool
        Summarize constituent variable types.
    categorize_roles : bool
        Summarize variables by domain role (i.e. report attributes,
        class_vars metas separately).
    drop_empty : bool
        Omit field summaries that contain no variables instead of
        reporting 0 counts.

    Returns
    -------
    fields : List[Tuple[str, str]]
        `name, body` pairs describing the domain.

    See also
    --------
    render_field_list
    render_description_list

    Examples
    --------
    >>> iris = Orange.data.Table("iris")
    >>> summarize_domain_long_items(iris.domain)  # doctest: +ELLIPSIS
    [('Features', '4 (numeric)'), ('Target', 'categorical with 3 values'), ...
    >>> hd = Orange.data.Table("heart_disease")
    >>> summarize_domain_long_items(hd.domain)    # doctest: +ELLIPSIS
    [('Features', '13 (categorical: 7, numeric: 6)'), ...
    >>> summarize_domain_long_items(hd.domain, categorize_types=False) # doctest: +ELLIPSIS
    [('Features', '13'), ('Target', 'categorical ...
    >>> summarize_domain_long_items(hd.domain, categorize_roles=False)
    [('Columns', '14 (categorical: 8, numeric: 6)')]
    """
    # Preferred order of variable types in the summary
    order = [
        Orange.data.DiscreteVariable,
        Orange.data.ContinuousVariable,
        Orange.data.TimeVariable,
        Orange.data.StringVariable,
    ]

    def summarize_variable(var):
        # type: (Orange.data.Variable) -> str
        if var.is_discrete:
            return "{} with {} value{s}".format(
                var_type_name(var), len(var.values),
                s="s" if len(var.values) != 1 else ""
            )
        else:
            return var_type_name(var)

    def summarize_variable_types(variables):
        # type: (List[Orange.data.Variable]) -> str
        counts = collections.Counter(type(var) for var in variables)
        if len(counts) == 0:
            return "none"
        elif len(counts) == 1:
            var_type, count = next(iter(counts.items()))
            return var_type_name(var_type)
        else:
            def index(type):
                try:
                    return order.index(type)
                except ValueError:
                    return sys.maxsize
            counts = sorted(counts.items(), key=lambda item: index(item[0]))
            return ", ".join("{}: {}".format(var_type_name(vartype), count)
                            for vartype, count in counts)

    def summarize_variables(variables, categorize_types=True,
                            fold_single=False):
        size = len(variables)
        if size == 0:
            body = "none"
        elif size == 1 and fold_single:
            body = summarize_variable(variables[0])
        elif categorize_types:
            body = "{} ({})".format(size, summarize_variable_types(variables))
        else:
            body = "{}".format(size)
        return body

    def describe_part(name, variables, singular=False):
        body = summarize_variables(
            variables, categorize_types=categorize_types, fold_single=singular
        )
        if singular and len(variables) > 1:
            name = name + "s"
        return name, body

    if not categorize_roles:
        return [describe_part("Column", (domain.attributes + domain.class_vars +
                                         domain.metas),
                              singular=True)]

    parts = []
    if domain.attributes or not drop_empty:
        parts.append(describe_part("Features", domain.attributes))

    if domain.class_vars or not drop_empty:
        parts.append(describe_part("Target", domain.class_vars, singular=True))

    if domain.metas or not drop_empty:
        parts.append(describe_part("Metas", domain.metas))
    return parts


def summarize_domain_long(domain, categorize_types=True, roles=True, drop_empty=False,
                          type="table"):
    fields = summarize_domain_long_items(domain, categorize_types, roles, drop_empty=drop_empty)
    return render_field_list([(name + ":", body) for name, body in fields], type)


def render_field_list(items, tag="table"):
    # type: (Sequence[Tuple[str, str], str]) -> str
    if tag == "dl":
        dl, dt, dd = "dl", "dt", "dd"
    elif tag == "table":
        dl, dt, dd = "table", "th", "td"
    else:
        raise ValueError
    dtframe = "<{dt} class='field-name'>{{title}}</{dt}>".format(dt=dt)
    ddframe = "<{dd} class='field-body'>{{description}}</{dd}>".format(dd=dd)

    if tag == "table":
        dlframe = ("<table class='field-list'>\n"
                   "{}"
                   "</table>"
                   )
        ditemframe = "<tr>" + dtframe + ddframe + "</tr>"
    else:
        dlframe = ("<{dl} class='field-list'>\n"
                   "{{}}\n"
                   "</{dl}>\n"
                   .format(dl=dl))
        ditemframe = dtframe + ddframe

    parts_rendered = []
    for title, content in items:
        parts_rendered.append(
            ditemframe.format(title=title, description=content))
    return dlframe.format("  \n".join(parts_rendered))


def render_description_list(items):
    # type: (Sequence[Tuple[str, str]]) -> str
    return render_description_list(items, type="dl")

field_list_item_template = '<tr><th class="field-name">{name}</th><td class="field-body">{body}</tr>'


def render_field_list_items(
        items,
        itemformat="<dt>{name}</dt><dd>{body}</dd>".format):
    return (itemformat(title=name, body=body) for name, body in items)


def main(argv=None):  # pragma: no cover
    from AnyQt.QtWidgets import QApplication
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(list(argv))

    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "brown-selected"

    w = OWSelectAttributes()
    data = Orange.data.Table(filename)
    w.set_data(data)
    w.show()
    w.raise_()
    rval = app.exec_()
    w.set_data(None)
    w.saveSettings()
    return rval


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
