"""
Utilities for summarizing/reporting formatting html fragments.
"""
# TODO: Separate data summary and html formatting tools
import sys
from collections import Counter
from xml.sax.saxutils import escape

# pylint: disable=unsused-import
from typing import Sequence, Union, List, Tuple
try:
    from typing import Type  # Only added in 3.5.2
except ImportError:
    pass

import Orange.data
import Orange.data.sql.table

from Orange.widgets.utils.report import plural


field_list_css = """\
/* vertical row header cell */
tr > th.field-name {
    text-align: right;
    padding-right: 0.2em;
    font-weight: bold;
}
dd.description-body > tr > th.field-name {
    text-align: left;
    font-weight: normal;
}
"""


def _render_field_list(items, tag="table"):
    # type: (Sequence[Tuple[str, str], str]) -> str
    """
    Render a sequence of (name, description) pairs into a specified html
    element

    Parameters
    ----------
    items : Sequence[Tuple[str, str]]
        A sequence of (name, description) pairs

    Returns
    -------
    text : str
    """
    if tag == "dl":
        dl, dt, dd = "dl", "dt", "dd"
    elif tag == "table":
        dl, dt, dd = "table", "th", "td"
    else:
        raise ValueError
    dlframe = "<{dl} class='field-list'>\n{{}}\n</{dl}>".format(dl=dl)
    dtframe = "<{dt} class='field-name'>{{title}}</{dt}>".format(dt=dt)
    ddframe = "<{dd} class='field-body'>{{description}}</{dd}>".format(dd=dd)

    if tag == "table":
        ditemframe = "<tr>" + dtframe + ddframe + "</tr>"
    else:
        ditemframe = dtframe + ddframe
    return dlframe.format(
        "  \n".join(ditemframe.format(title=escape(title),
                                      description=escape(content))
                    for title, content in items))


def render_field_list_table(items):
    return _render_field_list(items, tag="table")


def render_field_list_dl(items):
    return _render_field_list(items, tag="dl")


def render_field_list(items):
    return _render_field_list(items, tag="table")


def render_description_list(items):
    return _render_field_list(items, tag="dl")


def summary_table_shape_inline_(table):
    # type: (Orange.data.Table) -> str
    """
    Return a short inline shape summary of `table`.

    Parameters
    ----------
    table : Orange.data.Table

    Returns
    -------
    text : str
        A short shape description in the format N x M where M consists of
        counts for features, classes and metas reported separately
        (if required).

    Example
    -------
    >>> hd = Orange.data.Table("heart_disease")
    >>> summary_table_shape_inline_(hd)
    '303 × (13 | 1)'
    """
    # 12 x 5   # X only
    # 12 x (4 | 1)  # X + Y
    # 12 x (4 | 1 : 2)  # X + Y + M
    # 12 x (0 | 1 : 1)
    # 12 x (0 : 1)
    # 12 x 0
    N = len(table)
    M_a = len(table.domain.attributes)
    M_c = len(table.domain.class_vars)
    M_m = len(table.domain.metas)
    fmt = ["{M_a}"]
    if M_c:
        fmt += ["| {M_c}"]
    if M_m:
        fmt += [": {M_m}"]
    fmt = " ".join(fmt)
    if M_c + M_m:
        fmt = "(" + fmt + ")"
    return ("{N} \N{MULTIPLICATION SIGN} " + fmt).format(N=N, M_a=M_a, M_c=M_c, M_m=M_m)


def summary_table_shape_inline(table, fmt="{rows} \N{MULTIPLICATION SIGN} {columns}"):
    # type: (Orange.data.Table) -> str
    """
    Return a short inline shape summary of `table`.

    Parameters
    ----------
    table : Orange.data.Table

    Returns
    -------
    text : str
        A short shape description in the format N x M where M consists of
        counts for features, targets and metas reported separately.

    Example
    -------
    >>> zoo = Orange.data.Table("zoo")
    >>> summary_table_shape_inline(zoo)
    '101 × (16 + 1 + 1)'
    >>> summary_table_shape_inline(zoo[:, :2])  # X only
    '101 × 2'
    >>> summary_table_shape_inline(zoo[:, [0, 1, "type"]])  # X and Y
    '101 × (2 + 1)'
    >>> summary_table_shape_inline(zoo[:, ["type"]])  # Y only
    '101 × 1 target'
    >>> summary_table_shape_inline(zoo[:, ["type", "name"]])  # Y and meta
    '101 × (1 target + 1 meta)'
    >>> summary_table_shape_inline(zoo[:, []])  # no columns
    '101 × 0'
    """
    N = len(table)
    M_a = len(table.domain.attributes)
    M_c = len(table.domain.class_vars)
    M_m = len(table.domain.metas)
    M = [(M_a, 'feature'), (M_c, 'target'), (M_m, 'meta')]
    M_parts = []
    trip = False
    for c, tag in M:
        if c and trip:
            M_parts += ["{} {}".format(c, tag)]
        elif c:
            M_parts += [str(c)]
        else:
            trip = True
    if len(M_parts) == 0:
        M = "0"
    elif len(M_parts) == 1:
        M = M_parts[0]
    else:
        M = "(" + (" + ".join(M_parts)) + ")"

    return "{} \N{MULTIPLICATION SIGN} {}".format(N, M)


def variable_type_name(var):
    # type: (Union[Orange.data.Variable, Type[Orange.data.Variable]]) -> str
    """
    Return the variable type name for use in UI

    Parameters
    ----------
    var: Orange.data.Variable

    Returns
    -------
    name: str
        Display name for the variable type.
    """
    if isinstance(var, type):
        assert issubclass(var, Orange.data.Variable)
        vartype = var
    else:
        assert isinstance(var, Orange.data.Variable)
        vartype = type(var)

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


def summarize_variable(var):
    # type: (Orange.data.Variable) -> Tuple[str, str]
    """
    Summarize variable as a description pair

    Parameters
    ----------
    var : Orange.data.Variable

    Returns
    -------
    description : Tuple[str, str]
        (name, description) pair where name is the variable's name and
        description is a type appropriate description.

    Examples
    --------
    >>> iris = Orange.data.Table("iris")
    >>> summarize_variable(iris.domain[0])
    ('sepal length', 'numeric')
    >>> summarize_variable(iris.domain.class_var)
    ('iris', 'categorical with 3 values')
    """
    title = var.name
    if var.is_discrete:
        var = var  # type: Orange.data.DiscreteVariable
        description = "{} with {} value{s}".format(
            variable_type_name(var), len(var.values),
            s="s" if len(var.values) != 1 else "")
    else:
        description = variable_type_name(var)
    return title, description


#: Preferred order of variable types in UI display
PREFERRED_VARIABLE_ORDER = [
    Orange.data.DiscreteVariable,
    Orange.data.ContinuousVariable,
    Orange.data.TimeVariable,
    Orange.data.StringVariable,
]


def summarize_variable_types(variables, sep=", ", item_fmt="{name}: {count}"):
    # type: (List[Orange.data.Variable]) -> str
    """
    Return inline summary of types in `variables` list

    Parameters
    ----------
    variables : List[Orange.data.Variable]
        A list of variables to summarize.
    sep : str
        A subcomponent separator string (default: `', '`)
    item_fmt : str
        A subcomponent format string.

    Returns
    -------
    summary : str

    Examples
    --------
    >>> hd = Orange.data.Table("heart_disease")
    >>> summarize_variable_types(hd.domain.attributes)
    'categorical: 7, numeric: 6'
    >>> summarize_variable_types(hd.domain.attributes, item_fmt="{name}")
    'categorical, numeric'
    >>> iris = Orange.data.Table("iris")
    >>> summarize_variable_types(iris.domain.attributes, item_fmt="{name}")
    'numeric'
    >>> summarize_variable_types([])
    ''
    """
    counts = Counter(type(var) for var in variables)

    def index(type):
        try:
            return PREFERRED_VARIABLE_ORDER.index(type)
        except ValueError:
            return sys.maxsize

    counts = sorted(counts.items(), key=lambda item: index(item[0]))
    return sep.join(item_fmt.format(
                        name=variable_type_name(vartype), count=count)
                    for vartype, count in counts)


def summarize_variables(variables, fold_single=False):
    # type: (List[Orange.data.Variable], bool) -> str
    """
    Return an inline summary of variables in `variables` list.

    Parameters
    ----------
    variables : List[Orange.data.Variables]
        Variables to summarize.
    fold_single : bool
        If the `variables` list contains a single item report a more
        detailed type specific summary (using :func:`summarize_variable`).

    Returns
    -------
    summary : str

    Examples
    --------
    >>> hd = Orange.data.Table("heart_disease")
    >>> summarize_variables(hd.domain.attributes)
    '13 (categorical: 7, numeric: 6)'
    >>> summarize_variables(hd.domain.class_vars)
    '1 (categorical)'
    >>> summarize_variables(hd.domain.class_vars, fold_single=True)
    'categorical with 2 values'
    """
    size = len(variables)
    if size == 0:
        body = "none"
    elif size == 1 and fold_single:
        body = summarize_variable(variables[0])[1]
    elif len(set(type(var) for var in variables)) == 1:
        # homogeneous type
        body = "{} ({})".format(size, variable_type_name(variables[0]))
    else:
        # mixed type
        body = "{} ({})".format(size, summarize_variable_types(variables))
    return body


def categorize_variables(variables, fold_single=False):
    """

    Parameters
    ----------
    variables
    fold_single

    Returns
    -------

    Examples
    --------
    >>> hd = Orange.data.Table("heart_disease")
    >>> categorize_variables(hd.domain.attributes)
    'mixed categorical, numeric'
    >>> categorize_variables(hd.domain.class_vars)
    'categorical'
    >>> categorize_variables(hd.domain.class_vars, fold_single=True)
    'categorical with 2 values'
    """
    size = len(variables)
    if size == 0:
        body = "none"
    elif size == 1 and fold_single:
        body = summarize_variable(variables[0])[1]
    elif len(set(type(var) for var in variables)) == 1:
        # homogeneous type
        body = variable_type_name(variables[0])
    else:
        # mixed type
        body = "mixed " + summarize_variable_types(variables, item_fmt="{name}")
    return body


def summarize_domain_long(domain, categorize_types=True,
                          categorize_roles=True, drop_empty=False):
    # type: (Orange.data.Domain, bool, bool, bool) -> List[Tuple[str, str]]
    """
    Return a 'description list' (name, body) pairs describing the domain.

    Parameters
    ----------
    domain : Orange.data.Domain
        Domain to describe.
    categorize_types : bool
        Summarize constituent variable types.
    categorize_roles : bool
        Summarize variables by domain role (i.e. report `attributes`,
        `class_vars` and  `metas` separately).
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
    >>> summarize_domain_long(iris.domain)  # doctest: +ELLIPSIS
    [('features', '4 (numeric)'), ('target', 'categorical with 3 values'), ...
    >>> hd = Orange.data.Table("heart_disease")
    >>> summarize_domain_long(hd.domain)    # doctest: +ELLIPSIS
    [('features', '13 (categorical: 7, numeric: 6)'), ...
    >>> summarize_domain_long(hd.domain, categorize_types=False) # doctest: +ELLIPSIS
    [('features', '13'), ('target', 'categorical ...
    >>> summarize_domain_long(hd.domain, categorize_roles=False)
    [('columns', '14 (categorical: 8, numeric: 6)')]
    """

    def describe_part(name, variables, fold_single=False):
        if categorize_types or (fold_single and len(variables) == 1):
            body = summarize_variables(variables, fold_single=fold_single)
        else:
            body = "none" if not variables else str(len(variables))
        if fold_single and len(variables) > 1:
            name = name + "s"
        return name, body

    if not categorize_roles:
        return [describe_part("column", (domain.attributes + domain.class_vars +
                                         domain.metas),
                              fold_single=True)]
    parts = []
    if domain.attributes or not drop_empty:
        parts.append(describe_part("features", domain.attributes))
    if domain.class_vars or not drop_empty:
        parts.append(describe_part("target", domain.class_vars, fold_single=True))
    if domain.metas or not drop_empty:
        parts.append(describe_part("metas", domain.metas))
    return parts


def summarize_table(table, categorize_types=True, drop_empty=False,
                    missing=True):
    # type: (Orange.data.Table, bool, bool, bool) -> List[Tuple[str, str]]
    """
    Summarize a Orange.data.Table.

    Parameters
    ----------
    table: Orange.data.Table
        Table to summarize
    categorize_types: bool
        Passed to summarize_domain_long
    drop_empty: bool
        Omit X/Y/M parts from the summary if they are empty.
    missing: bool
        Include a basic missing values statistics in the summary.

    Returns
    -------
    summary : List[Tuple[str, str]]
        A description list

    Examples
    --------
    >>> hd = Orange.data.Table("heart_disease")
    >>> summarize_table(hd)  # doctest: +ELLIPSIS
    [('rows', '303'), ('features', '13 (categorical: ...
    """
    # TODO: On request expand the summary with '% missing', sparsity, ...
    # Features: 4 (2% missing over 2 columns) | (50% sparsity, sparse - 50%)
    # Features: 4 (mixed C, N; 20 % missing)
    # Target: discrete with 3 values (3% missing)
    # Shape: 2 rows x (1 + 3) columns

    from Orange.widgets.utils import datacaching
    from Orange.statistics import basic_stats
    from Orange.data import Table

    def describe_part(variables, stats, storage, fold_single=False):
        parts = []
        if categorize_types or (fold_single and len(variables) == 1):
            # body = summarize_variables(variables, fold_single=fold_single)
            body = categorize_variables(variables, fold_single=fold_single)
        else:
            body = "none" if not variables else str(len(variables))

        parts.append(body)

        if stats is not None and storage == Table.DENSE:
            nans = sum(stat.nans for stat in stats)
            non_nans = sum(stat.non_nans for stat in stats)
            total = nans + non_nans
            if nans:
                parts.append("{:.1f}% missing".format(100 * nans / total))

        return "; ".join(parts)

    domain = table.domain
    if missing and not isinstance(table, Orange.data.sql.table.SqlTable):
        bstats = datacaching.getCached(
            table, basic_stats.DomainBasicStats, (table, True)
        )
        X_stats = bstats.stats[:len(domain.attributes)]
        Y_stats = bstats.stats[len(domain.attributes): len(domain.variables)]
        M_stats = bstats.stats[len(domain.variables):]
    else:
        X_stats = Y_stats = M_stats = None

    parts = []
    if domain.attributes or not drop_empty:
        parts.append(
            ("features", describe_part(domain.attributes, X_stats,
                                       table.X_density()))
        )
    if domain.class_vars or not drop_empty:
        parts.append(
            ("target{}".format("s" if len(domain.class_vars) != 1 else ""),
             describe_part(domain.class_vars, Y_stats,
                           table.Y_density(), fold_single=True))
        )
    if domain.metas or not drop_empty:
        parts.append(
            ("metas", describe_part(domain.metas, M_stats, table.Y_density()))
        )

    if isinstance(table, Orange.data.sql.table.SqlTable):
        rows_count = "~" + str(table.approx_len())
    else:
        rows_count = str(len(table))
    return [("rows", rows_count)] + parts
