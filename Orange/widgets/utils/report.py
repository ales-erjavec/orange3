import collections

# TODO: Actually move report here.
from Orange.canvas.report import report as _report


def describe_data(data):
    return list(_report.describe_data_brief(data).items())


def describe_data_brief(data):
    return list(_report.describe_data_brief(data).items())


def describe_domain(domain):
    return list(_report.describe_domain(domain).items())


def describe_domain_brief(domain):
    return list(_report.describe_domain_brief(domain).items())


describe_data = _report.describe_data
describe_data_brief = _report.describe_data_brief
describe_domain = _report.describe_domain
describe_domain_brief = _report.describe_domain_brief

plural = _report.plural
plural_w = _report.plural_w
