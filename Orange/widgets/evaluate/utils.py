from typing import Tuple

import numpy

import Orange.data
from Orange.evaluation import Results
from Orange.widgets.utils import summary

def check_results_adequacy(results, error_group, check_nan=True):
    error_group.add_message("invalid_results")
    error_group.invalid_results.clear()

    def anynan(a):
        return numpy.any(numpy.isnan(a))

    if results is None:
        return None
    if results.data is None:
        error_group.invalid_results(
            "Results do not include information on test data")
    elif not results.data.domain.has_discrete_class:
        error_group.invalid_results(
            "Discrete outcome variable is required")
    elif check_nan and (anynan(results.actual) or
                        anynan(results.predicted) or
                        (results.probabilities is not None and
                         anynan(results.probabilities))):
        error_group.invalid_results(
            "Results contains invalid values")
    else:
        return results


def summarize_results(results, ):
    # type: (Results) -> str
    nmodels, ninst = results.predicted.shape
    nfolds = len(results.folds) if results.folds is not None else None
    domain = results.domain  # type: Orange.data.Domain

    if domain.has_discrete_class:
        rtype = "{number} classifier{s}"
    elif domain.has_continuous_class:
        rtype = "{number} regression model{s}"
    else:
        rtype = "{number} model{s}"
    part_nmodels = summary.plural(rtype, nmodels)
    part_ninst = summary.plural("{number} instance{s}", ninst)
    short = "{} tested on {}".format(part_nmodels, part_ninst)
    if nfolds is not None and nfolds > 1:
        short += summary.plural(" over {number} fold{s}", nfolds)
    return short
