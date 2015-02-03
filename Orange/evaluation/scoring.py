from functools import partial
from collections import namedtuple

import numpy
import sklearn.metrics

import Orange.data

from .testing import Results

__all__ = ["AUC", "AUC_binary", "AUC_one_vs_rest", "AUC_pairwise",
           "CA", "Precision", "Recall", "F1", "MSE", "RMSE", "R2",
           "Result_summary"]


def is_discrete(var):
    return isinstance(var, Orange.data.DiscreteVariable)


def is_continuous(var):
    return isinstance(var, Orange.data.ContinuousVariable)


def is_binary(var):
    return is_discrete(var) and len(var.values) == 2


# Use (abuse) class to define a namespace for utility functions.
class utils:
    """General scoring utilities."""

    #: Discrete results for a single learner/model
    #: (optional list of slice, (N,) ndarray, (N,) ndarray, (N, K) ndarray
    disc_results = namedtuple(
        "disc_results", ["folds", "true", "predicted", "probs"]
    )
    #: Continupus results for a single learner/model
    #: (optional list of slice, (N,) ndarray, (N,) ndarray)
    cont_results = namedtuple(
        "cont_results", ["folds", "true", "predicted"]
    )

    @staticmethod
    def results(results):
        """Return a sequence of utils.[disc|cont]_results from a `Results`."""
        folds = results.folds
        if results.probabilities is not None:
            return [utils.disc_results(folds, results.actual, pred, probs)
                    for pred, probs in
                    zip(results.predicted, results.probabilities)]
        else:
            return [utils.cont_results(folds, results.actual, pred)
                    for pred in results.predicted]

    @staticmethod
    def test_results_is_discrete(results):
        """Does :class:`Results` instance contain discrete target."""
        return results.probabilities is not None

    @staticmethod
    def test_results_nclasses(results):
        """Return number of classes from `Results`"""
        assert results.probabilities is not None
        return results.probabilities.shape[2]

    @staticmethod
    def nclasses(results):
        """Return number of classes from utils.disc_results."""
        assert isinstance(results, utils.disc_results)
        return results.probs.shape[1]

    @staticmethod
    def one_vs_rest(self, results, pos_label):
        """Return a one vs. rest result from a multiclass utils.disc_results.
        """
        assert isinstance(results, utils.disc_results)
        assert 0 <= pos_label < utils.nclasses(results)

        true = results.true == pos_label
        pred = results.predicted == pos_label
        prob_t = results.probs[:, pos_label]
        prob_sum = numpy.sum(results.probs, axis=1)
        probs = numpy.c_[prob_sum - prob_t, prob_t]
        return utils.disc_results(results.folds, true, pred, probs)

    @staticmethod
    def take(results, indices):
        """Return a subset of utils.[disc|cont]_results, indexed by indices.
        """
        if isinstance(results, utils.disc_results):
            return utils.disc_results(
                None, results.true[indices], results.predicted[indices],
                results.probs[indices]
            )
        elif isinstance(results, utils.cont_results):
            return utils.cont_results(
                None, results.true[indices], results.predicted[indices]
            )
        else:
            assert False

    @staticmethod
    def split_class_pairwise(results):
        """Iterate over all pairwise class results subsets.
        """
        assert isinstance(results, utils.disc_results)
        nclass = utils.nclasses(results)
        pairs = ((i, j) for i in range(nclass) for j in range(nclass)
                 if i != j)
        class_ind = {i: numpy.flatnonzero(results.true == i)
                     for i in range(nclass)}
        for i, j in pairs:
            pos, neg = class_ind[i], class_ind[j]
            yield (i, j), utils.take(results, numpy.r_[pos, neg])

    @staticmethod
    def pairwise_tabulate(score, results):
        assert isinstance(results, utils.disc_results)
        nclass = utils.nclasses(results)
        table = numpy.full((nclass, nclass), numpy.nan)
        for (i, j), pair_res in utils.split_class_pairwise(results):
            table[i, j] = score(pair_res)
        return table

    @staticmethod
    def score_folds(score, results):
        """Apply score over folds and return a sequence of results."""
        assert results.folds is not None
        return [score(utils.take(results, fold)) for fold in results.folds]

    @staticmethod
    def slice_len(slice_, container_len):
        """Return the `length` of a `slice` object."""
        start, stop, stride = slice_.indices(container_len)
        assert stride != 0
        span = stop - start
        if span == 0:
            return 0
        elif span % stride == 0:
            return max(span // stride, 0)
        else:
            return max(1 + span // stride, 0)

    @staticmethod
    def fold_len(index, sequence):
        if isinstance(index, slice):
            return utils.slice_len(index, len(sequence))
        else:
            return len(index)

    @staticmethod
    def class_counts(results):
        """Return true class counts for a utils.disc_results."""
        assert isinstance(results, utils.disc_results)
        nclass = utils.nclasses(results)
        return numpy.bincount(results.true.astype(int), minlength=nclass)


class utils_auc:
    """AUC utilities"""
    @staticmethod
    def auc_binary(results, pos_label):
        """Return the AUC score for a binary classification task."""
        assert isinstance(results, utils.disc_results)
        assert 0 <= pos_label < utils.nclasses(results)

        return sklearn.metrics.roc_auc_score(
            results.true == pos_label, results.probs[:, pos_label]
        )

    @staticmethod
    def auc_binary_by_folds(results, pos_label):
        """Return the AUC score averaged over folds."""
        assert isinstance(results, utils.disc_results)
        assert 0 <= pos_label < utils.nclasses(results)

        scores = utils.score_folds(
            partial(utils_auc.auc_binary, pos_label=pos_label),
            results)
        fold_lengths = [utils.slice_len(fold, len(results.true))
                        for fold in results.folds]
        return numpy.average(scores, weights=fold_lengths)

    @staticmethod
    def auc_ovr(results, weights=None):
        """Return the AUC for a multi-class task (averaged one vs rest score).

        :param utils.disc_results results: Input results
        :param weights:
            Class weights to use for one-vs-rest score averaging.
            Default is None, meaning no weighing. If `True`
            then the one-vs-rest scores are weighted by class priors.
            It can also be a (K,) array of weights to use.
        """
        if weights is True:
            weights = utils.class_counts(results)

        scores = [utils_auc.auc_binary(results, pos_label=label)
                  for label in range(utils.nclasses(results))]
        return numpy.average(scores, weights=weights)

    @staticmethod
    def auc_ovr_by_folds(results, weights=None):
        """Return the one-vs-rest auc score averaged over folds.

        .. seealso: utils_auc.auc_ovr
        """
        if weights is True:
            weights = utils.class_counts(results)

        scores = utils.score_folds(
            partial(utils_auc.auc_ovr, weights=weights), results)

        fold_lengths = [utils.slice_len(fold, len(results.true))
                        for fold in results.folds]
        return numpy.average(scores, weights=fold_lengths)

    @staticmethod
    def auc_pairwise_matrix(results):
        """Return a pairwise class one-vs-one AUC matrix.
        """
        nclass = utils.nclasses(results)
        table = numpy.full((nclass, nclass), numpy.nan)

        pairs_iter = utils.split_class_pairwise(results)
        for (i, j), pair_result in pairs_iter:
            score = utils_auc.auc_binary(pair_result, pos_label=j)
            table[i, j] = score
        return table

    @staticmethod
    def auc_pairwise(results, weights=None):
        if weights is True:
            weights = utils.class_counts(results)

        nclass = utils.nclasses(results)

        table = utils_auc.auc_pairwise_matrix(results)
        table = (table + table.T) / 2

        if weights is not None:
            weights = numpy.array(weights)
            if weights.ndim == 1:
                weights = numpy.c_[weights] * numpy.r_[weights]
        else:
            weights = numpy.ones((nclass, nclass))

        assert weights.shape == (nclass, nclass)
        diag_ind = numpy.diag_indices(nclass, ndim=2)
        weights[diag_ind] = 0.0
        table[diag_ind] = 0.0
        return numpy.average(table.ravel(), weights=weights.ravel())

    @staticmethod
    def auc_pairwise_by_folds(results, weights=None):
        if weights is True:
            weights = utils.class_counts(results)

        scores = utils.score_folds(
            partial(utils_auc.auc_pairwise, weights=weights), results)

        fold_lengths = [utils.slice_len(fold, len(results.true))
                        for fold in results.folds]
        return numpy.average(scores, weights=fold_lengths)

    @staticmethod
    def has_folds_for_auc(results):
        if results.folds is None:
            return False
        elif len(results.folds) > 1:
            min_len = min(map(lambda ind: utils.fold_len(ind, results.actual),
                              results.folds),
                          default=0)
            return min_len > 1
        else:
            return False

utils.auc = utils_auc


def AUC_binary(results, pos_label=None):
    """
    Return the AUC score for a binary classification task.

    :param testing.Results results: Test results
    :param int pos_label:
        If results contains a multi-class results then this must be
        the class index of the positive class.
    """
    if not utils.test_results_is_discrete(results):
        raise ValueError("Need discrete target class")

    if pos_label is None:
        if utils.test_results_nclasses(results) == 2:
            pos_label = 1
        else:
            raise ValueError("`pos_label` required for multiclass results")
    if 0 <= pos_label < utils.test_results_nclasses(results):
        raise ValueError("Invalid `pos_label`".format(pos_label))

    if utils.auc.has_folds_for_auc(results):
        score = partial(utils.auc.auc_binary_by_folds, pos_label=pos_label)
    else:
        score = partial(utils.auc.auc_binary, pos_label=pos_label)
    return [score(r) for r in utils.results(results)]


def AUC_one_vs_rest(results):
    """Return the averaged one-vs-rest AUC score.

    :param testing.Results results: Test results.
    """
    if not utils.test_results_is_discrete(results):
        raise ValueError("Need discrete target class")

    if utils.auc.has_folds_for_auc(results):
        score = utils.auc.auc_ovr_by_folds
    else:
        score = utils.auc.auc_ovr
    return [score(r) for r in utils.results(results)]


def AUC_pairwise(results):
    """Return the averaged pairwise class AUC score."""
    # Till & Hand - A Simple Generalization of AUC for Multiple ...
    if not utils.test_results_is_discrete(results):
        raise ValueError("Need discrete target class")

    if utils.auc.has_folds_for_auc(results):
        score = utils.auc.auc_pairwise_by_folds
    else:
        score = utils.auc.auc_pairwise

    return [score(r) for r in utils.results(results)]


def AUC(results):
    if not utils.test_results_is_discrete(results):
        raise ValueError("Need discrete target class")
    # AUC_binary and AUC_pairwise should be the same for binary task
    # but AUC_binary is faster.
    if utils.test_results_nclasses(results) == 2:
        return AUC_binary(results)
    else:
        return AUC_pairwise(results)


def _skl_metric(results, metric):
    return [metric(res.true, res.predicted) for res in utils.results(results)]


def CA(results):
    "Classification accuracy"
    return _skl_metric(results, sklearn.metrics.accuracy_score)


class utils_cm:
    """Utilities for confusion matrix scores."""

    @staticmethod
    def confusion_matrix(results):
        """Return the confusion matrix for utils.disc_results."""
        assert isinstance(results, utils.disc_results)
        labels = numpy.arange(utils.nclasses(results))
        return sklearn.metrics.confusion_matrix(
            results.true, results.predicted, labels)

    @staticmethod
    def precision(confusion_matrix):
        """Return the precision scores (one-vs-rest) for a confusion matrix.
        """
        tp = numpy.diag(confusion_matrix)
        predicted = numpy.sum(confusion_matrix, axis=0)

        with numpy.errstate(divide="ignore", invalid="ignore"):
            # TP / (TP + FP)
            precision = tp / predicted

        precision[~numpy.isfinite(precision)] = numpy.nan
        return precision

    @staticmethod
    def recall(confusion_matrix):
        """Return the recall scores (one-vs-rest) for a confusion matrix."""
        tp = numpy.diag(confusion_matrix)
        positive = numpy.sum(confusion_matrix, axis=1)

        with numpy.errstate(divide="ignore", invalid="ignore"):
            # TP / (TP + FN)
            recall = tp / positive

        recall[~numpy.isfinite(recall)] = numpy.nan
        return recall

    @staticmethod
    def f1(confusion_matrix):
        """Return the F1 scores (one-vs-rest) for a confusion matrix."""
        precision = utils_cm.precision(confusion_matrix)
        recall = utils_cm.recall(confusion_matrix)

        with numpy.errstate(divide="ignore", invalid="ignore"):
            f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def averaged_score(score, results):
        """
        Evaluate cm_score function on results and average the results,
        weighted by class prior.
        """
        cm = utils_cm.confusion_matrix(results)
        scores = score(cm)

        def average(a, w):
            mask = w > 0
            return numpy.average(a[mask], weights=w[mask])

        return average(scores, cm.sum(axis=1))

utils.cm = utils_cm


def confusion_matrices(results):
    """Return a list if confusion matrices, one for each model."""
    return [utils.cm.confusion_matrix(res) for res in utils.results(results)]


def Precision(results):
    "Precision TP / (TP + FP)"
    return [utils.cm.averaged_score(utils.cm.precision, res)
            for res in utils.results(results)]


def Recall(results):
    "Recall TP / (TP + FN)"
    return [utils.cm.averaged_score(utils.cm.recall, res)
            for res in utils.results(results)]


def F1(results):
    "F1 - harmonic mean of precision and recall."
    return [utils.cm.averaged_score(utils.cm.f1, res)
            for res in utils.results(results)]


#==== ++++ =====


class utils_regression:
    @staticmethod
    def mse(results):
        assert isinstance(results, utils.cont_results)
        mean = numpy.mean(results.true)
        return numpy.mean((results.predicted - mean) ** 2)

    @staticmethod
    def mae(results):
        assert isinstance(results, utils.cont_results)
        mean = numpy.mean(results.true)
        return numpy.mean(numpy.abs(results.predicted - mean))

utils.regression = utils_regression


def MSE(results):
    if utils.test_results_is_discrete(results):
        raise ValueError("Need continuous target")
    return [utils.regression.mse(res) for res in utils.results(results)]


def RMSE(results):
    if utils.test_results_is_discrete(results):
        raise ValueError("Need continuous target")
    return [numpy.sqrt(utils.regression.mse(res))
            for res in utils.results(results)]


def MAE(results):
    if utils.test_results_is_discrete(results):
        raise ValueError("Need continuous target")
    return [utils.regression.mae(res) for res in utils.results(results)]


def R2(results):
    if utils.test_results_is_discrete(results):
        raise ValueError("Need continuous target")
    return [sklearn.metrics.r2_score(res.true, res.predicted)
            for res in utils.results(results)]

#==== ++++ =====


class Summary(object):
    def __init__(self, data, __str__=None):
        self.data = data
        self._str = __str__

    def __str__(self):
        if self._str is not None:
            return self._str()
        else:
            return super().__str__()

    def __repr__(self):
        return self.__str__()


def Result_summary(results, digits=3):
    assert digits >= 0
    nmodels = results.predicted.shape[0]
    names = getattr(results, "names", None)
    if names is None:
        names = ["{}".format(i + 1) for i in range(nmodels)]

    if utils.test_results_is_discrete(results):
        scores = [AUC, CA, Precision, Recall]
        header = ["AUC", "CA", "Precision", "Recall"]
    else:
        scores = [RMSE, MAE, R2]
        header = ["RMSE", "MAE", "R2"]

    data = list(zip(*[score(results) for score in scores]))
    cellfmt = "{{:.{}f}}".format(digits)
    table = ([["Learner"] + header] +
             [[names[i]] + [cellfmt.format(sc) for sc in row]
              for i, row in enumerate(data)])
    colwidth = [max(map(len, column)) for column in zip(*table)]
    colalign = ["<"] + ([">"] * len(header))
    fmt = "  ".join("{{:{}{}}}".format(aligh, width)
                   for aligh, width in zip(colalign, colwidth))

    def __str__():
        return "\n".join(map(lambda row: fmt.format(*row), table))

    return Summary(numpy.array(data), __str__)

#==== ++++ =====
