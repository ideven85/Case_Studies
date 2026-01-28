import functools
import itertools
import operator
import numpy as np

from lib import cv, lin_reg_classify, get_num_correct, ridge_analytic_with_bias
from plotting import plot_data, plot_nonlin_sep
from test_suite import run_test_suite


def mul(seq):
    return functools.reduce(operator.mul, seq, 1)


def make_polynomial_feature_fun(order):
    # raw_features is d by n
    # return is k by n where k = sum_{i = 0}^order  multichoose(d, i)
    def f(raw_features):
        d, n = raw_features.shape
        result = []  # list of column vectors
        for j in range(n):
            features = []
            for o in range(order + 1):
                indexTuples = itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)

    return f


### Data Sets
# Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
def super_simple_separable_through_origin():
    X = np.array([[2, 3, 9, 12], [5, 1, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y


def super_simple_separable():
    X = np.array([[2, 3, 9, 12], [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y


def xor():
    X = np.array([[1, 2, 1, 2], [1, 2, 2, 1]])
    y = np.array([[1, 1, -1, -1]])
    return X, y


def xor_more():
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3], [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, -1, -1, 1, 1, -1, -1]])
    return X, y


### Test cases


def test_one_hot(fn):
    """
    Tests one_hot() function
    """
    cases = [
        (3, 5, np.array([[0.0], [0.0], [1.0], [0.0], [0.0]])),
        (4, 7, np.array([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])),
    ]
    run_test_suite(cases, fn)


def test_linear_classifier_with_features(
    dataFun, learner, feature_fun, draw=True, refresh=True
):
    raw_data, labels = dataFun()
    data = feature_fun(raw_data) if feature_fun else raw_data

    draw_sep = None
    if draw:
        ax = plot_data(raw_data, labels)

        def draw_sep(params):
            th, th0 = params
            plot_nonlin_sep(
                lambda x1, x2: int(
                    lin_reg_classify(feature_fun(cv([x1, x2])), th, th0)
                ),
                ax=ax,
            )
            plot_data(raw_data, labels, ax)

    th, th0 = learner(data, labels, lam=0.001)
    if draw_sep:
        draw_sep((th, th0))
    print(
        "Final accuracy:",
        int(get_num_correct(data, labels, th, th0)),
        "/",
        data.shape[1],
    )
    print("Params: th =", np.transpose(th), "th0 =", th0)


def test_with_features(dataFun, order=2, draw=True):
    """
    Use this function to train separators with polynomial order features and visualize.

    dataFun: function that returns the dataset as a tuple (data, labels)
    order (int): polynomial order for featurization
    draw (bool): whether to draw the data and separator
    """
    test_linear_classifier_with_features(
        dataFun,  # data
        ridge_analytic_with_bias,  # learner
        make_polynomial_feature_fun(order),  # feature maker
        draw=draw,
    )


def test_compute_dist_matrix(fn):
    """
    Tests for compute_dist_matrix()
    """
    ans = np.array(
        [
            [0.0, 5.19615242, 10.39230485, 15.58845727],
            [5.19615242, 0.0, 5.19615242, 10.39230485],
            [10.39230485, 5.19615242, 0.0, 5.19615242],
            [15.58845727, 10.39230485, 5.19615242, 0.0],
        ]
    )
    cases = [(np.arange(12).reshape(4, 3), ans)]
    run_test_suite(cases, fn)


def test_R_minus_D(fn):
    """
    Tests for get_R_minus_D()
    """
    ans = np.array([[[1.0, 0.0], [-1.0, -2.0]], [[2.0, 1.0], [0.0, -1.0]]])
    cases = [(np.array([1.0, 2.0]), np.array([[0.0, 1.0], [2.0, 3.0]]), ans)]
    run_test_suite(cases, fn)


def test_rdf(fn):
    """
    Tests for get_rdf()
    """
    cases = [
        (np.array([1.0, 2.0]), np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([0.5, 0.5]))
    ]
    run_test_suite(cases, fn)
