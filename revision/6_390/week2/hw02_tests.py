"""
Fall 2022 Homework 2 Test Suite

To use:
1. (For staff) Make sure this file is in the homework/hw02/_STATIC_/ folder

2. In Colab notebook, have a cell:

    !rm hw02_tests.py
    !wget --no-check-certificate https://introml.mit.edu/_static/fall22/homework/hw02/hw02_tests.py
    from hw02_tests.py import *

3. Under each function to test, create a new cell with the function corresponding
   to the function you want to test, and pass in your implementation
   as an argument to that function. For example, to test your rv() function:

       TestRowVector(rv)

Last update: 2022-09-09
"""

import numpy as np
from sklearn.model_selection import ShuffleSplit

### Predefined functions.
### You don't need to know this in detail for the homework, but feel free to peruse it.


def lin_reg(x, th, th0):
    return np.dot(th.T, x) + th0


def mse(x, y, th, th0):
    return np.mean((y - lin_reg(x, th, th0)) ** 2, axis=1, keepdims=True)


def ridge_analytic(X, Y, lam):
    if lam == 0:
        # If you'd like to use this condition (lam=0) in your Colab,
        # copy this function into a cell of your Colab, remove the
        # following raise line and run it with your implementation
        # of lin_reg_analytic() as our just returns None instead
        # of the analytical solution.
        raise ValueError("See implementation.")
        return lin_reg_analytic(X, Y), 0
    d, n = X.shape
    xm = np.mean(X, axis=1, keepdims=True)  # d x 1
    ym = np.mean(Y)  # 1 x 1
    Z = X - xm  # d x n
    T = Y - ym  # 1 x n
    th = np.dot(np.linalg.inv(np.dot(Z, Z.T) + lam * np.identity(d)), np.dot(Z, T.T))
    # th_0 account for the centering
    th_0 = ym - np.dot(th.T, xm)  # 1 x 1
    return th, th_0


def make_splits(X, Y, n_splits):
    rs = ShuffleSplit(n_splits=n_splits, random_state=0, test_size=None)
    ret = list(rs.split(X=X.T, y=Y.T))
    split_datasets = []
    for train_idx, test_idx in ret:
        X_train = X[:, train_idx]
        Y_train = Y[:, train_idx]
        X_test = X[:, test_idx]
        Y_test = Y[:, test_idx]
        split_datasets.append((X_train, Y_train, X_test, Y_test))
    return split_datasets


### Testing code


def TestCase(inputs, expected, fn=None):
    """
    Runs a single test case.
    """
    result = inputs[0]
    if fn:
        result = fn(*inputs)

    shape_error = False
    try:
        if np.allclose(result, expected, atol=1e-3):
            if isinstance(expected, np.ndarray):
                assert np.array_equal(expected.shape, result.shape)
            print("Test case passed!")
            print("----------------------------")
            return True
    except:
        shape_error = True

    # display single inputs in a nicer way
    if len(inputs) == 1:
        inputs = inputs[0]

    if not fn:
        # don't show function inputs if not testing a function
        print("FAILED")
    else:
        print("FAILED on inputs:\n", inputs)

    print("Expected:\n", expected)
    print("Got:\n", result)

    if shape_error:
        print("Hint: Check the shapes of your output.")
    print("----------------------------")
    return False


def RunTestSuite(cases, fn=None, seed=None):
    """
    Cases is a list of lists (or other 2D iterable) where:
        * each row is a list/tuple/iterable, representing a test case
        * last element of each row is the expected answer
        * all elements before are the inputs to the function to check, in order
        * if the last element (expected answer) is a np array object, the
          student's answer will also be checked for getting shape correct.

    fn is the function to check. if fn=None, only the first element of Cases
       will be checked to see if it is np.allclose() to the last element.
    """
    passed = 0
    failed = 0
    for entries in cases:
        if seed is not None:
            np.random.seed(seed)
        expected = entries[-1]
        input = entries[:-1]
        if TestCase(input, expected, fn):
            passed += 1
        else:
            failed += 1
    if passed == len(cases):
        print("\nAll tests passed!")
    else:
        print("\nRan %d tests: %d passed, %d failed." % (len(cases), passed, failed))


def lin_reg_analytic(X, Y):
    th = None  # Insert your code here
    return th


def TestLinRegAnalytic(fn):
    """
    Test function for the analytic solution for linear regression

    First complete lin_reg_analytic() then pass that function to this one.
    """
    X = np.array([[1, 2, 3, 4], [1, 1, 1, 1]])
    Y = np.array([[1, 2.2, 2.8, 4.1]])
    ans = np.array([[0.99], [0.05]])
    cases = [(X, Y, ans)]
    RunTestSuite(cases, fn)


def TestCrossValidate(fn):
    """
    Test function for cross_validate()
    """
    X = np.array(
        [
            [4, 6, 8, 2, 9, 10, 11, 17],
            [1, 1, 6, 0, 5, 8, 7, 9],
            [2, 2, 2, 6, 7, 4, 9, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    )
    Y = np.array([[1, 3, 3, 4, 7, 6, 7, 7]])
    n_splits = 3
    lam = 0.1
    ans = 2.448471890218119
    cases = [(X, Y, n_splits, lam, ridge_analytic, mse, ans)]
    RunTestSuite(cases, fn)


def TestCrossValidateErrors(errors):
    """
    Test function to get cross_validate errors with 3 splits

    First run the following:

        X = np.array([[4, 6, 8, 2, 9, 10, 11, 17],
                      [1, 1, 6, 0, 5, 8, 7, 9],
                      [2, 2, 2, 6, 7, 4, 9, 8],
                      [1, 2, 3, 4, 5, 6, 7, 8]])
        Y = np.array([[1, 3, 3, 4, 7, 6, 7, 7]])
        lams = [0.2, 0.3]
        errors = None  # Replace with your code

    Then pass `errors` to this function
    """
    try:
        assert np.allclose(errors, [2.342579, 2.250919])
        print("Test case passed!")
        print("----------------------------")
        print("All tests passed!")
    except:
        print("FAILED")
