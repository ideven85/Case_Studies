import numpy as np


def rv(value_list):
    """
    Takes a list of numbers and returns a row vector: 1 x n
    """
    return np.array([value_list])


def cv(value_list):
    """
    Takes a list of numbers and returns a column vector:  n x 1
    """
    return np.transpose(rv(value_list))


def lin_reg(x, th, th0):
    """
    Gets predictions from linear regression

    x (np.array): d-by-n
    th (np.array): d-by-m where m is the number of different settings of th
    th0 (np.array): m-by-1
    Returns m-by-n np.array of predictions
    """
    return np.dot(th.T, x) + th0


def lin_reg_classify(x, th, th0):
    preds = np.sign(lin_reg(x, th, th0))
    preds[preds == 0] = -1  # default 0 to label -1
    return preds


def get_num_correct(data, labels, th, th0):
    return np.sum(lin_reg_classify(data, th, th0) == labels)


def ridge_analytic(X, y, lam):
    """
    Determine best theta using the analytic (closed-form) expression for ridge regression.

    X (np.array): d-by-n array of features
    y (np.array): 1-by-n array of targets
    lam (float): lambda i.e. L2-regularizer coefficient

    Returns: d-by-1 array of theta values
    """
    d, _ = X.shape
    return np.linalg.inv(X @ X.T + lam * np.identity(d)) @ X @ y.T


def ridge_analytic_with_bias(X, Y, lam):
    d, n = X.shape
    xm = np.mean(X, axis=1, keepdims=True)  # d x 1
    ym = np.mean(Y)  # 1 x 1
    Z = X - xm  # d x n
    T = Y - ym  # 1 x n
    th = np.dot(np.linalg.inv(np.dot(Z, Z.T) + lam * np.identity(d)), np.dot(Z, T.T))
    # th_0 account for the centering
    th_0 = ym - np.dot(th.T, xm)  # 1 x 1
    return th, th_0
