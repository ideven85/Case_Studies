import numpy as np


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
    except ValueError as e:
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


def TestCreateArray(arr):
    """
    Test function for creating 2x3 array
    """
    try:
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr.shape, [2, 3])
        print("Test case passed!")
        print("----------------------------")
        print("All tests passed!")
    except:
        print(
            "FAILED: Either your input is not an numpy array or its shape is incorrect."
        )


def TestTranspose(fn):
    """
    Test function for transpose `tp` function
    """
    cases = [
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 4], [2, 5], [3, 6]])),
        (np.array([[1, 2], [3, 4], [5, 6]]), np.array([[1, 3, 5], [2, 4, 6]])),
    ]
    RunTestSuite(cases, fn)


def TestRowVector(fn):
    """
    Test function for rv()
    """
    cases = [([1, 2], np.array([[1, 2]])), ([1, 3, 5], np.array([[1, 3, 5]]))]
    RunTestSuite(cases, fn)


def TestColVector(fn):
    """
    Test function for cv()
    """
    cases = [([1, 2], np.array([[1], [2]])), ([1, 3, 5], np.array([[1], [3], [5]]))]
    RunTestSuite(cases, fn)


def TestLength(fn):
    """
    Test function for getting col vector's length `length()`
    """
    # Ensure that fn returns a scalar.
    case1 = np.array([[1], [2]])
    output = fn(case1)
    if np.shape(output) != np.shape(1.0):
        print(
            "FAILED: The output from your function to compute length must be a scalar."
        )
        return
    cases = [
        (np.array([[1], [2]]), 2.23606797749979),
        (np.array([[1], [3], [5]]), 5.916079783099616),
    ]
    RunTestSuite(cases, fn)


def TestNormalize(fn):
    """
    Test function for normalize()
    """
    cases = [
        (np.array([[1], [2]]), np.array([[0.4472135954999579], [0.8944271909999159]])),
        (
            np.array([[1], [3], [5]]),
            np.array([[0.1690308509457033], [0.50709255283711], [0.8451542547285166]]),
        ),
    ]
    RunTestSuite(cases, fn)


def TestIndexing(fn):
    """
    Test function for getting last column `index_final_col()`
    """
    cases = [
        (np.array([[1, 2], [3, 4]]), np.array([[2], [4]])),
        (np.array([[1, 2, 7], [9, 3, 4]]), np.array([[7], [4]])),
    ]
    RunTestSuite(cases, fn)


def TestInverse(fn):
    """
    Test function for getting matrix inverse
    """
    cases = [
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])),
        (np.array([[1, 2], [3, 4]]), np.array([[-2, 1], [1.5, -0.5]])),
        (
            np.array([[1, 2, 3], [3, 4, 5], [2, 6, 7]]),
            np.array(
                [
                    [-1 / 3, 2 / 3, -1 / 3],
                    [-11 / 6, 1 / 6, 2 / 3],
                    [5 / 3, -1 / 3, -1 / 3],
                ]
            ),
        ),
    ]
    RunTestSuite(cases, fn)


def TestData(data):
    """
    Test function for weight/height data
    """
    cases = [(data, [[150, 130, 120], [5.8, 5.5, 5.3]])]
    RunTestSuite(cases, fn=None)


def TestTransform(fn):
    """
    Test function for transforming weight/height data `transform()`
    """
    cases = [
        (
            np.array([[150, 130, 120], [5.8, 5.5, 5.3]]),
            np.array([[155.8, 135.5, 125.3]]),
        )
    ]
    RunTestSuite(cases, fn)


def TestLinPredict(fn):
    """
    Test function for lin_reg_predict with one data point
    """
    cases = [
        (np.array([[1.5]]), np.array([[2.0]]), np.array([[-1.0]]), np.array([[2.0]])),
        (
            np.array([[1.5], [-2]]),
            np.array([[2.0], [-3.0]]),
            np.array([[-2.0]]),
            np.array([[7.0]]),
        ),
    ]
    RunTestSuite(cases, fn)


def TestLinPredict_N(fn):
    """
    Test function for lin_reg_predict with more than one data point
    """
    cases = [
        (
            np.array([[1.5, 3.0], [-2.0, 1.0]]),
            np.array([[2.0], [-3.0]]),
            np.array([[-2.0]]),
            np.array([[7.0, 1.0]]),
        )
    ]
    RunTestSuite(cases, fn)


def TestMSE(fn):
    """
    Test function for calculating MSE with one data point
    """
    cases = [
        (np.array([[1.5, 3.0]]), np.array([[2.0, 2.0]]), np.array([[1.25 / 2]])),
        (
            np.array([[-2.0, 1.0, 3.0]]),
            np.array([[2.0, 1.0, 2.0]]),
            np.array([[17 / 3]]),
        ),
    ]
    RunTestSuite(cases, fn)


def TestMSE_N(fn):
    """
    Test function for calculating MSE with more than one data point
    """
    cases = [
        (np.array([[1.5, 3.0]]), np.array([[2.0, 2.0]]), np.array([[1.25 / 2]])),
        (
            np.array([[1.5, 3.0, 4], [-2.0, 1.0, 3.0]]),
            np.array([[2.0, 2.0, 0.0], [2.0, 1.0, 2.0]]),
            np.array([[17.25 / 3], [17 / 3]]),
        ),
    ]
    RunTestSuite(cases, fn)


def TestLinRegError(fn):
    """
    Test function for lin_reg_err()
    """
    cases = [
        (
            np.array([[1.5, 3.0]]),
            np.array([[2.1, 4.5]]),
            np.array([[2.0]]),
            np.array([[-1.0]]),
            np.array([[0.13]]),
        ),
        (
            np.array([[1.5, 3.0], [-2.0, 1.0]]),
            np.array([[1.0, 3.0]]),
            np.array([[2.0], [-3.0]]),
            np.array([[-2.0]]),
            np.array([[20.0]]),
        ),
    ]
    RunTestSuite(cases, fn)


### Ignore test cases below - for old questions, no longer relevant


def TestSignedDist(fn):
    cases = [
        (np.array([[0], [0]]), np.array([[3], [4]]), 5, np.array([[1.0]])),
        (np.array([[2], [3]]), np.array([[3], [4]]), 5, np.array([[4.6]])),
        (np.array([[2], [3]]), np.array([[3], [4]]), -5, np.array([[2.6]])),
        (np.array([[2], [3]]), np.array([[-3], [-4]]), 5, np.array([[-2.6]])),
    ]
    RunTestSuite(cases, fn)


def TestPositive(fn):
    cases = [
        (np.array([[0], [0]]), np.array([[3], [4]]), 5, np.array([[1.0]])),
        (np.array([[2], [3]]), np.array([[3], [4]]), 5, np.array([[1.0]])),
        (np.array([[2], [3]]), np.array([[3], [4]]), -5, np.array([[1.0]])),
        (np.array([[2], [3]]), np.array([[-3], [-4]]), 5, np.array([[-1.0]])),
    ]
    RunTestSuite(cases, fn)


def Test26a(input):
    cases = [(input, np.array([[1, 1, 1, -1, -1]]))]
    RunTestSuite(cases, fn=None)


def Test26b(input):
    cases = [(input, np.array([[False, False, True, False, False]]))]
    RunTestSuite(cases, fn=None)


def TestScore(fn):
    data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
    labels = np.array([[-1, -1, +1, +1, +1]])
    cases = [
        (data, labels, np.array([[1], [1]]), -2, 1),
        (data, labels, np.array([[-1], [-1]]), 2, 4),
        (data, labels, np.array([[0], [1]]), 1, 1),
    ]
    RunTestSuite(cases, fn)


def TestBestSeparator(fn):
    data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
    labels = np.array([[-1, -1, +1, +1, +1]])
    thetas = np.array(
        [
            [
                0.98645534,
                -0.02061321,
                -0.30421124,
                -0.62960452,
                0.61617711,
                0.17344772,
                -0.21804797,
                0.26093651,
                0.47179699,
                0.32548657,
            ],
            [
                0.87953335,
                0.39605039,
                -0.1105264,
                0.71212565,
                -0.39195678,
                0.00999743,
                -0.88220145,
                -0.73546501,
                -0.7769778,
                -0.83807759,
            ],
        ]
    )
    theta_0s = np.array(
        [
            [0.65043158],
            [0.61626967],
            [0.84632592],
            [-0.43047804],
            [-0.91768579],
            [-0.3214327],
            [0.0682113],
            [-0.20678004],
            [-0.33963784],
            [0.74308104],
        ]
    ).T
    output = fn(data, labels, thetas, theta_0s)
    expected = (np.array([[0.32548657], [-0.83807759]]), np.array([[0.74308104]]))

    try:
        assert isinstance(output, tuple)
        assert len(output) == len(expected)
        assert np.array_equal(output[0], expected[0])
        assert np.array_equal(output[1], expected[1])
        print("Test case passed!")
        print("----------------------------")
        print("\nAll tests passed!")
    except:
        print("FAILED")
        print("Expected:\n", expected)
        print("Got:\n", output)
        print("----------------------------")
