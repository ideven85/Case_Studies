import numpy as np


def run_test_case(inputs, expected, fn=None, check_type=True):
    """
    Runs a single test case.

    We make the assumption that we're testing for some numerical output,
    i.e. expected is numerical, either a scalar or numpy array, and not say a string.
    """
    result = inputs[0]
    if fn:
        result = fn(*inputs)

    success = False
    hint = None

    if check_type and not isinstance(result, type(expected)):
        hint = "Check the type of your output."
    elif isinstance(expected, np.ndarray) and not np.array_equal(
        expected.shape, result.shape
    ):
        hint = "Check the shape of your output."
    elif np.allclose(result, expected, atol=1e-3):
        success = True

    if success:
        print("Test case passed!")
    else:
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

        # Additional hint if desired
        if hint is not None:
            print("Hint:", hint)

    print("----------------------------")
    return success


def run_test_suite(cases, fn=None, seed=None):
    """
    cases is a list of lists (or other 2D iterable) where:
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
    for test_case in cases:
        if seed is not None:
            np.random.seed(seed)

        inputs = test_case[:-1]
        expected = test_case[-1]

        did_succeed = run_test_case(inputs, expected, fn)

        if did_succeed:
            passed += 1
        else:
            failed += 1

    if passed == len(cases):
        print("\nAll tests passed!")
    else:
        print("\nRan %d tests: %d passed, %d failed." % (len(cases), passed, failed))
