"""
Question 4: Convert the my_enumerate function into a generator!

You may test this functions by running `pytest q4_enum.py` (no quotes)
in the terminal.
"""


def my_enumerate(x):
    """
    Create a generator that returns the same output as enumerate
    without using enumerate!
    """

    i = 0
    for i in range(len(x)):
        yield i, x[i]


def test_my_enumerate():
    import types

    # erase enumerate
    global enumerate
    f = enumerate
    enumerate = None

    x = range(5, 10)
    x1, x2 = f(x), my_enumerate(x)
    assert isinstance(
        x2, types.GeneratorType
    ), f"My enumerate should produce a generator!"
    assert list(x1) == list(x2)
    # check that the generator is exhausted
    assert list(x1) == list(x2)

    # check that it is efficient - not storing all results
    # before yielding
    x = range(5, 10_000_000_000)
    x1, x2 = f(x), my_enumerate(x)
    res1, res2 = [], []
    for _ in range(5):
        res1.append(next(x1))
        res2.append(next(x2))
    assert res1 == res2
