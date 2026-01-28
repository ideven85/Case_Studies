"""
Question 6: Convert the flatten function into a generator

You may test this functions by running `pytest q6_flatten.py` (no quotes)
in the terminal.
"""


def flatten(x):
    if not x:
        yield []
        return
    for el in x:
        if isinstance(el, list):
            yield from flatten(el)  # When calling a generator function use yield from
        else:
            yield el


def test_my_flatten():
    import types

    x = [1, [2, [3, [4]]]]
    y = [[[[[[[1, 2, 3]]]]], 4], 5]
    z = [[[[[[[[[[[1]]]]]]]]]]]

    assert isinstance(
        flatten(x), types.GeneratorType
    ), f"my_zip should produce a generator!"
    assert list(flatten(x)) == [1, 2, 3, 4]
    assert list(flatten(y)) == [1, 2, 3, 4, 5]
    assert list(flatten(z)) == [1]
