"""
Question 5: Implement the my_zip function as a generator

You may test this functions by running `pytest q5_zip.py` (no quotes)
in the terminal.
"""


def my_zip(x, y):
    # if not x or not y:
    #     return None
    x = list(x)
    y = list(y)
    # print("bite me:",x,y)
    for i in range(min(len(x), len(y))):
        yield x[i], y[i]
    # print(x,y)


if __name__ == "__main__":
    import types

    # erase zip
    global zip
    f = zip
    zip = None

    x = range(0, 4)
    y = "ABCD"
    z = [2, 4, 6]
    x1 = f(x, y)

    x1, x2 = f(x, y), my_zip(x, y)
    # print(list(x1))
    # print(list(list(list(my_zip(x,y)))))

    assert isinstance(
        x2, types.GeneratorType
    ), f"my_zip first should produce a generator!"
    # print(list(x2))
    assert list(x1) == list(x2)
    # check that the generator is exhausted
    assert list(x1) == list(x2)
    # print(list(my_zip(y,z)))
    assert list(f(y, z)) == list(my_zip(y, z))
    assert list(f(f(y, z), x)) == list(my_zip(my_zip(y, z), x))
    print("Done you dude")
