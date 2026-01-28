"""
6.101 Mines Optional Practice Exercises: Nesting
"""

# no imports allowed!


def pascal1(n):
    """
    Use recursion to create the first n rows of Pascal's Triangle.

    Parameters:
        * n (int) : a positive integer, representing the number of rows in triangle.

    Returns:
        A list of length n, where the element at index i is a list of ints
        representing the ith row of Pascal's triangle.


    >>> pascal(1)
    [[1]]
    >>> pascal(2)
    [[1], [1, 1]]
    >>> pascal(3)
    [[1], [1, 1], [1, 2, 1]]
    >>> pascal(4)
    [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]
    >>> pascal(5)
    [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
    """

    out = [[1]]

    if n == 1:
        return out
    for i in range(n - 1):
        temp = [1]
        for j in range(i):
            temp.append(
                out[i][j] + out[i][j + 1]
            )  # Mathematical Combinatorial Formula.. simple
        temp.append(1)
        out.append(temp)

    return out


def pascal(n):
    """
    Use recursion to create the first n rows of Pascal's Triangle.

    Parameters:
        * n (int) : a positive integer, representing the number of rows in triangle.

    Returns:
        A list of length n, where the element at index i is a list of ints
        representing the ith row of Pascal's triangle.


    >>> pascal(1)
    [[1]]
    >>> pascal(2)
    [[1], [1, 1]]
    >>> pascal(3)
    [[1], [1, 1], [1, 2, 1]]
    >>> pascal(4)
    [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]
    >>> pascal(5)
    [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
    """
    out = []
    for i in range(1, n + 1):
        out.append(grow(i))
    return out


def grow(m):
    temp = [1]
    for i in range(1, m):
        temp.append((temp[i - 1] * (m - i) // i))

    return temp


def fill_n_cube(n):
    """
    Use recursion to make a cube of n layers of nested lists, where each layer is length n.
    The nth layer is an n-length list of zeroes.

    Parameters:
        * n (int) : a positive integer representing the length of each layer in the cube.

    Returns:
        A nested n-layer cube filled with zeroes.

    >>> fill_n_cube(1)
    [0]
    >>> fill_n_cube(2)
    [[0, 0], [0, 0]]
    >>> x = fill_n_cube(3)
    >>> x
    [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    >>> x[0][0][0] = 1  # make sure to avoid list aliasing!
    >>> x
    [[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Test with doctests. Helpful to debug individual practice.py functions.
    import doctest

    _doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    doctest.testmod(optionflags=_doctest_flags)  # runs ALL doctests
    # Alternatively, can run the doctests JUST for specified function/methods,
    # e.g., for pascal or any other function you might want.  To
    # do so, comment out the above line, and uncomment the below line of code.
    # This may be useful as you write/debug individual doctests or functions.
    # Also, the verbose flag can be set to True to see all test results,
    # including those that pass.
    #
    # doctest.run_docstring_examples(
    #    pascal,
    #    globals(),
    #    optionflags=_doctest_flags,
    #    verbose=False
    # )

    # Your code below:
