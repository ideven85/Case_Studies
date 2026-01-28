"""
6.101 recitation: minesweeper midpoint

Fill in the body of the linked list functions provided below.
For an extra challenge, try implementing the functions
both recursively and iteratively.

To test the functions, run the doctests in the
if __name__ == "__main__": block of code at the bottom of the file.
"""

import doctest
import sys

# import debug_recursion

# this is a very small recursion limit (normally 1000)
# but the test cases are tiny, setting a small recursion limit will help
# debug any infinite recursion errors

############################# Linked lists intro
# For this recitation empty linked list is None,
# otherwise length two tuple of (element, linked_list)

# Python list vs. linked-list
lst1 = [1, 2, 3, 4]
ll1 = (1, (2, (3, (4, None))))

lst2 = [3]
ll2 = (3, None)

lst3 = []
ll3 = None


def first(ll):
    """
    returns the first element of a non-empty linked list

    >>> first( (5, (10, (15, None))) )
    5
    >>> first((5,None))
    5
    """
    if not ll:
        return None
    return ll[0]


# @debug_recursion.show_recursive_structure
def rest(ll):
    """
    returns the rest of a nonempty linked list
    (omitting the first element)

    >>> rest( (5, (10, (15, None))) )
    (10, (15, None))
    >>> rest((5,None))
    None
    """
    if not ll:
        return None
    return first(ll[1:])


def ll_len(ll):
    """


    >>> ll_len( ('a',('b',('c',None)) ))
    2
    >>> ll_len(None)
    0
    >>> ll_len((1, (2, (3, (4, None)))))
    2
    """
    if ll is None:
        return 0
    if ll[1]:
        return 1 + ll_len(ll[1])
    else:
        return 0


def ll_get(ll, i):
    """
    get the ith element of a linked list

    >>> ll_get( ('a',('b',('c',None))), 2)
    'c'
    >>> ll_get( ('a',('b',('c',None))), 4)
    'b'
    """
    # list1 = list(ll_elements(ll))
    # return list1[i]
    if ll is None:
        return None

    if i == 0:
        return ll[i]
    return ll_get(ll[1], i - 1)


# @debug_recursion.show_recursive_structure
def make_ll(*elements):
    """
    given an arbitrary number of elements as arguments,
    make a linked-list of (first,rest) pairs


    >>> make_ll(1,2,3,4)
    (1, (2, (3, (4, None))))
    """
    if not elements:
        return None
    first = elements[0]
    rest = elements[1:]

    if not rest:
        return first, None
    return first, make_ll(*rest)


def ll_elements(ll):
    """
    return a generator that yields each element in a linked list
    >>> ll_gen = ll_elements(make_ll(1, 2, 3,4))
    >>> next(ll_gen)
    1
    >>> next(ll_gen)
    2
    >>> list(ll_gen)
    [3, 4]
    """
    if not ll:
        return
    first = ll[0]
    rest = ll[1:]
    # print(first,*rest)

    yield first
    yield from ll_elements(*rest)


def ll_plus(ll1, ll2):
    """
    return the concatenation of two linked lists

    >>> ll_plus(make_ll(1), make_ll(2,3))
    (1, (2, (3, None)))
    >>> ll_plus(None, make_ll(4,5))
    (4, (5, None))
    >>> ll_plus(make_ll(2, 3), make_ll(5, 4))
    (2, (3, (5, (4, None))))
    """
    return make_ll(ll1, ll2)


def ll_reverse(ll):
    """
    return the reverse of a linked list

    >>> ll_reverse(make_ll(1,2,3))
    (3, (2, (1, None)))
    """
    # Cheap Trick
    l = list(ll_elements(ll))
    l = l[::-1]

    return make_ll(*l)


if __name__ == "__main__":
    # _doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    # doctest.testmod(optionflags=_doctest_flags, verbose=True)  # runs ALL doctests

    # Alternatively, can run the doctests JUST for specified function/methods,
    # e.g., for render_2d_locations or any other function you might want.  To
    # do so, comment out the above line, and uncomment the below line of code.
    # This may be useful as you write/debug individual doctests or functions.
    # Also, the verbose flag can be set to True to see all test results,
    # including those that pass.
    #
    # doctest.run_docstring_examples(
    #    ll_elements,
    #    globals(),
    #    optionflags=_doctest_flags,
    #    verbose=False
    # )
    ll = (1, (2, None))

    x = make_ll(1, 2, 3)
    print(x)
